"""
Módulo de Clasificación de Sentimientos con Regresión Logística — SaborIA.

Implementa:
1. Pipeline integral (TF-IDF + LogisticRegression) serializable con joblib.
2. Optimización de hiperparámetros con GridSearchCV y validación cruzada estratificada.
3. Tratamiento del desbalance de clases (class_weight='balanced').
4. Comparación contra Baseline Dummy (clase más frecuente).
5. Evaluación exhaustiva de métricas (Accuracy, Precision, Recall, F1 macro/weighted, Matriz de Confusión).
6. Análisis cualitativo de errores (falsos positivos y negativos con probabilidades).
"""
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from ..nlp.normalizer import normalizar_resena
from ..nlp.tokenizer import tokenizar_resena


def _preprocesador(texto: str) -> str:
    """Preprocesador para TfidfVectorizer en el pipeline."""
    return normalizar_resena(texto)


def _tokenizador(texto: str) -> List[str]:
    """Tokenizador para TfidfVectorizer en el pipeline."""
    return tokenizar_resena(texto)


class ClasificadorRegresionLogistica:
    """
    Clasificador supervisado de sentimientos basado en Regresión Logística
    y representación TF-IDF para SaborIA.
    """

    def __init__(
        self,
        c_param: float = 1.0,
        class_weight: Optional[str] = "balanced",
        rango_ngramas: Tuple[int, int] = (1, 2),
        max_features: Optional[int] = 2500,
        min_df: int = 2,
        random_state: int = 42,
    ):
        self.c_param = c_param
        self.class_weight = class_weight
        self.rango_ngramas = rango_ngramas
        self.max_features = max_features
        self.min_df = min_df
        self.random_state = random_state

        self.pipeline: Optional[Pipeline] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.mejores_parametros_: Optional[Dict[str, Any]] = None
        self.clases_: Optional[List[str]] = None

    def _construir_pipeline_base(self) -> Pipeline:
        """Construye el pipeline de sklearn que conecta TF-IDF con Regresión Logística."""
        vectorizador = TfidfVectorizer(
            preprocessor=_preprocesador,
            tokenizer=_tokenizador,
            ngram_range=self.rango_ngramas,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=0.95,
            sublinear_tf=True,
            token_pattern=None,
        )
        clasificador = LogisticRegression(
            C=self.c_param,
            class_weight=self.class_weight,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        return Pipeline([
            ("vectorizador", vectorizador),
            ("clasificador", clasificador),
        ])

    def entrenar_con_grid_search(
        self,
        X_train: List[str],
        y_train: List[str],
        cv_folds: int = 5,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Optimiza hiperparámetros clave usando GridSearchCV con validación cruzada estratificada.
        Compara unigramas vs bigramas, regularización C y ponderación de clases.
        """
        if param_grid is None:
            param_grid = {
                "vectorizador__ngram_range": [(1, 1), (1, 2)],
                "vectorizador__max_features": [1500, 2500, None],
                "clasificador__C": [0.1, 1.0, 5.0, 10.0],
                "clasificador__class_weight": ["balanced", None],
            }

        pipeline_base = self._construir_pipeline_base()
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        self.grid_search = GridSearchCV(
            estimator=pipeline_base,
            param_grid=param_grid,
            cv=skf,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=0,
        )
        self.grid_search.fit(X_train, y_train)

        self.pipeline = self.grid_search.best_estimator_
        self.mejores_parametros_ = self.grid_search.best_params_
        self.clases_ = list(self.pipeline.classes_)

        return {
            "mejores_parametros": self.mejores_parametros_,
            "mejor_score_cv_f1_weighted": round(float(self.grid_search.best_score_), 4),
        }

    def entrenar_directo(self, X_train: List[str], y_train: List[str]):
        """Entrena directamente el pipeline con la configuración predeterminada."""
        self.pipeline = self._construir_pipeline_base()
        self.pipeline.fit(X_train, y_train)
        self.clases_ = list(self.pipeline.classes_)
        return self

    def predecir(self, X: List[str]) -> List[str]:
        """Predice la clase de sentimiento para una lista de textos."""
        if self.pipeline is None:
            raise RuntimeError("El modelo debe ser entrenado antes de predecir.")
        return list(self.pipeline.predict(X))

    def predecir_probabilidades(self, X: List[str]) -> List[Dict[str, float]]:
        """
        Devuelve la distribución de probabilidades por clase para cada texto.
        Útil para el backend (/predict) y la barra de confianza en el Frontend.
        """
        if self.pipeline is None:
            raise RuntimeError("El modelo debe ser entrenado antes de predecir probabilidades.")
        probs = self.pipeline.predict_proba(X)
        clases = self.pipeline.classes_

        resultados = []
        for prob_fila in probs:
            distribucion = {
                clase: round(float(prob_fila[i]), 4)
                for i, clase in enumerate(clases)
            }
            resultados.append(distribucion)
        return resultados

    def evaluar(self, X_test: List[str], y_test: List[str]) -> Dict[str, Any]:
        """
        Calcula exhaustivamente todas las métricas requeridas por la rúbrica
        en el conjunto de prueba aislado.
        """
        if self.pipeline is None:
            raise RuntimeError("El modelo debe ser entrenado antes de evaluar.")

        y_pred = self.pipeline.predict(X_test)
        clases = sorted(list(set(y_test) | set(y_pred)))

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_wei = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=clases).tolist()
        reporte_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        return {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_macro": round(rec_macro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_macro": round(f1_mac, 4),
            "f1_weighted": round(f1_wei, 4),
            "clases": clases,
            "matriz_confusion": cm,
            "reporte_clasificacion": reporte_dict,
            "total_muestras_test": len(y_test),
        }

    def comparar_con_baseline(
        self,
        X_train: List[str],
        y_train: List[str],
        X_test: List[str],
        y_test: List[str],
    ) -> Dict[str, Any]:
        """
        Compara el rendimiento de la Regresión Logística contra un DummyClassifier
        (línea base que predice la clase más frecuente) para demostrar el valor aportado.
        """
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)

        acc_dummy = float(accuracy_score(y_test, dummy_pred))
        f1_dummy = float(f1_score(y_test, dummy_pred, average="weighted", zero_division=0))

        metricas_modelo = self.evaluar(X_test, y_test)

        return {
            "baseline_dummy": {
                "estrategia": "most_frequent",
                "accuracy": round(acc_dummy, 4),
                "f1_weighted": round(f1_dummy, 4),
            },
            "regresion_logistica": {
                "accuracy": metricas_modelo["accuracy"],
                "f1_weighted": metricas_modelo["f1_weighted"],
            },
            "mejora_accuracy_porcentual": round((metricas_modelo["accuracy"] - acc_dummy) * 100, 2),
            "mejora_f1_porcentual": round((metricas_modelo["f1_weighted"] - f1_dummy) * 100, 2),
        }

    def analizar_errores(
        self,
        X_test: List[str],
        y_test: List[str],
        max_errores: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Identifica y estructura ejemplos específicos de falsos positivos y negativos
        para el análisis cualitativo exigido en la Capa 3 y la defensa individual.
        """
        if self.pipeline is None:
            raise RuntimeError("El modelo debe ser entrenado antes de analizar errores.")

        y_pred = self.pipeline.predict(X_test)
        probs = self.predecir_probabilidades(X_test)

        errores = []
        for i, (real, pred, texto) in enumerate(zip(y_test, y_pred, X_test)):
            if real != pred:
                errores.append({
                    "indice_test": i,
                    "texto_resena": texto,
                    "etiqueta_real": real,
                    "etiqueta_predicha": pred,
                    "confianza_predicha": probs[i].get(pred, 0.0),
                    "confianza_real": probs[i].get(real, 0.0),
                    "distribucion_probabilidades": probs[i],
                })
                if len(errores) >= max_errores:
                    break
        return errores

    def guardar(self, ruta_archivo: str):
        """Serializa el pipeline completo entrenado a disco con joblib."""
        if self.pipeline is None:
            raise RuntimeError("No hay pipeline entrenado para guardar.")
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        joblib.dump(self.pipeline, ruta_archivo)

    @classmethod
    def cargar(cls, ruta_archivo: str) -> "ClasificadorRegresionLogistica":
        """Carga un pipeline serializado desde disco listo para inferencia."""
        instancia = cls()
        instancia.pipeline = joblib.load(ruta_archivo)
        instancia.clases_ = list(instancia.pipeline.classes_)
        return instancia
