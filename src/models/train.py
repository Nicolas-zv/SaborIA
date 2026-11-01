"""
Script Reproducible de Entrenamiento y Evaluación — SaborIA.

Ejecutable mediante:
    python -m src.models.train

Realiza de punta a punta:
1. Carga del dataset de restaurantes en data/resenas_restaurantes.csv.
2. Limpieza y validación de datos.
3. Partición estratificada fija (80% train / 20% test con random_state=42).
4. Ajuste de hiperparámetros con GridSearchCV y validación cruzada estratificada (5-folds).
5. Evaluación en conjunto de prueba aislado (Accuracy, Precision, Recall, F1 macro/weighted).
6. Comparación contra Baseline Dummy (most_frequent).
7. Análisis cualitativo de errores (falsos positivos/negativos).
8. Serialización del artefacto final en models/regresion_logistica.joblib.
9. Exportación de métricas a models/metricas_regresion_logistica.json.
"""
import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from .logistic_regression import ClasificadorRegresionLogistica


def entrenar_modelo_completo():
    """Ejecuta el pipeline completo de entrenamiento, evaluación y serialización."""
    ruta_raiz = Path(__file__).resolve().parent.parent.parent
    ruta_datos = ruta_raiz / "data" / "resenas_restaurantes.csv"
    ruta_salida_modelo = ruta_raiz / "models" / "regresion_logistica.joblib"
    ruta_salida_metricas = ruta_raiz / "models" / "metricas_regresion_logistica.json"
    ruta_salida_errores = ruta_raiz / "models" / "analisis_errores.json"

    print("=" * 70)
    print("  SABORIA: PIPELINE REPRODUCIBLE DE ENTRENAMIENTO (REGRESIÓN LOGÍSTICA)")
    print("=" * 70)

    # 1. Cargar datos
    if not ruta_datos.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {ruta_datos}")

    print(f"\n[1/6] Cargando datos desde {ruta_datos.name}...")
    df = pd.read_csv(ruta_datos)
    print(f"      Total de filas iniciales: {len(df)}")

    # Validar campos obligatorios
    df = df.dropna(subset=["texto", "sentimiento"])
    df["texto"] = df["texto"].astype(str).str.strip()
    df = df[df["texto"].str.len() > 0]
    print(f"      Total de filas limpias: {len(df)}")

    print("\n      Distribución de clases:")
    conteo = df["sentimiento"].value_counts()
    for clase, cant in conteo.items():
        pct = (cant / len(df)) * 100
        print(f"      - {clase:10s}: {cant:4d} ({pct:.1f}%)")

    # 2. Split estratificado
    print("\n[2/6] Dividiendo datos (80% train / 20% test, estratificado, seed=42)...")
    X = df["texto"].tolist()
    y = df["sentimiento"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"      Muestras de entrenamiento: {len(X_train)}")
    print(f"      Muestras de prueba aislada: {len(X_test)}")

    # 3. GridSearchCV
    print("\n[3/6] Optimizando hiperparámetros con GridSearchCV (5-folds, StratifiedKFold)...")
    print("      Espacio de búsqueda: C=[0.1, 1.0, 5.0, 10.0], n-gramas=[(1,1), (1,2)], class_weight=['balanced', None]")
    
    clasificador = ClasificadorRegresionLogistica(random_state=42)
    res_cv = clasificador.entrenar_con_grid_search(X_train, y_train, cv_folds=5)
    
    print("\n      ¡Optimización finalizada!")
    print(f"      Mejores parámetros encontrados:")
    for k, v in res_cv["mejores_parametros"].items():
        print(f"        • {k}: {v}")
    print(f"      F1-Score ponderado en CV (validación cruzada): {res_cv['mejor_score_cv_f1_weighted']}")

    # 4. Evaluación en conjunto de prueba
    print("\n[4/6] Evaluando en conjunto de prueba independiente (test set no visto)...")
    metricas = clasificador.evaluar(X_test, y_test)
    
    print("      " + "-" * 55)
    print(f"      Exactitud (Accuracy):        {metricas['accuracy'] * 100:.2f}%")
    print(f"      Precisión (Macro):           {metricas['precision_macro'] * 100:.2f}%")
    print(f"      Precisión (Ponderada):       {metricas['precision_weighted'] * 100:.2f}%")
    print(f"      Recall (Sensibilidad Macro): {metricas['recall_macro'] * 100:.2f}%")
    print(f"      Recall (Ponderado):          {metricas['recall_weighted'] * 100:.2f}%")
    print(f"      F1-Score (Macro):            {metricas['f1_macro'] * 100:.2f}%")
    print(f"      F1-Score (Ponderado):        {metricas['f1_weighted'] * 100:.2f}%")
    print("      " + "-" * 55)

    print("\n      Matriz de Confusión:")
    print(f"      Clases: {metricas['clases']}")
    for fila in metricas["matriz_confusion"]:
        print(f"        {fila}")

    # 5. Comparativa con Baseline Dummy
    print("\n[5/6] Comparando contra Baseline Dummy (Most Frequent)...")
    comparativa = clasificador.comparar_con_baseline(X_train, y_train, X_test, y_test)
    print(f"      Línea Base Dummy - Accuracy: {comparativa['baseline_dummy']['accuracy'] * 100:.2f}% | F1: {comparativa['baseline_dummy']['f1_weighted'] * 100:.2f}%")
    print(f"      Regresión Log.   - Accuracy: {comparativa['regresion_logistica']['accuracy'] * 100:.2f}% | F1: {comparativa['regresion_logistica']['f1_weighted'] * 100:.2f}%")
    print(f"      Mejora sobre Dummy: +{comparativa['mejora_accuracy_porcentual']:.2f}% en Accuracy (+{comparativa['mejora_f1_porcentual']:.2f}% en F1)")

    # Análisis de errores cualitativo
    errores = clasificador.analizar_errores(X_test, y_test, max_errores=6)
    print(f"\n      Análisis cualitativo de fallos ({len(errores)} ejemplos detectados):")
    for err in errores[:3]:
        print(f"      • Real: '{err['etiqueta_real']}' | Pred: '{err['etiqueta_predicha']}' (Conf: {err['confianza_predicha']*100:.1f}%)")
        print(f"        Reseña: \"{err['texto_resena'][:80]}...\"")

    # 6. Serialización a disco
    print("\n[6/6] Serializando artefactos a disco...")
    clasificador.guardar(str(ruta_salida_modelo))
    print(f"      Modelo guardado en: {ruta_salida_modelo}")

    datos_exportacion = {
        "modelo": "Regresión Logística (TF-IDF)",
        "autor": "Roberto (@Roberto)",
        "hiperparametros": res_cv["mejores_parametros"],
        "score_cv_f1_weighted": res_cv["mejor_score_cv_f1_weighted"],
        "metricas_test": metricas,
        "comparativa_baseline": comparativa,
        "clases": clasificador.clases_,
    }
    with open(ruta_salida_metricas, "w", encoding="utf-8") as f:
        json.dump(datos_exportacion, f, indent=2, ensure_ascii=False)
    print(f"      Métricas guardadas en: {ruta_salida_metricas}")

    with open(ruta_salida_errores, "w", encoding="utf-8") as f:
        json.dump(errores, f, indent=2, ensure_ascii=False)
    print(f"      Análisis de errores guardado en: {ruta_salida_errores}")

    print("\n" + "=" * 70)
    print("  ¡ENTRENAMIENTO Y SERIALIZACIÓN EXITOSA!")
    print("=" * 70)
    return clasificador, datos_exportacion


if __name__ == "__main__":
    entrenar_modelo_completo()
