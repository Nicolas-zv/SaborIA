"""
Módulo de Vectorización TF-IDF para Reseñas Gastronómicas (SaborIA).
Implementa la representación vectorial numérica de texto basada en TF-IDF
(Term Frequency - Inverse Document Frequency) con parámetros documentados y justificados.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .normalizer import normalizar_resena
from .tokenizer import tokenizar_resena


def preprocesador_personalizado(texto: str) -> str:
    """Preprocesador integrado que ejecuta la normalización de SaborIA."""
    return normalizar_resena(texto)


def tokenizador_personalizado(texto: str) -> List[str]:
    """Tokenizador integrado que ejecuta el tokenizador de casos difíciles."""
    return tokenizar_resena(texto)


class VectorizadorResenas:
    """
    Vectorizador TF-IDF configurable y documentado para reseñas.

    Parámetros documentados y justificados técnicamente:
    --------------------------------------------------
    rango_ngramas : Tuple[int, int], default=(1, 1)
        Rango de n-gramas a extraer:
        - (1, 1): Solo unigramas (palabras individuales).
        - (1, 2): Unigramas y bigramas (captura pares clave como 'no recomiendo', 'muy bueno').
    max_caracteristicas : Optional[int], default=2500
        Tamaño máximo del vocabulario retenido (las N palabras/n-gramas con mayor TF-IDF).
        Evita la explosión dimensional y previene el sobreajuste (overfitting).
    frecuencia_minima_doc : int or float, default=2
        Frecuencia mínima de documento (min_df). Ignora términos en menos de 2 reseñas.
        Filtra erratas mecanográficas únicas o palabras ultra-raras sin valor estadístico.
    frecuencia_maxima_doc : float, default=0.95
        Frecuencia máxima de documento (max_df). Ignora términos en más del 95% de las reseñas.
        Filtra palabras omnipresentes del dominio (ej. 'restaurante', 'comida') que no discriminan.
    tf_sublineal : bool, default=True
        Aplica escala logarítmica: 1 + log(tf).
        Atenúa el impacto de palabras repetidas múltiples veces en una misma reseña.
    """

    def __init__(
        self,
        rango_ngramas: Tuple[int, int] = (1, 1),
        max_caracteristicas: Optional[int] = 2500,
        frecuencia_minima_doc: int = 2,
        frecuencia_maxima_doc: float = 0.95,
        tf_sublineal: bool = True,
        # Parámetros retrocompatibles en inglés
        ngram_range: Optional[Tuple[int, int]] = None,
        max_features: Optional[int] = None,
        min_df: Optional[int] = None,
        max_df: Optional[float] = None,
        sublinear_tf: Optional[bool] = None,
    ):
        self.rango_ngramas = ngram_range if ngram_range is not None else rango_ngramas
        self.max_caracteristicas = max_features if max_features is not None else max_caracteristicas
        self.frecuencia_minima_doc = min_df if min_df is not None else frecuencia_minima_doc
        self.frecuencia_maxima_doc = max_df if max_df is not None else frecuencia_maxima_doc
        self.tf_sublineal = sublinear_tf if sublinear_tf is not None else tf_sublineal

        # Instancia subyacente de TfidfVectorizer de Scikit-Learn
        self.vectorizador = TfidfVectorizer(
            preprocessor=preprocesador_personalizado,
            tokenizer=tokenizador_personalizado,
            ngram_range=self.rango_ngramas,
            max_features=self.max_caracteristicas,
            min_df=self.frecuencia_minima_doc,
            max_df=self.frecuencia_maxima_doc,
            sublinear_tf=self.tf_sublineal,
            token_pattern=None,
        )

    def fit(self, textos: List[str]):
        """Aprende el vocabulario y las frecuencias IDF a partir del corpus de entrenamiento."""
        self.vectorizador.fit(textos)
        return self

    def transform(self, textos: List[str]):
        """Transforma una lista de textos en una matriz dispersa TF-IDF."""
        return self.vectorizador.transform(textos)

    def fit_transform(self, textos: List[str]):
        """Aprende el vocabulario y devuelve la matriz dispersa TF-IDF."""
        return self.vectorizador.fit_transform(textos)

    @property
    def tamano_vocabulario_(self) -> int:
        """Devuelve el número de términos aprendidos en el vocabulario."""
        return len(self.vectorizador.vocabulary_)

    # Alias en inglés para tamano_vocabulario_
    vocabulary_size_ = tamano_vocabulario_

    def obtener_nombres_caracteristicas(self) -> List[str]:
        """Devuelve la lista ordenada de términos y n-gramas del vocabulario."""
        return list(self.vectorizador.get_feature_names_out())

    # Alias en inglés
    get_feature_names = obtener_nombres_caracteristicas

    def obtener_terminos_principales(self, texto: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Obtiene los términos con mayor peso TF-IDF para una reseña específica.
        Permite explicar por qué el modelo interpretó cierta polaridad.
        """
        tfidf_vec = self.transform([texto]).toarray()[0]
        nombres = self.obtener_nombres_caracteristicas()
        indices_top = np.argsort(tfidf_vec)[::-1][:top_n]
        return [(nombres[i], round(float(tfidf_vec[i]), 4)) for i in indices_top if tfidf_vec[i] > 0]

    # Alias en inglés
    get_top_features_for_text = obtener_terminos_principales

    def obtener_resumen_parametros(self) -> Dict[str, object]:
        """Devuelve un diccionario resumen de los parámetros configurados para documentación."""
        return {
            "rango_ngramas": f"{self.rango_ngramas[0]}-{self.rango_ngramas[1]}",
            "ngram_range": f"{self.rango_ngramas[0]}-{self.rango_ngramas[1]}",
            "max_caracteristicas": self.max_caracteristicas,
            "max_features": self.max_caracteristicas,
            "frecuencia_minima_doc": self.frecuencia_minima_doc,
            "min_df": self.frecuencia_minima_doc,
            "frecuencia_maxima_doc": self.frecuencia_maxima_doc,
            "max_df": self.frecuencia_maxima_doc,
            "tf_sublineal": self.tf_sublineal,
            "sublinear_tf": self.tf_sublineal,
            "vocabulario_aprendido": getattr(self, "tamano_vocabulario_", 0),
        }

    # Alias en inglés
    get_summary_params = obtener_resumen_parametros


# Alias para retrocompatibilidad
ReviewVectorizer = VectorizadorResenas
custom_preprocessor = preprocesador_personalizado
custom_tokenizer = tokenizador_personalizado
