"""
Módulo de Vectorización TF-IDF para Reseñas Gastronómicas (SaborIA).
Implementa la representación vectorial numérica de texto basada en TF-IDF
(Term Frequency - Inverse Document Frequency) con parámetros documentados y justificados.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .normalizer import normalize_review
from .tokenizer import tokenize_review


def custom_preprocessor(text: str) -> str:
    """Preprocesador integrado que ejecuta la normalización de SaborIA."""
    return normalize_review(text)


def custom_tokenizer(text: str) -> List[str]:
    """Tokenizador integrado que ejecuta el tokenizador de casos difíciles."""
    return tokenize_review(text)


class ReviewVectorizer:
    """
    Vectorizador TF-IDF configurable y documentado para reseñas.

    Parámetros documentados y justificados técnicamente:
    --------------------------------------------------
    ngram_range : Tuple[int, int], default=(1, 1)
        Rango de n-gramas a extraer:
        - (1, 1): Solo unigramas (palabras individuales).
        - (1, 2): Unigramas y bigramas (captura pares clave como 'no recomiendo', 'muy bueno').
    max_features : Optional[int], default=2500
        Tamaño máximo del vocabulario retenido (las N palabras/n-gramas con mayor TF-IDF).
        Evita la explosión dimensional y previene el sobreajuste (overfitting).
    min_df : int or float, default=2
        Frecuencia mínima de documento. Ignora términos que aparecen en menos de 2 reseñas.
        Filtra erratas mecanográficas únicas o palabras ultra-raras sin valor estadístico.
    max_df : float, default=0.95
        Frecuencia máxima de documento. Ignora términos que aparecen en más del 95% de las reseñas.
        Filtra palabras omnipresentes del dominio (ej. 'restaurante', 'comida') que no discriminan sentimiento.
    sublinear_tf : bool, default=True
        Aplica escala logarítmica: 1 + log(tf).
        Atenúa el impacto de palabras repetidas múltiples veces en una misma reseña.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = 2500,
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        # Instancia subyacente de TfidfVectorizer de Scikit-Learn
        self.vectorizer = TfidfVectorizer(
            preprocessor=custom_preprocessor,
            tokenizer=custom_tokenizer,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            token_pattern=None,  # Desactivado porque usamos custom_tokenizer
        )

    def fit(self, texts: List[str]):
        """Aprende el vocabulario y las frecuencias IDF a partir del corpus de entrenamiento."""
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]):
        """Transforma una lista de textos en una matriz dispersa TF-IDF."""
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        """Aprende el vocabulario y devuelve la matriz dispersa TF-IDF."""
        return self.vectorizer.fit_transform(texts)

    @property
    def vocabulary_size_(self) -> int:
        """Devuelve el número de términos aprendidos en el vocabulario."""
        return len(self.vectorizer.vocabulary_)

    def get_feature_names(self) -> List[str]:
        """Devuelve la lista ordenada de términos y n-gramas del vocabulario."""
        return list(self.vectorizer.get_feature_names_out())

    def get_top_features_for_text(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Obtiene los términos con mayor peso TF-IDF para una reseña específica.
        Permite explicar por qué el modelo interpretó cierta polaridad.
        """
        tfidf_vec = self.transform([text]).toarray()[0]
        feature_names = self.get_feature_names()
        top_indices = np.argsort(tfidf_vec)[::-1][:top_n]
        return [(feature_names[i], round(float(tfidf_vec[i]), 4)) for i in top_indices if tfidf_vec[i] > 0]

    def get_summary_params(self) -> Dict[str, object]:
        """Devuelve un diccionario resumen de los parámetros configurados para documentación."""
        return {
            "ngram_range": f"{self.ngram_range[0]}-{self.ngram_range[1]}",
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "vocabulario_aprendido": getattr(self, "vocabulary_size_", 0),
        }
