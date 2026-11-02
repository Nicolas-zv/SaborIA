"""
Módulo del Pipeline Completo de Procesamiento de Lenguaje Natural (NLP) — SaborIA.

Integra en una secuencia estandarizada y reproducible:
1. Normalización (minúsculas, emojis gastronómicos, urls, signos de exclamación).
2. Tokenización especializada (compuestos gastronómicos, contracciones, puntuación repetida).
3. Filtrado de palabras vacías (stop words) conservando negaciones críticas (no, ni, nunca, sin, tampoco).
4. Lematización con spaCy o Stemming con SnowballStemmer (configurable con flag).
5. Desglose paso a paso para el backend (/preprocess) y el frontend interactivo.
"""
from typing import Any, Dict, List, Literal, Optional
import spacy
from nltk.stem.snowball import SnowballStemmer

from .normalizer import normalizar_resena
from .tokenizer import tokenizar_resena
from ..stopwords_es import STOPWORDS_PERSONALIZADAS

# Cargar modelo de spaCy para lematización en español
try:
    nlp_spacy = spacy.load("es_core_news_sm", disable=["parser", "ner"])
except Exception:
    import spacy.cli
    spacy.cli.download("es_core_news_sm")
    nlp_spacy = spacy.load("es_core_news_sm", disable=["parser", "ner"])

# Stemmer Snowball en español
stemmer_es = SnowballStemmer("spanish")


class PipelineNLP:
    """
    Pipeline integral de NLP para SaborIA con soporte para desglose paso a paso
    y selección configurable entre lematización y stemming.
    """

    def __init__(
        self,
        metodo_reduccion: Literal["lematizacion", "stemming", "ninguno"] = "lematizacion",
        conservar_negaciones: bool = True,
    ):
        self.metodo_reduccion = metodo_reduccion
        self.conservar_negaciones = conservar_negaciones
        self.stopwords = STOPWORDS_PERSONALIZADAS if conservar_negaciones else set()

    def normalizar(self, texto: str) -> str:
        """Aplica la normalización robusta de texto de SaborIA."""
        return normalizar_resena(texto)

    def tokenizar(self, texto: str) -> List[str]:
        """Aplica la tokenización considerando modismos gastronómicos y contracciones."""
        return tokenizar_resena(texto)

    def filtrar_stopwords(self, tokens: List[str]) -> List[str]:
        """Filtra palabras vacías conservando términos clave como negaciones."""
        return [
            t for t in tokens
            if t.lower() not in self.stopwords and len(t.strip()) > 0
        ]

    def reducir_tokens(self, tokens: List[str]) -> List[str]:
        """Aplica lematización o stemming a la lista de tokens según la configuración."""
        if self.metodo_reduccion == "stemming":
            return [stemmer_es.stem(t.lower()) for t in tokens]
        elif self.metodo_reduccion == "lematizacion":
            doc = nlp_spacy(" ".join(tokens))
            return [t.lemma_.lower() for t in doc if t.text.strip()]
        return tokens

    def procesar(self, texto: str) -> str:
        """
        Ejecuta todo el pipeline y devuelve el texto preprocesado como cadena
        unificada, ideal para alimentar el Vectorizador TF-IDF.
        """
        if not texto or not isinstance(texto, str):
            return ""

        texto_norm = self.normalizar(texto)
        tokens = self.tokenizar(texto_norm)
        tokens_sin_sw = self.filtrar_stopwords(tokens)
        tokens_reducidos = self.reducir_tokens(tokens_sin_sw)
        return " ".join(tokens_reducidos)

    def procesar_paso_a_paso(self, texto: str) -> Dict[str, Any]:
        """
        Ejecuta el pipeline exponiendo cada etapa intermedia.
        Esencial para el endpoint /preprocess de la API REST y el visualizador del Frontend.
        """
        if not isinstance(texto, str):
            texto = ""

        # Paso 1: Normalización
        texto_normalizado = self.normalizar(texto)

        # Paso 2: Tokenización
        tokens = self.tokenizar(texto_normalizado)

        # Paso 3: Filtrado de stop words
        tokens_sin_stopwords = self.filtrar_stopwords(tokens)

        # Paso 4: Reducción morfológica (Lematización / Stemming)
        tokens_reducidos = self.reducir_tokens(tokens_sin_stopwords)

        # Paso 5: Texto final ensamblado
        texto_final = " ".join(tokens_reducidos)

        return {
            "texto_original": texto,
            "paso_1_normalizacion": texto_normalizado,
            "paso_2_tokenizacion": tokens,
            "paso_3_sin_stopwords": tokens_sin_stopwords,
            "paso_4_reduccion": {
                "metodo": self.metodo_reduccion,
                "tokens": tokens_reducidos,
            },
            "paso_5_texto_final": texto_final,
            "estadisticas": {
                "longitud_original": len(texto),
                "total_tokens_iniciales": len(tokens),
                "total_tokens_filtrados": len(tokens_sin_stopwords),
                "reduccion_porcentual": (
                    round((1 - len(tokens_sin_stopwords) / max(len(tokens), 1)) * 100, 2)
                    if tokens else 0.0
                ),
            },
        }


# Instancia global lista para uso directo
pipeline_nlp_default = PipelineNLP(metodo_reduccion="lematizacion")


def preprocesar_resena_completa(texto: str) -> str:
    """Función rápida para preprocesar una reseña de inicio a fin."""
    return pipeline_nlp_default.procesar(texto)
