"""
Subpaquete de Procesamiento de Lenguaje Natural (NLP) para SaborIA.
Incluye módulos de normalización, tokenización, vectorización TF-IDF y comparación de n-gramas.
Nombres principales estandarizados en español con alias en inglés para compatibilidad.
"""
from .normalizer import (
    NormalizadorTexto,
    normalizar_resena,
    MAPA_SENTIMIENTO_EMOJIS,
    TextNormalizer,
    normalize_review,
    EMOJI_SENTIMENT_MAP,
)
from .tokenizer import (
    TokenizadorResenas,
    tokenizar_resena,
    COMPUESTOS_GASTRONOMICOS,
    MAPA_CONTRACCIONES,
    ReviewTokenizer,
    tokenize_review,
    GASTRONOMIC_COMPOUNDS,
    CONTRACTIONS_MAP,
)
from .vectorizer import (
    VectorizadorResenas,
    preprocesador_personalizado,
    tokenizador_personalizado,
    ReviewVectorizer,
    custom_preprocessor,
    custom_tokenizer,
)
from .ngram_comparator import (
    ComparadorNgramas,
    NgramComparator,
)
from .pipeline import (
    PipelineNLP,
    pipeline_nlp_default,
    preprocesar_resena_completa,
)

__all__ = [
    # Nombres principales en español
    "NormalizadorTexto",
    "normalizar_resena",
    "MAPA_SENTIMIENTO_EMOJIS",
    "TokenizadorResenas",
    "tokenizar_resena",
    "COMPUESTOS_GASTRONOMICOS",
    "MAPA_CONTRACCIONES",
    "VectorizadorResenas",
    "preprocesador_personalizado",
    "tokenizador_personalizado",
    "ComparadorNgramas",
    "PipelineNLP",
    "pipeline_nlp_default",
    "preprocesar_resena_completa",
    # Alias retrocompatibles en inglés
    "TextNormalizer",
    "normalize_review",
    "EMOJI_SENTIMENT_MAP",
    "ReviewTokenizer",
    "tokenize_review",
    "GASTRONOMIC_COMPOUNDS",
    "CONTRACTIONS_MAP",
    "ReviewVectorizer",
    "custom_preprocessor",
    "custom_tokenizer",
    "NgramComparator",
]
