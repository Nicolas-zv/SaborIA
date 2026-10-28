"""
Subpaquete de Procesamiento de Lenguaje Natural (NLP) para SaborIA.
Incluye módulos de normalización, tokenización, vectorización TF-IDF y experimentos de n-gramas.
"""
from .normalizer import TextNormalizer, normalize_review
from .tokenizer import ReviewTokenizer, tokenize_review
from .vectorizer import ReviewVectorizer

__all__ = [
    "TextNormalizer",
    "normalize_review",
    "ReviewTokenizer",
    "tokenize_review",
    "ReviewVectorizer",
]
