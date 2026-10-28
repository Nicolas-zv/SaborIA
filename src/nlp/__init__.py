"""
Subpaquete de Procesamiento de Lenguaje Natural (NLP) para SaborIA.
Incluye módulos de normalización, tokenización, stop words, lematización y vectorización.
"""
from .normalizer import TextNormalizer, normalize_review

__all__ = ["TextNormalizer", "normalize_review"]
