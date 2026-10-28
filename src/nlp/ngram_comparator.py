"""
Módulo de Comparación Experimental de N-gramas (SaborIA).
Compara formalmente unigramas (1, 1) frente a unigramas + bigramas (1, 2)
evaluando tamaño del vocabulario, esparcimiento y captura de modificadores de sentimiento.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from .vectorizer import VectorizadorResenas


class ComparadorNgramas:
    """
    Compara experimentalmente el efecto de usar unigramas vs. unigramas + bigramas en reseñas.
    """

    def __init__(self, corpus_muestra: Optional[List[str]] = None, sample_corpus: Optional[List[str]] = None):
        self.corpus = corpus_muestra or sample_corpus or [
            "La comida estuvo muy buena y el personal fue sumamente amable 😋",
            "No recomiendo este lugar, la atención fue pésima y la comida fría 🤮",
            "Excelente relación calidad precio, volvería sin dudarlo.",
            "Nada rico el plato principal y el precio demasiado caro.",
            "Muy buen ambiente pero servicio un poco lento.",
            "No me gustó la salsa, estaba demasiado salada.",
            "Totalmente recomendado, el mejor restaurante tex-mex de la zona.",
            "Una decepción total, no volveremos jamás.",
        ]

    def comparar(self) -> Dict[str, object]:
        """
        Ejecuta la comparación estadística entre:
        - Configuración A: Solo unigramas (1, 1)
        - Configuración B: Unigramas + Bigramas (1, 2)
        """
        # Vectorizador A: Unigramas
        vec_unigram = VectorizadorResenas(rango_ngramas=(1, 1), frecuencia_minima_doc=1, max_caracteristicas=None)
        matrix_unigram = vec_unigram.fit_transform(self.corpus)
        vocab_unigram = vec_unigram.obtener_nombres_caracteristicas()

        # Vectorizador B: Unigramas + Bigramas
        vec_bigram = VectorizadorResenas(rango_ngramas=(1, 2), frecuencia_minima_doc=1, max_caracteristicas=None)
        matrix_bigram = vec_bigram.fit_transform(self.corpus)
        vocab_bigram = vec_bigram.obtener_nombres_caracteristicas()

        # Extraer solo los bigramas generados (aquellos términos con espacio)
        extracted_bigrams = [term for term in vocab_bigram if " " in term]

        # Calcular esparcimiento (sparsity = % de ceros en la matriz)
        sparsity_unigram = 1.0 - (matrix_unigram.nnz / (matrix_unigram.shape[0] * matrix_unigram.shape[1]))
        sparsity_bigram = 1.0 - (matrix_bigram.nnz / (matrix_bigram.shape[0] * matrix_bigram.shape[1]))

        # Identificar bigramas clave de polaridad (pares modificador + adjetivo)
        sentiment_bigrams = [
            b for b in extracted_bigrams
            if any(k in b for k in ["muy", "no", "nada", "poco", "buen", "excelente", "pesim", "caro", "recomiend"])
        ]

        return {
            "unigramas": {
                "rango": "(1, 1)",
                "tamano_vocabulario": len(vocab_unigram),
                "esparcimiento_pct": round(sparsity_unigram * 100, 2),
                "ejemplos_vocabulario": vocab_unigram[:8],
            },
            "unigramas_y_bigramas": {
                "rango": "(1, 2)",
                "tamano_vocabulario": len(vocab_bigram),
                "bigramas_extraidos_total": len(extracted_bigrams),
                "esparcimiento_pct": round(sparsity_bigram * 100, 2),
                "bigramas_de_sentimiento_capturados": sentiment_bigrams[:10],
            },
            "conclusion_tecnica": (
                "Los unigramas producen matrices más densas y compactas, pero son ciegos a la negación "
                "y a la intensidad (ej. confunden 'recomiendo' con 'no recomiendo'). "
                "La combinación (1, 2) expande el vocabulario pero captura pares semánticos decisivos "
                "para clasificar sentimientos con alta precisión."
            ),
        }

    # Alias en inglés
    compare = comparar


# Alias para retrocompatibilidad
NgramComparator = ComparadorNgramas
