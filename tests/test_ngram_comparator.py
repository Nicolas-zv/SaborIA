"""
Pruebas unitarias para el módulo NgramComparator.
"""
import pytest
from src.nlp.ngram_comparator import NgramComparator


def test_ngram_comparison_metrics():
    comparator = NgramComparator()
    results = comparator.compare()

    assert "unigramas" in results
    assert "unigramas_y_bigramas" in results
    assert "conclusion_tecnica" in results

    unigram_res = results["unigramas"]
    bigram_res = results["unigramas_y_bigramas"]

    # El vocabulario con bigramas debe ser estrictamente mayor que el de solo unigramas
    assert bigram_res["tamano_vocabulario"] > unigram_res["tamano_vocabulario"]
    assert bigram_res["bigramas_extraidos_total"] > 0
    assert len(bigram_res["bigramas_de_sentimiento_capturados"]) > 0


def test_custom_corpus_comparison():
    custom = [
        "buena pizza pero mala atencion",
        "excelente postre y buen trato",
    ]
    comparator = NgramComparator(sample_corpus=custom)
    results = comparator.compare()
    assert results["unigramas"]["tamano_vocabulario"] > 0
    assert results["unigramas_y_bigramas"]["tamano_vocabulario"] > results["unigramas"]["tamano_vocabulario"]
