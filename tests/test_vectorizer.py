"""
Pruebas unitarias para el módulo ReviewVectorizer.
"""
import pytest
from src.nlp.vectorizer import ReviewVectorizer


@pytest.fixture
def sample_reviews():
    return [
        "La comida estuvo deliciosa y el ambiente muy agradable 😋",
        "Pésimo servicio y la comida fría, no vuelvo nunca 🤮",
        "Buena relación calidad precio en el menú ejecutivo.",
        "Muy rica la pizza pero la atención un poco lenta.",
        "Excelente restaurante, totalmente recomendado para ir en familia.",
    ]


def test_vectorizer_fit_transform(sample_reviews):
    vectorizer = ReviewVectorizer(min_df=1, max_features=100)
    matrix = vectorizer.fit_transform(sample_reviews)
    assert matrix.shape[0] == len(sample_reviews)
    assert vectorizer.vocabulary_size_ > 0


def test_top_features_extraction(sample_reviews):
    vectorizer = ReviewVectorizer(min_df=1)
    vectorizer.fit(sample_reviews)
    top_words = vectorizer.get_top_features_for_text(sample_reviews[0], top_n=3)
    assert len(top_words) > 0
    assert isinstance(top_words[0], tuple)
    assert isinstance(top_words[0][0], str)
    assert isinstance(top_words[0][1], float)


def test_vectorizer_summary_params():
    vectorizer = ReviewVectorizer(ngram_range=(1, 2), max_features=1500, min_df=3)
    summary = vectorizer.get_summary_params()
    assert summary["ngram_range"] == "1-2"
    assert summary["max_features"] == 1500
    assert summary["min_df"] == 3
