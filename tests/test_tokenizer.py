"""
Pruebas unitarias para el módulo ReviewTokenizer.
"""
import pytest
from src.nlp.tokenizer import ReviewTokenizer, tokenize_review


def test_basic_tokenization():
    text = "la comida estuvo deliciosa y el servicio fue rapido"
    tokens = tokenize_review(text)
    assert tokens == ["la", "comida", "estuvo", "deliciosa", "y", "el", "servicio", "fue", "rapido"]


def test_contractions_expansion():
    text = "fuimos al restaurante del centro"
    tokenizer = ReviewTokenizer(expand_contractions=True)
    tokens = tokenizer.tokenize(text)
    # 'al' -> 'a', 'el'
    # 'del' -> 'de', 'el'
    assert "a" in tokens
    assert "de" in tokens
    assert "el" in tokens
    assert "al" not in tokens
    assert "del" not in tokens


def test_gastronomic_compounds():
    text = "pedimos comida tex-mex y hamburguesas fast-food gluten-free"
    tokens = tokenize_review(text)
    assert "tex-mex" in tokens
    assert "fast-food" in tokens
    assert "gluten-free" in tokens


def test_emojis_and_emphasis():
    text = "el postre estuvo increible !! emoji_delicioso emoji_amor"
    tokens = tokenize_review(text)
    assert "!!" in tokens
    assert "emoji_delicioso" in tokens
    assert "emoji_amor" in tokens


def test_explain_difficult_cases():
    text = "fuimos al restaurante tex-mex del centro y estuvo genial !!"
    tokenizer = ReviewTokenizer()
    analysis = tokenizer.explain_difficult_cases(text)
    assert len(analysis["contracciones_detectadas"]) >= 2
    assert "tex-mex" in analysis["terminos_compuestos"]
    assert "!!" in analysis["emojis_o_enfasis"]


def test_scikit_learn_transformer_compatibility():
    texts = [
        "excelente comida al momento",
        "pesima atencion del mesero",
    ]
    transformer = ReviewTokenizer()
    transformed = transformer.fit_transform(texts)
    assert len(transformed) == 2
    assert isinstance(transformed[0], list)
    assert "excelente" in transformed[0]
