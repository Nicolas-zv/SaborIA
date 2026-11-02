"""
Pruebas unitarias para el módulo TextNormalizer.
"""
import pytest
from src.nlp.normalizer import TextNormalizer, normalize_review


def test_basic_lowercasing_and_accents():
    text = "¡Excelente COMIDA! El salmón estuvo PÉSIMO y carísimo."
    norm = normalize_review(text)
    assert "excelente" in norm
    assert "salmon" in norm
    assert "pesimo" in norm
    assert "carisimo" in norm


def test_emoji_mapping():
    text = "La pizza estuvo deliciosa 😋🤤 pero la atención fue 🤮"
    norm = normalize_review(text)
    assert "emoji_delicioso" in norm
    assert "emoji_vomito" in norm


def test_url_and_mention_removal():
    text = "Visita https://mirestaurante.com o sigue a @chef_juan para más info!"
    norm = normalize_review(text)
    assert "https" not in norm
    assert "chef_juan" not in norm
    assert "mirestaurante" not in norm


def test_repeated_punctuation():
    text = "¡¡¡El postre estuvo genial!!!!!! ¿¿¿Volveré???"
    norm = normalize_review(text)
    assert "genial !!" in norm
    assert "volvere ??" in norm


def test_sklearn_transformer_compatibility():
    texts = [
        "Plato muy rico 😍",
        "Pésimo servicio 😡",
    ]
    transformer = TextNormalizer()
    transformed = transformer.fit_transform(texts)
    assert len(transformed) == 2
    assert "emoji_encanto" in transformed[0]
    assert "emoji_enojo" in transformed[1]


def test_explain_steps():
    text = "¡¡INCREÍBLE sabor 😋!! Visita http://test.com"
    normalizer = TextNormalizer()
    explanation = normalizer.explain(text)
    assert "original" in explanation
    assert "emojis_mapeados" in explanation
    assert "sin_urls_menciones" in explanation
    assert "minusculas" in explanation
    assert "sin_tildes" in explanation
    assert "resultado_final" in explanation
