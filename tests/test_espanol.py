"""
Pruebas unitarias para validar las interfaces nativas en español de SaborIA.
"""
import pytest
from src.nlp import (
    NormalizadorTexto,
    normalizar_resena,
    TokenizadorResenas,
    tokenizar_resena,
    VectorizadorResenas,
    ComparadorNgramas,
)


def test_interfaz_espanol_normalizador():
    normalizador = NormalizadorTexto()
    texto = "¡¡Excelente comida 😋 en https://restaurante.com!!"
    resultado = normalizador.normalizar(texto)
    assert "excelente" in resultado
    assert "emoji_delicioso" in resultado
    assert "https" not in resultado

    pasos = normalizador.explicar_pasos(texto)
    assert "original" in pasos
    assert "resultado_final" in pasos


def test_interfaz_espanol_tokenizador():
    texto = "comimos comida tex-mex del restaurante !!"
    tokens = tokenizar_resena(texto)
    assert "tex-mex" in tokens
    assert "de" in tokens
    assert "el" in tokens
    assert "del" not in tokens

    tokenizador = TokenizadorResenas()
    analisis = tokenizador.explicar_casos_dificiles(texto)
    assert len(analisis["contracciones_detectadas"]) > 0


def test_interfaz_espanol_vectorizador_y_ngramas():
    textos = [
        "buena comida y excelente atencion",
        "pesimo servicio no lo recomiendo",
    ]
    vectorizador = VectorizadorResenas(min_df=1)
    vectorizador.fit(textos)
    assert vectorizador.tamano_vocabulario_ > 0
    assert len(vectorizador.obtener_nombres_caracteristicas()) > 0

    comparador = ComparadorNgramas(sample_corpus=textos)
    res = comparador.comparar()
    assert res["unigramas"]["tamano_vocabulario"] > 0
