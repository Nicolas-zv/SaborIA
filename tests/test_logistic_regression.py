"""
Pruebas unitarias para el módulo de Regresión Logística de SaborIA.
"""
import os
import tempfile
import pytest
from src.models.logistic_regression import ClasificadorRegresionLogistica


@pytest.fixture
def datos_pequenos():
    X = [
        "La comida estuvo deliciosa y el servicio fue impecable, volveremos pronto.",
        "Experiencia gourmet maravillosa, todo perfecto y muy sabroso.",
        "Pésimo servicio, la comida llegó fría y los camareros fueron groseros.",
        "Horrible atención, comida cruda y ni una disculpa, no lo recomiendo.",
        "El lugar es normal, platos aceptables sin destacar, precios regulares.",
        "Comida estándar, nada especial pero cumple para una comida rápida.",
    ]
    y = [
        "positivo",
        "positivo",
        "negativo",
        "negativo",
        "neutral",
        "neutral",
    ]
    return X, y


def test_entrenar_directo_y_predecir(datos_pequenos):
    X, y = datos_pequenos
    modelo = ClasificadorRegresionLogistica(min_df=1)
    modelo.entrenar_directo(X, y)

    predicciones = modelo.predecir(["deliciosa comida gourmet", "Pésimo servicio y comida"])
    assert len(predicciones) == 2
    assert predicciones[0] == "positivo"
    assert predicciones[1] == "negativo"


def test_predecir_probabilidades(datos_pequenos):
    X, y = datos_pequenos
    modelo = ClasificadorRegresionLogistica(min_df=1)
    modelo.entrenar_directo(X, y)

    probs = modelo.predecir_probabilidades(["Muy rico todo"])
    assert len(probs) == 1
    dist = probs[0]
    assert "positivo" in dist
    assert "negativo" in dist
    assert "neutral" in dist
    # La suma de probabilidades debe ser aproximadamente 1.0
    suma = sum(dist.values())
    assert 0.99 <= suma <= 1.01


def test_evaluar_metricas(datos_pequenos):
    X, y = datos_pequenos
    modelo = ClasificadorRegresionLogistica(min_df=1)
    modelo.entrenar_directo(X, y)

    res = modelo.evaluar(X, y)
    assert "accuracy" in res
    assert "f1_weighted" in res
    assert "matriz_confusion" in res
    assert "reporte_clasificacion" in res
    assert res["accuracy"] >= 0.0


def test_guardar_y_cargar_modelo(datos_pequenos):
    X, y = datos_pequenos
    modelo = ClasificadorRegresionLogistica(min_df=1)
    modelo.entrenar_directo(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        ruta = os.path.join(tmpdir, "test_modelo.joblib")
        modelo.guardar(ruta)
        assert os.path.exists(ruta)

        modelo_cargado = ClasificadorRegresionLogistica.cargar(ruta)
        pred_original = modelo.predecir(X[:2])
        pred_cargado = modelo_cargado.predecir(X[:2])
        assert pred_original == pred_cargado
