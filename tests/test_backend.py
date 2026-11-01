"""
Pruebas automatizadas con pytest para la API REST de FastAPI (SaborIA).
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

cliente = TestClient(app)


def test_endpoint_health():
    response = cliente.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["estado"] == "operativo"
    assert "modelo_cargado" in data


def test_endpoint_predict_positivo():
    payload = {"texto": "La comida estuvo exquisita y los postres simplemente increíbles. 😋"}
    response = cliente.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentimiento_predicho"] in ["positivo", "negativo", "neutral"]
    assert "confianza" in data
    assert "probabilidades" in data
    assert "positivo" in data["probabilidades"]


def test_endpoint_predict_negativo():
    payload = {"texto": "Pésima atención, tardaron dos horas y la comida estaba fría y cruda."}
    response = cliente.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentimiento_predicho"] in ["negativo", "neutral", "positivo"]
    assert data["confianza"] > 0


def test_endpoint_predict_validacion_texto_vacio():
    payload = {"texto": " "}
    response = cliente.post("/predict", json=payload)
    assert response.status_code in [400, 422]


def test_endpoint_preprocess_paso_a_paso():
    payload = {"texto": "¡Excelente vino y atención rápida, no duden en venir! 😍"}
    response = cliente.post("/preprocess", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "paso_1_normalizacion" in data
    assert "paso_2_tokenizacion" in data
    assert "paso_3_sin_stopwords" in data
    assert "paso_4_reduccion" in data
    assert "paso_5_texto_final" in data
    assert "estadisticas" in data


def test_endpoint_metrics():
    response = cliente.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metricas_test" in data
    assert "accuracy" in data["metricas_test"]
    assert "f1_weighted" in data["metricas_test"]
    assert "matriz_confusion" in data["metricas_test"]


def test_endpoint_predict_batch():
    payload = {
        "textos": [
            "Delicioso menú degustación.",
            "Horrible experiencia, nunca volveremos.",
        ]
    }
    response = cliente.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert "sentimiento_predicho" in data[0]
