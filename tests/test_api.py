
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_inicio():
    respuesta = client.get("/")

    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "activo"


def test_prediccion_positiva():
    respuesta = client.post(
        "/predecir",
        json={
            "texto": "La comida estaba deliciosa y la atención fue excelente."
        },
    )

    assert respuesta.status_code == 200
    assert respuesta.json()["sentimiento"] == "positivo"


def test_prediccion_negativa():
    respuesta = client.post(
        "/predecir",
        json={
            "texto": "La comida estaba fría, la atención fue pésima y esperamos demasiado tiempo."
        },
    )

    assert respuesta.status_code == 200
    assert respuesta.json()["sentimiento"] == "negativo"


def test_resena_vacia():
    respuesta = client.post(
        "/predecir",
        json={"texto": ""},
    )

    assert respuesta.status_code == 422


def test_resena_solo_espacios():
    respuesta = client.post(
        "/predecir",
        json={"texto": "   "},
    )

    assert respuesta.status_code == 400


def test_sin_campo_texto():
    respuesta = client.post(
        "/predecir",
        json={},
    )

    assert respuesta.status_code == 422
