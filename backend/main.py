
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Crear aplicación FastAPI
app = FastAPI(
    title="SaborIA API",
    description="API para análisis de sentimientos de reseñas de restaurantes",
    version="1.0.0",
)

# Ruta del modelo entrenado
RUTA_MODELO = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "naive_bayes.joblib"
)

# Cargar modelo al iniciar la aplicación
modelo = None

if RUTA_MODELO.exists():
    modelo = joblib.load(RUTA_MODELO)


# Datos que recibirá la API
class ResenaEntrada(BaseModel):
    texto: str = Field(min_length=1)


# Endpoint de bienvenida
@app.get("/")
def inicio():
    return {
        "mensaje": "Bienvenido a SaborIA",
        "estado": "activo",
    }


# Endpoint para predecir sentimientos
@app.post("/predecir")
def predecir_sentimiento(resena: ResenaEntrada):

    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo de Machine Learning no está disponible.",
        )

    texto = resena.texto.strip()

    if not texto:
        raise HTTPException(
            status_code=400,
            detail="La reseña no puede estar vacía.",
        )

    prediccion = modelo.predict([texto])[0]

    return {
        "texto": texto,
        "sentimiento": str(prediccion),
    }
