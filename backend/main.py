
from pathlib import Path

import joblib
import json
from src.models.logistic_regression import ClasificadorRegresionLogistica
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.nlp.pipeline import PipelineNLP
# Crear aplicación FastAPI
app = FastAPI(
    title="SaborIA API",
    description="API para análisis de sentimientos de reseñas de restaurantes",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Ruta del modelo entrenado
RUTA_MODELO = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "naive_bayes.joblib"
)


RUTA_MODELO_LOGISTICO = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "regresion_logistica.joblib"
)

modelo_logistico = None

if RUTA_MODELO_LOGISTICO.exists():
    modelo_logistico = ClasificadorRegresionLogistica.cargar(
        str(RUTA_MODELO_LOGISTICO)
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


@app.get("/health")
def comprobar_estado():
    return {
        "estado": "operativo",
        "servicio": "SaborIA REST API",
        "modelo_cargado": modelo is not None,
        "clases": (
            [str(clase) for clase in modelo.classes_]
            if modelo is not None
            else []
        ),
    }


@app.post("/predict")
def predecir_para_frontend(resena: ResenaEntrada):
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo Naive Bayes no está disponible.",
        )

    texto = resena.texto.strip()

    if not texto:
        raise HTTPException(
            status_code=400,
            detail="La reseña no puede estar vacía.",
        )

    prediccion = str(modelo.predict([texto])[0])

    probabilidades_modelo = modelo.predict_proba([texto])[0]

    probabilidades = {
        str(clase): float(probabilidad)
        for clase, probabilidad in zip(
            modelo.classes_,
            probabilidades_modelo
        )
    }

    return {
        "texto": texto,
        "sentimiento_predicho": prediccion,
        "confianza": probabilidades[prediccion],
        "probabilidades": probabilidades,
    }

class LoteResenas(BaseModel):
    textos: list[str]


@app.post("/predict/batch")
def predecir_lote(lote: LoteResenas):
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo Naive Bayes no está disponible."
        )

    textos = [texto.strip() for texto in lote.textos if texto.strip()]

    if not textos:
        raise HTTPException(
            status_code=400,
            detail="Debes ingresar al menos una reseña válida."
        )

    predicciones = modelo.predict(textos)
    probabilidades_modelo = modelo.predict_proba(textos)

    resultados = []

    for texto, prediccion, probabilidades_fila in zip(
        textos, predicciones, probabilidades_modelo
    ):
        sentimiento = str(prediccion)

        probabilidades = {
            str(clase): float(probabilidad)
            for clase, probabilidad in zip(
                modelo.classes_, probabilidades_fila
            )
        }

        resultados.append({
            "texto": texto,
            "sentimiento_predicho": sentimiento,
            "confianza": probabilidades[sentimiento],
            "probabilidades": probabilidades
        })

    return resultados


@app.post("/preprocess")
def preprocesar_texto(resena: ResenaEntrada):
    texto = resena.texto.strip()

    if not texto:
        raise HTTPException(
            status_code=400,
            detail="La reseña no puede estar vacía."
        )

    pipeline = PipelineNLP(
        metodo_reduccion="lematizacion",
        conservar_negaciones=True
    )

    resultado = pipeline.procesar_paso_a_paso(texto)

    return resultado


@app.get("/metrics")
def obtener_metricas():
    ruta_metricas = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "metricas_regresion_logistica.json"
    )

    if not ruta_metricas.exists():
        raise HTTPException(
            status_code=404,
            detail="No se encontró el archivo de métricas."
        )

    try:
        with open(ruta_metricas, "r", encoding="utf-8") as archivo:
            metricas = json.load(archivo)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail="No se pudieron leer las métricas."
        )

    return metricas



@app.get("/errors")
def obtener_errores():
    ruta_errores = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "analisis_errores.json"
    )

    if not ruta_errores.exists():
        raise HTTPException(
            status_code=404,
            detail="No se encontró el archivo de análisis de errores."
        )

    try:
        with open(ruta_errores, "r", encoding="utf-8") as archivo:
            errores = json.load(archivo)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail="No se pudo leer el análisis de errores."
        )

    if not isinstance(errores, list):
        raise HTTPException(
            status_code=500,
            detail="El archivo de errores tiene un formato incorrecto."
        )

    return errores


@app.post("/predict/logistic")
def predecir_regresion_logistica(resena: ResenaEntrada):
    if modelo_logistico is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo de regresión logística no está disponible."
        )

    texto = resena.texto.strip()

    if not texto:
        raise HTTPException(
            status_code=400,
            detail="La reseña no puede estar vacía."
        )

    sentimiento = str(modelo_logistico.predecir([texto])[0])

    probabilidades_originales = (
        modelo_logistico.predecir_probabilidades([texto])[0]
    )

    probabilidades = {
        str(clase): float(valor)
        for clase, valor in probabilidades_originales.items()
    }

    return {
        "texto": texto,
        "modelo": "regresion_logistica",
        "sentimiento_predicho": sentimiento,
        "confianza": probabilidades[sentimiento],
        "probabilidades": probabilidades
    }


@app.post("/predict/logistic/batch")
def predecir_lote_logistico(lote: LoteResenas):
    if modelo_logistico is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo de regresión logística no está disponible."
        )

    textos = [
        texto.strip()
        for texto in lote.textos
        if isinstance(texto, str) and texto.strip()
    ]

    if not textos:
        raise HTTPException(
            status_code=400,
            detail="Debes ingresar al menos una reseña válida."
        )

    predicciones = modelo_logistico.predecir(textos)
    probabilidades_lista = modelo_logistico.predecir_probabilidades(textos)

    resultados = []

    for texto, prediccion, probabilidades_originales in zip(
        textos, predicciones, probabilidades_lista
    ):
        sentimiento = str(prediccion)

        probabilidades = {
            str(clase): float(valor)
            for clase, valor in probabilidades_originales.items()
        }

        resultados.append({
            "texto": texto,
            "modelo": "regresion_logistica",
            "sentimiento_predicho": sentimiento,
            "confianza": probabilidades[sentimiento],
            "probabilidades": probabilidades
        })

    return resultados

