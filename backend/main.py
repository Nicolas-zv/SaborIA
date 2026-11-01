"""
Backend API REST (FastAPI) — SaborIA.

Expone:
- GET /health: Estado del servicio y modelo serializado en memoria.
- POST /predict: Predicción individual y por lotes con distribución de probabilidades por clase.
- POST /preprocess: Visualizador paso a paso del pipeline de NLP (Capa 2 / Frontend).
- GET /metrics: Métricas completas, matriz de confusión y comparativa contra Baseline.
- GET /errors: Análisis cualitativo de errores detectados en la evaluación.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models.logistic_regression import ClasificadorRegresionLogistica
from src.nlp.pipeline import pipeline_nlp_default

app = FastAPI(
    title="SaborIA – API de Análisis de Sentimientos",
    description="API REST de clasificación de reseñas gastronómicas y desglose de NLP.",
    version="1.0.0",
)

# Configurar CORS para permitir comunicación con el Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de artefactos
RUTA_RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = RUTA_RAIZ / "models" / "regresion_logistica.joblib"
RUTA_METRICAS = RUTA_RAIZ / "models" / "metricas_regresion_logistica.json"
RUTA_ERRORES = RUTA_RAIZ / "models" / "analisis_errores.json"
RUTA_FRONTEND = RUTA_RAIZ / "frontend"

# Cargar modelo en memoria al iniciar (evitando reentrenar o reimplementar preprocesamiento)
clasificador_activo: Optional[ClasificadorRegresionLogistica] = None
if RUTA_MODELO.exists():
    try:
        clasificador_activo = ClasificadorRegresionLogistica.cargar(str(RUTA_MODELO))
    except Exception as e:
        print(f"Advertencia al cargar modelo: {e}")


# Modelos Pydantic para validación estricta de entrada/salida
class ResenaIndividualInput(BaseModel):
    texto: str = Field(..., min_length=2, max_length=5000, description="Texto de la reseña a analizar")


class ResenasLoteInput(BaseModel):
    textos: List[str] = Field(..., min_length=1, max_length=200, description="Lista de reseñas para inferencia por lotes")


class PrediccionOutput(BaseModel):
    texto: str
    sentimiento_predicho: str
    confianza: float
    probabilidades: Dict[str, float]


@app.get("/health", tags=["Salud"])
def health_check():
    """Endpoint de comprobación de salud del servicio y estado del modelo."""
    return {
        "estado": "operativo",
        "servicio": "SaborIA REST API",
        "modelo_cargado": clasificador_activo is not None,
        "clases": clasificador_activo.clases_ if clasificador_activo else [],
    }


@app.post("/predict", response_model=PrediccionOutput, tags=["Inferencia"])
def predecir_sentimiento(payload: ResenaIndividualInput):
    """
    Predice el sentimiento (positivo, negativo o neutral) de una reseña individual
    y retorna la clase predicha con su vector de probabilidades.
    """
    texto = payload.texto.strip()
    if not texto:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La reseña no puede estar vacía o compuesta solo de espacios.",
        )

    if clasificador_activo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de Machine Learning no está cargado. Ejecuta 'python -m src.models.train'.",
        )

    pred = clasificador_activo.predecir([texto])[0]
    probs = clasificador_activo.predecir_probabilidades([texto])[0]
    confianza = probs.get(pred, 0.0)

    return {
        "texto": texto,
        "sentimiento_predicho": pred,
        "confianza": confianza,
        "probabilidades": probs,
    }


@app.post("/predict/batch", response_model=List[PrediccionOutput], tags=["Inferencia"])
def predecir_lote(payload: ResenasLoteInput):
    """Predice el sentimiento de un conjunto de reseñas (análisis masivo)."""
    if clasificador_activo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo de Machine Learning no está cargado.",
        )

    textos_validos = [t.strip() for t in payload.textos if t.strip()]
    if not textos_validos:
        raise HTTPException(status_code=400, detail="No se enviaron textos válidos en el lote.")

    preds = clasificador_activo.predecir(textos_validos)
    probs_lista = clasificador_activo.predecir_probabilidades(textos_validos)

    resultados = []
    for txt, pred, probs in zip(textos_validos, preds, probs_lista):
        resultados.append({
            "texto": txt,
            "sentimiento_predicho": pred,
            "confianza": probs.get(pred, 0.0),
            "probabilidades": probs,
        })
    return resultados


@app.post("/preprocess", tags=["NLP"])
def preprocesar_paso_a_paso(payload: ResenaIndividualInput):
    """
    Expone la transformación del texto a través de cada fase del pipeline de NLP:
    Original ➔ Normalización ➔ Tokenización ➔ Filtrado Stop Words ➔ Lematización.
    """
    texto = payload.texto.strip()
    if not texto:
        raise HTTPException(status_code=400, detail="El texto para preprocesar está vacío.")

    return pipeline_nlp_default.procesar_paso_a_paso(texto)


@app.get("/metrics", tags=["Métricas"])
def obtener_metricas():
    """Retorna las métricas del modelo evaluado, matriz de confusión y comparativa baseline."""
    if not RUTA_METRICAS.exists():
        raise HTTPException(
            status_code=404,
            detail="Archivo de métricas no encontrado. Ejecuta 'python -m src.models.train'.",
        )
    with open(RUTA_METRICAS, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/errors", tags=["Métricas"])
def obtener_analisis_errores():
    """Retorna ejemplos concretos de falsos positivos y negativos para el análisis cualitativo."""
    if not RUTA_ERRORES.exists():
        return []
    with open(RUTA_ERRORES, "r", encoding="utf-8") as f:
        return json.load(f)


# Montar Frontend estático para servir la interfaz web directamente
if RUTA_FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(RUTA_FRONTEND), html=True), name="frontend")

