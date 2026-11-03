# 🍽️ SaborIA — Sistema Inteligente de Análisis de Sentimientos de Reseñas de Restaurantes

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![Tests](https://img.shields.io/badge/Tests-34%20Passing-brightgreen)
![Status](https://img.shields.io/badge/Status-Entrega%20Final-success)

> **Materia:** Tópicos y Programación Avanzada  
> **Integrantes del Equipo:**  
> - **Nicolás** (`@Nicolas-zv`): Administrador del Repositorio / Datos, EDA, Naive Bayes y Backend FastAPI.  
> - **Roberto** (`@Roberto`): Pipeline de NLP, Regresión Logística con GridSearchCV y Frontend Web Interactivo.

---

## 📖 1. Descripción del Proyecto

**SaborIA** es una plataforma integral de Procesamiento de Lenguaje Natural (NLP) y Machine Learning supervisado diseñada específicamente para el dominio gastronómico en español. Su objetivo es clasificar opiniones y reseñas de restaurantes en tres polaridades: **Positivo**, **Neutral** o **Negativo**, proporcionando además el desglose paso a paso del texto procesado y la distribución de probabilidades de confianza.

### 🌟 Características Principales:
- **Preprocesamiento Lingüístico Riguroso:** Normalización de modismos culinarios, tratamiento semántico de emojis gastronómicos y conservación de partículas de negación (*no, ni, nunca, sin, tampoco*).
- **Modelos Entrenados y Comparados:**
  - **Naive Bayes Multinomial:** Modelo probabilístico de alta velocidad.
  - **Regresión Logística:** Optimizado con validación cruzada estratificada de 5 pliegues (`StratifiedKFold`) y `GridSearchCV`.
- **API REST Robusta con FastAPI:** Endpoints de inferencia individual, predicción por lotes (batch), inspección paso a paso del pipeline NLP y métricas.
- **Frontend Interactivo Glassmorphism:** Cuatro pestañas con analizador en tiempo real, visualizador dinámico de NLP, dashboard de métricas con matriz de confusión y análisis cualitativo de fallos.
- **Calidad de Código y Reproducibilidad:** Batería de **34 pruebas automatizadas** con `pytest` y serialización de modelos con `joblib`.

---

## 🏗️ 2. Arquitectura del Sistema (5 Capas)

```
┌─────────────────────────────────────────────────────────────┐
│                   CAPA 5: INTERFAZ WEB                      │
│   Frontend (HTML5 / CSS Glassmorphism / JS Vanilla)         │
│   • Analizador en vivo  • Pipeline NLP  • Dashboard Métricas │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON
┌──────────────────────────────▼──────────────────────────────┐
│                    CAPA 4: BACKEND REST                     │
│   FastAPI (backend/main.py) en puerto 8000                  │
│   • /predict  • /predict/batch  • /preprocess  • /metrics   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Serialización (.joblib)
┌──────────────────────────────▼──────────────────────────────┐
│                   CAPA 3: MACHINE LEARNING                  │
│   • Naive Bayes (MultinomialNB)                             │
│   • Regresión Logística con GridSearchCV (Scikit-Learn)     │
└──────────────────────────────┬──────────────────────────────┘
                               │ Vectores TF-IDF
┌──────────────────────────────▼──────────────────────────────┐
│                CAPA 2: PIPELINE NLP ESPECIALIZADO           │
│   • Normalización • Tokenización • Stopwords (negaciones)   │
│   • Stemming / Lematización (spaCy & NLTK)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Carga CSV
┌──────────────────────────────▼──────────────────────────────┐
│             CAPA 1: DATOS Y ANÁLISIS EXPLORATORIO           │
│   • data/resenas_restaurantes.csv (730 reseñas)             │
│   • notebooks/01_analisis_exploratorio.ipynb                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 3. Guía de Instalación Rápida

### Requisitos Previos:
- **Python 3.10** o superior instalado en el sistema.
- **Git** instalado.

### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/Nicolas-zv/SaborIA.git
cd SaborIA
```

### Paso 2: Crear y activar el entorno virtual

En **Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En **Linux / macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar dependencias exactas
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Opcional: Si spaCy requiere el modelo en español, se descarga automáticamente al primer uso o manualmente con: `python -m spacy download es_core_news_sm`)*

---

## 🧪 4. Ejecución de Pruebas Automatizadas

Para validar que todos los módulos (NLP, modelos y endpoints de la API) se encuentran operativos:

```bash
python -m pytest
```

> **Resultado:** 34 pruebas aprobadas exitosamente (`34 passed`).

---

## ⚙️ 5. Entrenamiento y Reproducibilidad de los Modelos

Ambos modelos pueden reentrenarse y serializarse con un solo comando:

### 1. Entrenar y Evaluar Naive Bayes:
```bash
python src/entrenar_naive_bayes.py
```
*Genera el artefacto en `models/naive_bayes.joblib`.*

### 2. Entrenar y Optimizar Regresión Logística (`GridSearchCV`):
```bash
python -m src.models.train
```
*Ejecuta la validación cruzada (5 folds), evalúa en el test set ciego (146 reseñas), compara contra el Dummy Baseline y genera `models/regresion_logistica.joblib` y `models/metricas_regresion_logistica.json`.*

---

## 🌐 6. Cómo Levantar y Utilizar el Sistema

### Paso 1: Iniciar el Backend API REST (FastAPI)
Con el entorno virtual activado, ejecuta desde la raíz del proyecto:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

- La API quedará disponible en: `http://127.0.0.1:8000`
- **Documentación Swagger interactiva:** `http://127.0.0.1:8000/docs`
- **Documentación ReDoc:** `http://127.0.0.1:8000/redoc`

#### Endpoints Principales:
| Método | Endpoint | Descripción |
| :---: | :--- | :--- |
| `GET` | `/health` | Estado del servicio y modelos cargados |
| `POST` | `/predict` | Inferencia individual con distribución de probabilidades |
| `POST` | `/predict/batch` | Inferencia por lotes (múltiples reseñas) |
| `POST` | `/preprocess` | Desglose paso a paso de las 5 fases del pipeline NLP |
| `GET` | `/metrics` | Métricas de rendimiento y matriz de confusión en JSON |
| `GET` | `/errors` | Casos de estudio y análisis cualitativo de fallos |

---

### Paso 2: Abrir la Interfaz Web (Frontend)

Para abrir la interfaz web interactiva:
1. Abre directamente el archivo `frontend/index.html` en tu navegador web preferido (doble clic o arrastrándolo a Chrome/Edge/Firefox).
2. O alternativamente, si utilizas la extensión *Live Server* de VS Code, ábrelo en `http://127.0.0.1:5500`.

La aplicación detectará automáticamente si el backend está en línea (mostrando el indicador en verde) y conectará todas las inferencias en tiempo real. Si el backend está apagado, la aplicación cuenta con un modo de respaldo local para demostración.

---

## 📊 7. Comparativa de Modelos y Métricas Obtenidas

Evaluación realizada sobre el conjunto de prueba aislado (146 muestras reales):

| Métrica | Baseline Dummy (Most Frequent) | Naive Bayes | Regresión Logística (SaborIA) | Mejora sobre Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Exactitud (Accuracy)** | 53.42% | 67.12% | **69.86%** | **+16.44%** |
| **F1-Score Ponderado** | 37.21% | 66.85% | **69.80%** | **+32.59%** |
| **F1-Score Macro** | 23.21% | 61.40% | **63.88%** | **+40.67%** |
| **Tiempo de Inferencia** | <1 ms | <2 ms | <2 ms | Óptimo para producción |

*Documentación extendida de comparativa en [docs/COMPARACION_MODELOS.md](docs/COMPARACION_MODELOS.md).*

---

## 📁 8. Estructura del Repositorio

```text
SaborIA/
├── backend/
│   └── main.py                     # API REST con FastAPI y endpoints
├── data/
│   ├── resenas_restaurantes.csv    # Dataset consolidado (730 reseñas)
│   └── errores_naive_bayes.csv     # Registro de errores de Naive Bayes
├── docs/
│   ├── ANALISIS_ERRORES.md         # Análisis cualitativo de errores
│   ├── COMPARACION_MODELOS.md      # Comparativa técnica entre modelos
│   ├── COMPARACION_NLP.md          # Comparativa Stemming vs Lematización
│   ├── DATASET.md                  # Documentación del dataset gastronómico
│   ├── guion_defensa_roberto.md    # Guion de defensa y diapositivas
│   └── enunciado_proyecto.md       # Rúbrica oficial de evaluación
├── frontend/
│   ├── index.html                  # Interfaz web principal
│   ├── styles.css                  # Estilos modernos y glassmorphism
│   └── app.js                      # Lógica interactiva y conexión API
├── models/
│   ├── naive_bayes.joblib          # Modelo Naive Bayes serializado
│   ├── regresion_logistica.joblib  # Modelo Regresión Logística serializado
│   ├── metricas_regresion_logistica.json
│   └── analisis_errores.json
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb # EDA documentado
├── src/
│   ├── models/
│   │   ├── logistic_regression.py  # Clasificador y optimización GridSearchCV
│   │   └── train.py                # Script reproducible de entrenamiento
│   ├── nlp/
│   │   ├── normalizer.py           # Normalización de textos y emojis
│   │   ├── tokenizer.py            # Tokenización culinaria
│   │   ├── vectorizer.py           # Vectorización TF-IDF
│   │   ├── ngram_comparator.py     # Comparativa unigramas vs bigramas
│   │   └── pipeline.py             # Pipeline unificado paso a paso
│   ├── stopwords_es.py             # Stopwords con negaciones protegidas
│   └── entrenar_naive_bayes.py     # Script de entrenamiento Naive Bayes
├── tests/
│   ├── test_api.py                 # Pruebas de integración de la API
│   ├── test_normalizer.py          # Pruebas de normalización
│   ├── test_tokenizer.py           # Pruebas de tokenización
│   ├── test_vectorizer.py          # Pruebas de TF-IDF
│   ├── test_pipeline_nlp.py        # Pruebas del pipeline integral
│   └── test_logistic_regression.py # Pruebas del modelo de Regresión Logística
├── CONTRIBUCIONES.md               # Registro detallado de aportes por integrante
├── README.md                       # Guía completa de uso e instalación
└── requirements.txt                # Dependencias exactas del proyecto
```

---

## 👥 9. Equipo de Desarrollo y Rúbrica Individual

Para consultar el registro de commits, responsabilidades por etapa y trazabilidad de Git según la rúbrica de evaluación individual (60 pts):
- Consulta **[CONTRIBUCIONES.md](CONTRIBUCIONES.md)**.
- Consulta el material de preparación para la defensa individual en **[docs/guion_defensa_roberto.md](docs/guion_defensa_roberto.md)**.