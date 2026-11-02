# SaborIA 🍽️🤖

**Sistema Inteligente de Análisis de Sentimientos de Reseñas de Restaurantes**  
*Materia: Tópicos y Programación Avanzada*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/pytest-35%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Status](https://img.shields.io/badge/Estado-100%25%20Operativo-success.svg)](#)

---

## 📌 Descripción

**SaborIA** es una aplicación web integral que clasifica reseñas gastronómicas en español en tres polaridades: **Positivo**, **Neutral** y **Negativo**.

El sistema implementa de extremo a extremo las **5 capas obligatorias**: desde la ingesta del dato crudo hasta una interfaz web moderna en modo oscuro, pasando por un pipeline de NLP riguroso (normalización, modismos gastronómicos, stopwords protegiendo negaciones, lematización con spaCy y TF-IDF), optimización de hiperparámetros con `GridSearchCV`, serialización con `joblib`, y una API REST desacoplada construida con FastAPI.

---

## 🏗️ Arquitectura del Sistema (Las 5 Capas)

```text
[1. Dataset Crudo] ──────────► 730 reseñas en data/resenas_restaurantes.csv
         │
         ▼
[2. Pipeline NLP] ───────────► Normalización ➔ Tokenización ➔ Stop Words (con negaciones) ➔ Lematización ➔ TF-IDF
         │
         ▼
[3. Machine Learning] ───────► Regresión Logística + GridSearchCV (5-folds, class_weight='balanced')
         │
         ├───► models/regresion_logistica.joblib (Artefacto serializado)
         ├───► models/metricas_regresion_logistica.json (Métricas completas)
         └───► models/analisis_errores.json (Falsos positivos/negativos explicados)
         │
         ▼
[4. Backend API REST] ───────► FastAPI en backend/main.py (/predict, /preprocess, /metrics, /health)
         ▲
         │ (Peticiones HTTP JSON)
         ▼
[5. Frontend Web] ───────────► Interfaz interactiva en frontend/ (Analizador en vivo, visualizador NLP y dashboard)
```

---

## 👥 Equipo de Trabajo y Contribuciones

| Integrante | Rol Principal | Aportes Clave en el Repositorio |
| :--- | :--- | :--- |
| **Nicolás** (`@Nicolas-zv`) | Datos, NLP Base & Backend | Ingesta de dataset, EDA en Jupyter, stopwords base y comparativa Stemming vs. Lematización. |
| **Roberto** (`@Roberto`) | NLP Avanzado, ML & Frontend | Pipeline NLP completo, Regresión Logística con `GridSearchCV`, serialización `joblib`, API FastAPI y Frontend interactivo. |

*El desglose detallado de commits y justificaciones se encuentra en [CONTRIBUCIONES.md](CONTRIBUCIONES.md).*

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el Repositorio
```powershell
git clone https://github.com/Nicolas-zv/SaborIA.git
cd SaborIA
```

### 2. Configurar el Entorno Virtual
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Descargar Recursos Lingüísticos (NLTK y spaCy)
```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
python -m spacy download es_core_news_sm
```

---

## 🧪 Ejecución y Reproducibilidad

### 1. Reentrenar el Modelo (Pipeline Reproducible de 1 solo comando)
Toma los datos crudos, ejecuta el preprocesamiento, optimiza con `GridSearchCV` en 5 folds estratificados, evalúa y serializa los artefactos en `models/`:
```powershell
python -m src.models.train
```

### 2. Ejecutar la Batería de Pruebas Automatizadas
Ejecuta las **35 pruebas unitarias** que validan normalización, tokenización, vectorización TF-IDF, comparador de n-gramas, pipeline NLP, regresión logística y todos los endpoints de la API:
```powershell
pytest
```

### 3. Iniciar el Servidor Web y la API REST
Levanta el backend FastAPI (que automáticamente sirve el Frontend en la raíz):
```powershell
uvicorn backend.main:app --reload --port 8001
```

- **Aplicación Web Frontend:** 👉 **[http://localhost:8001/](http://localhost:8001/)**
- **Documentación Interactiva Swagger:** 👉 **[http://localhost:8001/docs](http://localhost:8001/docs)**
- **Endpoint de Salud (Health Check):** 👉 **[http://localhost:8001/health](http://localhost:8001/health)**

---

## 📊 Métricas de Rendimiento del Modelo Activo

Evaluado sobre el conjunto de prueba independiente (**146 reseñas no vistas**):

| Métrica | Línea Base (Dummy Most Frequent) | Regresión Logística (SaborIA) | Ganancia Neta |
| :--- | :---: | :---: | :---: |
| **Exactitud (Accuracy)** | 53.42% | **69.86%** | **+16.44%** |
| **F1-Score Ponderado** | 37.21% | **69.80%** | **+32.59%** |
| **Precisión Ponderada** | 28.54% | **69.89%** | **+41.35%** |
| **Recall Ponderado** | 53.42% | **69.86%** | **+16.44%** |
| **F1-Score Macro** | 23.21% | **63.88%** | **+40.67%** |

---

## 📂 Estructura del Repositorio

```text
SaborIA/
├── data/
│   └── resenas_restaurantes.csv        # Dataset de 730 reseñas etiquetadas
├── docs/
│   ├── enunciado_proyecto.md           # Requisitos y rúbrica oficial (100 pts)
│   ├── informe_tecnico.md              # Informe técnico formal del proyecto
│   ├── DATASET.md                      # Documentación del dataset gastronómico
│   └── COMPARACION_NLP.md              # Comparativa de stemming vs lematización
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb  # Análisis exploratorio (EDA) de clases y longitud
├── src/
│   ├── nlp/                            # Módulos del pipeline de procesamiento
│   │   ├── normalizer.py               # Normalización con emojis y exclamaciones
│   │   ├── tokenizer.py                # Tokenización con compuestos gastronómicos
│   │   ├── vectorizer.py               # Vectorización TF-IDF parametrizada
│   │   ├── ngram_comparator.py         # Comparador de unigramas vs bigramas
│   │   └── pipeline.py                 # Pipeline NLP unificado paso a paso
│   ├── models/                         # Modelado de Machine Learning
│   │   ├── logistic_regression.py      # Clasificador y optimizador GridSearchCV
│   │   └── train.py                    # Script reproducible de entrenamiento
│   ├── stopwords_es.py                 # Stop words con conservación de negaciones
│   └── comparar_stemming_lematizacion.py
├── backend/
│   └── main.py                         # API REST (FastAPI) y servidor estático
├── frontend/
│   ├── index.html                      # Interfaz web con las 4 pestañas
│   ├── styles.css                      # Sistema de diseño dark mode y glassmorphism
│   └── app.js                          # Lógica interactiva y conexión con la API
├── models/
│   ├── regresion_logistica.joblib      # Modelo y pipeline serializados
│   ├── metricas_regresion_logistica.json
│   └── analisis_errores.json
├── tests/                              # 35 pruebas automatizadas con pytest
├── CONTRIBUCIONES.md                   # Registro formal de aportes por integrante
└── README.md                           # Guía principal de uso e instalación
```