# Informe Técnico de Ingeniería — Proyecto SaborIA
**Sistema Inteligente de Análisis de Sentimientos de Reseñas de Restaurantes**  
**Materia:** Tópicos y Programación Avanzada  
**Autores:** Roberto (@Roberto) & Nicolás (@Nicolas-zv)  
**Fecha:** Noviembre 2026  
**Versión:** 1.0.0 (Entrega Parcial / Final)

---

## 1. Introducción y Propósito del Sistema

El análisis automático de opiniones en plataformas gastronómicas representa un desafío clave en el Procesamiento de Lenguaje Natural (NLP). Las reseñas de restaurantes en español combinan modismos coloquiales, lenguaje figurado, contracciones informales, emojis y construcciones sintácticas complejas con dobles negaciones.

**SaborIA** es una solución integral y reproducible que clasifica reseñas gastronómicas en tres categorías de polaridad: **Positivo**, **Neutral** y **Negativo**. El sistema abarca el ciclo de vida completo de un producto de datos: desde la ingesta del dato crudo hasta una interfaz web interactiva, respaldada por un pipeline de NLP estricto, modelos de Machine Learning optimizados con validación cruzada y una API REST desacoplada.

```text
[Dataset Crudo]
      │
      ▼
[Pipeline de NLP] (Normalización ➔ Tokenización ➔ Stop Words con Negaciones ➔ Lematización ➔ TF-IDF)
      │
      ▼
[Machine Learning] (Regresión Logística + GridSearchCV 5 Folds + Class Weight Balanced)
      │
      ▼ (Serialización .joblib)
[Artefacto de Inferencia]
      │
      ▼ (Carga en memoria sin duplicar lógica)
[Backend API REST] (FastAPI: /predict, /predict/batch, /preprocess, /metrics, /health)
      ▲
      │ (JSON REST)
      ▼
[Frontend Web] (Inferencia en vivo/lotes, Visualizador paso a paso NLP, Dashboard de Métricas)
```

---

## 2. Capa 1: Datos y Análisis Exploratorio (EDA)

### 2.1. Dataset
- **Fuente:** Dataset `HorecaReviews-ES` adaptado para el dominio gastronómico de restaurantes.
- **Total de registros limpios:** 730 reseñas etiquetadas.
- **Distribución de clases (Desbalance detectado):**
  - **Positivo:** 391 muestras (53.6%)
  - **Negativo:** 209 muestras (28.6%)
  - **Neutral:** 130 muestras (17.8%)

### 2.2. Partición Estratificada Aislada
Para garantizar la validez científica y evitar el filtrado de información (*data leakage*):
- Se aplicó una división 80% entrenamiento (584 muestras) y 20% prueba (146 muestras).
- Se utilizó división estratificada fija (`train_test_split(stratify=y, test_size=0.2, random_state=42)`).
- El conjunto de prueba se mantuvo completamente aislado y solo se evaluó al final de la optimización.

### 2.3. Tratamiento del Desbalance
Dado que la clase positiva supera ampliamente a la neutral, se implementó el parámetro `class_weight='balanced'` en el clasificador, el cual pondera inversamente las frecuencias de clase:

$$w_j = \frac{N}{K \cdot n_j}$$

Donde $N$ es el total de muestras, $K$ el número de clases y $n_j$ las muestras de la clase $j$.

---

## 3. Capa 2: Pipeline de NLP (Justificación Técnica de Decisiones)

Cada fase fue diseñada para preservar la semántica del sentimiento y documentar el efecto **antes vs. después**:

| Fase | Técnica Aplicada | Justificación Técnica | Ejemplo Antes ➔ Después |
| :--- | :--- | :--- | :--- |
| **1. Normalización** | Minúsculas, mapeo léxico de emojis gastronómicos y normalización de exclamaciones. | Los emojis (`😋` ➔ `delicioso`, `😡` ➔ `pésimo`) y signos repetidos (`!!!` ➔ `[EXCLAMACION]`) concentran alta carga emocional y no deben ser eliminados como ruido. | `"¡¡Excelente cena!! 😋"` ➔ `"excelente cena [EXCLAMACION] delicioso"` |
| **2. Tokenización** | Segmentación lingüística de modismos compuestos y contracciones. | Protege expresiones gastronómicas compuestas (`"menú degustación"`, `"relación calidad-precio"`) para que no se separen en términos aislados. | `"probamos el menú degustación"` ➔ `['probamos', 'menu_degustacion']` |
| **3. Stop Words** | Filtrado con lista en español **conservando negaciones críticas**. | Palabras de negación (`"no"`, `"ni"`, `"nunca"`, `"sin"`, `"tampoco"`) cambian radicalmente la polaridad. Eliminarlas convertiría `"no es bueno"` en `"bueno"` (falso positivo). | `"la atención no fue buena"` ➔ `['atencion', 'no', 'buena']` |
| **4. Reducción Morfológica** | **Lematización (spaCy)** con opción configurable de **Stemming (Snowball)**. | La lematización produce formas canónicas válidas del idioma (`"comiendo"` ➔ `"comer"`), mientras que el stemming trunca raíces (`"com"`). Se comprobó mayor interpretabilidad con lematización. | `"los camareros atendieron"` ➔ `['camarero', 'atender']` |
| **5. N-gramas y TF-IDF** | Vectorización TF-IDF con `sublinear_tf=True`, `min_df=2`, `max_df=0.95`. | La escala logarítmica de término $1 + \log(\text{tf})$ amortigua repeticiones enfáticas. `min_df=2` filtra erratas tipográficas y `max_df=0.95` remueve términos omnipresentes. | Matriz dispersa de 1.500 características de máxima relevancia. |

---

## 4. Capa 3: Machine Learning, Optimización y Métricas

### 4.1. Algoritmo y Optimización
Se seleccionó la **Regresión Logística Multinomial** combinada con TF-IDF por su alta interpretabilidad, rapidez de inferencia y calibración probabilística.

Se utilizó `GridSearchCV` con 5 Folds estratificados (`StratifiedKFold`):
- **Espacio de búsqueda:**
  - `C`: $[0.1, 1.0, 5.0, 10.0]$
  - `class_weight`: `['balanced', None]`
  - `vectorizador__ngram_range`: $[(1, 1), (1, 2)]$
  - `vectorizador__max_features`: $[1500, 2500, \text{None}]$
- **Mejores Hiperparámetros Encontrados:**
  - `C = 1.0` (balance óptimo de regularización $L_2$).
  - `class_weight = 'balanced'` (compensación efectiva del desbalance).
  - `ngram_range = (1, 1)` y `max_features = 1500`.

### 4.2. Resultados en el Conjunto de Prueba (Test Set Aislado — 146 Reseñas)

| Métrica | Línea Base (Dummy Most Frequent) | Regresión Logística (SaborIA) | Ganancia Neta Aportada |
| :--- | :---: | :---: | :---: |
| **Exactitud (Accuracy)** | 53.42% | **69.86%** | **+16.44%** |
| **F1-Score Ponderado** | 37.21% | **69.80%** | **+32.59%** |
| **Precisión Ponderada** | 28.54% | **69.89%** | **+41.35%** |
| **Recall Ponderado** | 53.42% | **69.86%** | **+16.44%** |
| **F1-Score Macro** | 23.21% | **63.88%** | **+40.67%** |

### 4.3. Matriz de Confusión
```text
                  Pred. Negativo    Pred. Neutral    Pred. Positivo
Real Negativo           27                6                9
Real Neutral             9               12                5
Real Positivo           10                5               63
```

### 4.4. Análisis Cualitativo de Errores (Falsos Positivos y Negativos)
Para la defensa técnica se identificaron tres fuentes principales de error:
1. **Sarcasmo e ironía:** Reseñas que utilizan vocabulario positivo para expresar frustración (ej. *"Maravilloso esperar hora y media por un café"*).
2. **Ambigüedad en neutrales:** Reseñas con aspectos mixtos (ej. *"Comida excelente pero lugar incómodo"*), donde el modelo tiende a inclinarse hacia el adjetivo dominante.
3. **Hipérboles metafóricas:** Reseñas poéticas o metafóricas que carecen de adjetivos tradicionales de valoración.

---

## 5. Capa 4: Backend API REST (FastAPI)

El backend sigue el principio de **cero reimplementación de preprocesamiento**: el archivo serializado `models/regresion_logistica.joblib` empaqueta tanto el vectorizador entrenado como el clasificador. Al invocar `.predict()`, la entrada cruda se transforma idénticamente a como se hizo en entrenamiento.

### Endpoints Principales:
- `GET /health`: Comprobación de salud y modelo activo en memoria.
- `POST /predict`: Inferencia individual con validación de esquema Pydantic (devuelve clase predicha, confianza y vector de probabilidades).
- `POST /predict/batch`: Inferencia masiva por lotes.
- `POST /preprocess`: Desglose paso a paso del pipeline de NLP para fines pedagógicos e interactivos.
- `GET /metrics`: Consulta de las métricas de rendimiento, matriz de confusión y parámetros de `GridSearchCV`.
- `GET /errors`: Casos concretos de fallos evaluados para revisión técnica.
- `GET /`: Servidor web integrado que monta y entrega directamente la interfaz frontend.

---

## 6. Capa 5: Frontend Web Interactivo

Construido bajo estándares modernos con HTML5 semántico, Vanilla CSS (Dark Mode con Glassmorphism) y JavaScript asíncrono.
- **Pestaña 1 (Analizador en Vivo):** Formulario con botones de presets rápidos, barra de confianza y gráfico de barras de probabilidad. Incluye sección colapsable para análisis masivo por archivos `.csv` / `.txt`.
- **Pestaña 2 (Visualizador NLP Paso a Paso):** Vista interactiva en cascada que muestra cómo el texto original viaja por Normalización, Tokenización, Filtrado (resaltando negaciones en rojo), Lematización y TF-IDF.
- **Pestaña 3 (Dashboard de Métricas):** Visualización de KPIs, Matriz de Confusión interactiva con calor térmico y gráfica de comparación contra el Dummy.
- **Pestaña 4 (Casos de Estudio & Errores):** Catálogo de ejemplos reales para la defensa individual.

---

## 7. Catálogo de Modificaciones en Vivo Preparadas (Defensa Individual)

El código se estructuró modularmente para ejecutar modificaciones en directo durante la evaluación docente:
1. **Modificación de Stopwords:** Agregar o retirar términos de negación en `src/stopwords_es.py` o modificar listas en `src/nlp/pipeline.py`.
2. **Modificación de N-gramas:** Alternar `ngram_range=(1, 2)` a `(1, 3)` directamente en el constructor de `VectorizadorResenas` o en `ClasificadorRegresionLogistica`.
3. **Alternancia de Stemming / Lematización:** Cambiar el flag `metodo_reduccion="stemming"` en `PipelineNLP` de forma inmediata.
4. **Validación de Backend:** Ajustar la restricción de longitud mínima de palabras en el endpoint `/predict` de `backend/main.py`.
5. **Ajuste de Frontend:** Modificar el umbral de confianza visual en `frontend/app.js` para alertas dinámicas.

---

## 8. Conclusiones

El proyecto **SaborIA** demuestra cómo la articulación rigurosa de las técnicas de NLP en español combinadas con modelos probabilísticos calibrados logra superar ampliamente una línea base trivial (+32.59% en F1 ponderado). La separación limpia de responsabilidades en 5 capas, respaldada por **35 pruebas unitarias automatizadas**, garantiza un sistema robusto, explicable y listo para producción.
