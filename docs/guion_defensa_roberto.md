# 🎓 Guion de Defensa y Preparación Individual — Roberto (`@Roberto`)

**Proyecto:** SaborIA – Sistema Inteligente de Análisis de Sentimientos de Reseñas de Restaurantes  
**Materia:** Tópicos y Programación Avanzada  
**Rol de Roberto:** Pipeline de NLP, Modelado ML (Regresión Logística con GridSearchCV) y Frontend Interactivo.  
**Puntos en juego en la defensa individual:** **60 puntos** (20 Git + 15 Explicación + 10 Preguntas + 15 Modificación en Vivo).

---

## 📑 1. Estructura de Diapositivas Recomendada (Aporte de Roberto)

Si preparan diapositivas en Canva / PowerPoint / Google Slides, Roberto debe defender estas láminas:

* **Diapositiva A: Capa 2 — Pipeline de NLP y Preprocesamiento Especializado**
  * *Contenido:* Diagrama del flujo de 5 fases: Normalización (emojis gastronómicos, signos de exclamación), Tokenización (compuestos culinarios, contracciones), Stopwords selectivas (conservando negaciones como *"no, ni, nunca, sin, tampoco"*), Reducción léxica y Vectorización TF-IDF.
  * *Highlight:* Justificar por qué no se pueden borrar las negaciones en gastronomía (*"No me gustó"* vs *"Me gustó"*).

* **Diapositiva B: Capa 3 — Regresión Logística y Optimización de Hiperparámetros**
  * *Contenido:* Modelo supervisado multiclase (*positivo*, *neutral*, *negativo*).
  * *Highlight:* Uso de `GridSearchCV` con `StratifiedKFold(n_splits=5)`. Parámetro `class_weight='balanced'` para contrarrestar el desbalance natural de reseñas (donde abundan las positivas).
  * *Métricas logradas:* Exactitud: **69.86%** (+16.44% sobre Dummy Baseline) | F1-Score ponderado: **69.80%** (+32.59% sobre Dummy).

* **Diapositiva C: Capa 5 — Interfaz Web Interactiva y Dashboard**
  * *Contenido:* Screenshots del Frontend en acción: 4 pestañas (Analizador en vivo, Visualizador NLP paso a paso, Dashboard de métricas y Casos de error).
  * *Highlight:* Capacidad de auto-recuperación y modo local cuando el backend no está disponible.

---

## ⏱️ 2. Guion de Exposición Individual (~3 minutos)

> **Consejo:** Habla con seguridad, con el código y la terminal abiertos de fondo.

### Minuto 0:00 - 1:00 | Mis aportes en el Pipeline de NLP
> *"Buenas tardes profesor/a. En el marco del proyecto SaborIA, mi rol principal abarcó el desarrollo del pipeline de procesamiento de lenguaje natural en `src/nlp/`, el modelado y optimización de Regresión Logística en `src/models/`, y la interfaz interactiva en `frontend/`.*
> 
> *En NLP implementé la normalización y tokenización en `normalizer.py` y `tokenizer.py`. Diseñamos reglas específicas para el dominio restaurantero: convertimos emojis de comida y emoción en tokens semánticos (por ejemplo '😋' pasa a ser interpretado como refuerzo positivo) y cuidamos que contracciones y signos repetidos no fragmentaran las palabras. Además, en `vectorizer.py` y `ngram_comparator.py` configuré la matriz TF-IDF con sublinear TF y evalué el impacto de unigramas frente a bigramas."*

### Minuto 1:00 - 2:00 | Regresión Logística y Validación Rigurosa
> *"En la capa de Machine Learning, desarrollé `ClasificadorRegresionLogistica` en `src/models/logistic_regression.py`. Para no depender de parámetros por defecto, implementé optimización con `GridSearchCV` sobre validación cruzada estratificada de 5 pliegues.*
> 
> *Exploramos diferentes valores del parámetro de regularización $C$, pesos de clase `balanced` y tamaños de vocabulario. Al evaluar en nuestro conjunto de prueba ciego de 146 reseñas, el modelo alcanzó un **Accuracy del 69.86%** y un **F1 ponderado del 69.80%**, superando con creces al Baseline Dummy que solo obtenía 53.42% y 37.21% de F1. Además, serializamos el pipeline entrenado con `joblib` para garantizar reproducibilidad total con un único comando: `python -m src.models.train`."*

### Minuto 2:00 - 3:00 | Frontend y Experiencia de Usuario
> *"Finalmente, desarrollé la aplicación frontend en `frontend/` con cuatro pestañas: un clasificador en tiempo real con distribución de probabilidades, un visualizador que desglosa en vivo las 5 etapas del pipeline de NLP, un dashboard con la matriz de confusión interactiva y un visor de casos de estudio sobre fallos cualitativos como sarcasmo y negación compleja. Todo el código cuenta con su suite de pruebas unitarias en `tests/` que validan el comportamiento con 100% de éxito."*

---

## 💡 3. Banco de Preguntas Conceptuales del Docente (10 pts)

### P1: ¿Por qué eligieron Regresión Logística y no directamente una red neuronal o Deep Learning?
**Respuesta:**  
*"Para clasificación de texto con datasets de escala pequeña a mediana (~730 reseñas), la Regresión Logística con TF-IDF es un estándar de la industria: es sumamente rápida de entrenar e inferir (milisegundos), no sufre de sobreajuste severo como las redes profundas con pocos datos, es matemáticamente interpretable mediante los pesos asignados a cada término/n-grama y ofrece probabilidades calibradas vía Softmax."*

### P2: ¿Qué significa el parámetro `C` en Regresión Logística?
**Respuesta:**  
*"El parámetro $C$ es el inverso de la fuerza de regularización ($C = 1/\lambda$). Un valor pequeño de $C$ aplica una penalización muy fuerte a los coeficientes grandes, simplificando el modelo para evitar sobreajuste. Un valor grande de $C$ le da más libertad al modelo para ajustarse a los datos de entrenamiento. En nuestro `GridSearchCV` evaluamos `[0.1, 1.0, 5.0, 10.0]` y encontramos que $C=1.0$ ofrecía el mejor equilibrio de generalización."*

### P3: ¿Por qué usaron `class_weight='balanced'`?
**Respuesta:**  
*"Porque en reseñas gastronómicas reales existe un sesgo natural: la mayoría de los comensales escriben para elogiar (positivo) o para quejarse agriamente (negativo), siendo la clase 'neutral' minoritaria. `class_weight='balanced'` ajusta automáticamente las ponderaciones inversamente proporcionales a las frecuencias de clase en los datos, evitando que el modelo ignore las clases menos frecuentes."*

### P4: ¿Por qué es fundamental conservar las palabras de negación en las Stop Words?
**Respuesta:**  
*"Las listas tradicionales de stopwords en español (como NLTK estándar) eliminan 'no', 'sin', 'nunca', 'tampoco'. Si el cliente escribe 'No me gustó la hamburguesa' y se borra 'no', el vectorizador se queda únicamente con 'gustó' y 'hamburguesa', clasificando erróneamente una queja como positiva. En SaborIA preservamos explícitamente estas partículas."*

### P5: ¿Por qué el F1-Score ponderado es más confiable que el simple Accuracy aquí?
**Respuesta:**  
*"El Accuracy mide aciertos globales, por lo que una clase mayoritaria infla el resultado. El F1 ponderado calcula la media armónica entre precisión y recall para cada clase por separado y luego hace el promedio ponderado por el número de instancias reales de cada una, castigando si el modelo tiene un pésimo desempeño en una clase en particular."*

---

## 🛠️ 4. Guía Maestra para la Modificación en Vivo (15 pts)

Durante la defensa, el docente pedirá realizar un cambio en vivo, ejecutarlo y demostrar el resultado en Git. Aquí están los escenarios exactos:

### Escenario A: "Agregue o quite una stopword y muestre el cambio"
1. Abre [src/stopwords_es.py](file:///c:/Users/Roberto/Desktop/PROYECTO%20DE%20TOPICOS/PROY-TOPIC/src/stopwords_es.py).
2. Si pide quitar/agregar, edita el conjunto `STOPWORDS_PERSONALIZADAS`:
   ```python
   # Ejemplo: quitar "muy" para que se conserve en el análisis
   STOPWORDS_PERSONALIZADAS.discard("muy")
   ```
3. Ejecuta el test para verificar:
   ```powershell
   .\.venv\Scripts\python -m pytest tests/test_pipeline_nlp.py
   ```
4. Commitea el cambio:
   ```powershell
   git add src/stopwords_es.py
   git commit -m "fix(nlp): ajustar stopwords personalizadas en vivo para defensa"
   ```

### Escenario B: "Modifique el rango de n-gramas a bigramas/trigramas y reentrene"
1. Abre [src/models/logistic_regression.py](file:///c:/Users/Roberto/Desktop/PROYECTO%20DE%20TOPICOS/PROY-TOPIC/src/models/logistic_regression.py#L55).
2. Modifica el parámetro `rango_ngramas`:
   ```python
   rango_ngramas: Tuple[int, int] = (1, 3) # Unigramas hasta trigramas
   ```
3. Reentrena en 5 segundos:
   ```powershell
   .\.venv\Scripts\python -m src.models.train
   ```
4. Muestra la nueva salida de métricas en consola y commitea:
   ```powershell
   git commit -am "feat(ml): ampliar rango de n-gramas a trigramas en vivo"
   ```

### Escenario C: "Modifique la regularización C a 0.01 o 10.0"
1. Abre [src/models/logistic_regression.py](file:///c:/Users/Roberto/Desktop/PROYECTO%20DE%20TOPICOS/PROY-TOPIC/src/models/logistic_regression.py#L53).
2. Cambia `c_param = 10.0`.
3. Reentrena:
   ```powershell
   .\.venv\Scripts\python -m src.models.train
   ```
4. Muestra cómo variaron las métricas y commitea:
   ```powershell
   git commit -am "feat(ml): ajustar hiperparametro C a 10.0 en vivo"
   ```

### Escenario D: "Cambie el color o umbral en el Frontend"
1. Abre [frontend/styles.css](file:///c:/Users/Roberto/Desktop/PROYECTO%20DE%20TOPICOS/PROY-TOPIC/frontend/styles.css) o [frontend/app.js](file:///c:/Users/Roberto/Desktop/PROYECTO%20DE%20TOPICOS/PROY-TOPIC/frontend/app.js).
2. Modifica el estilo o mensaje deseado y recarga el navegador con `Ctrl + F5`.
3. Commitea:
   ```powershell
   git commit -am "feat(frontend): actualizar estilo interactivo en vivo"
   ```

---

## 🚀 5. Comandos de Supervivencia para la Defensa

```powershell
# 1. Correr todas las pruebas (demostración de calidad)
.\.venv\Scripts\python -m pytest

# 2. Reentrenar el modelo de punta a punta (demostración de reproducibilidad)
.\.venv\Scripts\python -m src.models.train

# 3. Mostrar el historial impecable de commits de Roberto
git log --author="Roberto" --oneline -n 12

# 4. Abrir la interfaz web en el navegador
Start-Process .\frontend\index.html
```
