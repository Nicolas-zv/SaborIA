# Comparación de modelos de Machine Learning — SaborIA

## 1. Introducción

En el proyecto SaborIA se desarrollaron y evaluaron dos algoritmos de aprendizaje automático supervisado para clasificar reseñas de restaurantes en español según tres categorías de sentimiento: **positivo, negativo y neutral**.

Los modelos implementados fueron:

- **Naive Bayes Multinomial:** algoritmo probabilístico utilizado habitualmente en clasificación de textos.
- **Regresión Logística:** algoritmo de clasificación lineal que estima la probabilidad de pertenencia a cada categoría.

El objetivo de la comparación fue determinar cuál de los modelos ofrece un mejor rendimiento para su integración en la aplicación web.

## 2. Dataset utilizado

Se utilizó el dataset `mrcsgh/horeca-spanish-reviews`, disponible en Hugging Face.

Después de filtrar las reseñas correspondientes a restaurantes, se obtuvieron **730 registros**, distribuidos de la siguiente manera:

| Sentimiento | Cantidad | Porcentaje |
|---|---:|---:|
| Positivo | 391 | 53,56 % |
| Negativo | 209 | 28,63 % |
| Neutral | 130 | 17,81 % |
| **Total** | **730** | **100 %** |

Las etiquetas de sentimiento se obtuvieron a partir de las puntuaciones en estrellas: 1–2 para negativo, 3 para neutral y 4–5 para positivo.

El conjunto de datos contiene reseñas sintéticas, por lo que los resultados deben interpretarse considerando esta limitación.

## 3. Preparación de los datos

Para el entrenamiento y la evaluación se utilizó una división estratificada:

- **80 % para entrenamiento:** 584 reseñas.
- **20 % para pruebas:** 146 reseñas.
- **Semilla aleatoria:** `random_state=42`.

La estratificación permitió conservar aproximadamente la proporción original de cada sentimiento.

Los textos se representaron mediante TF-IDF, una técnica que transforma los documentos en características numéricas para su procesamiento por los modelos.

## 4. Entrenamiento y optimización

### 4.1. Naive Bayes Multinomial

Se utilizó `GridSearchCV` con cinco particiones de validación cruzada y optimización mediante F1 macro.

Los mejores hiperparámetros encontrados fueron:

- `alpha = 0.1`
- `fit_prior = False`
- `ngram_range = (1, 2)`

El mejor F1 macro durante la validación cruzada fue **0,5918**.

### 4.2. Regresión Logística

Se entrenó un clasificador de regresión logística con búsqueda de hiperparámetros mediante validación cruzada.

Entre los parámetros seleccionados se encuentran:

- `C = 1`
- `class_weight = balanced`
- `max_features = 1500`
- `ngram_range = (1, 1)`

La búsqueda utilizó F1 ponderado como criterio de optimización, obteniendo aproximadamente **0,7096** en validación cruzada.

Es importante aclarar que los valores de validación cruzada de ambos modelos no son directamente comparables porque se optimizaron métricas diferentes.

## 5. Resultados en el conjunto de prueba

| Métrica | Naive Bayes | Regresión Logística |
|---|---:|---:|
| Exactitud (Accuracy) | 69,86 % | 69,86 % |
| Precisión macro | 59 % | 64,23 % |
| F1 macro | 58 % | 63,88 % |
| F1 ponderado | 68 % | 69,80 % |
| Aciertos en reseñas neutrales | 6 de 26 | 12 de 26 |

Aunque ambos modelos obtuvieron la misma exactitud, regresión logística consiguió mejores resultados en F1 macro y F1 ponderado.

### 5.1. Matriz de confusión — Naive Bayes

| Real / Predicho | Negativo | Neutral | Positivo |
|---|---:|---:|---:|
| Negativo | 26 | 10 | 6 |
| Neutral | 10 | 6 | 10 |
| Positivo | 4 | 4 | 70 |

### 5.2. Matriz de confusión — Regresión Logística

| Real / Predicho | Negativo | Neutral | Positivo |
|---|---:|---:|---:|
| Negativo | 27 | 6 | 9 |
| Neutral | 9 | 12 | 5 |
| Positivo | 10 | 5 | 63 |

Las filas representan las etiquetas reales y las columnas las predicciones.

## 6. Análisis de resultados

Los dos modelos alcanzaron una exactitud cercana al 70 %, equivalente a 102 predicciones correctas de las 146 reseñas evaluadas.

Sin embargo, la exactitud no permite observar completamente las diferencias de rendimiento entre clases, especialmente cuando el conjunto de datos está desbalanceado.

El F1 macro asigna la misma importancia a cada clase, por lo que resulta útil para evaluar el desempeño general en sentimientos positivos, negativos y neutrales.

Regresión logística obtuvo un F1 macro superior y mejoró el reconocimiento de reseñas neutrales, pasando de 6 aciertos con Naive Bayes a 12.

Por otra parte, Naive Bayes reconoció correctamente más reseñas positivas: 70 frente a 63 de regresión logística.

## 7. Limitaciones identificadas

La principal dificultad fue la clasificación del sentimiento neutral.

Algunas reseñas contienen expresiones mixtas, ironía, críticas moderadas o términos que pueden interpretarse de diferentes maneras.

Además, el conjunto de datos presenta un desequilibrio entre categorías: las reseñas positivas representan más de la mitad de los registros, mientras que las neutrales constituyen menos de una quinta parte.

Otra limitación es el uso de reseñas sintéticas y etiquetas derivadas de puntuaciones en estrellas, que no siempre reflejan todos los matices del lenguaje.

## 8. Selección del modelo final

Se seleccionó **regresión logística** como modelo principal de SaborIA debido a los siguientes resultados:

1. Alcanzó la misma exactitud que Naive Bayes: 69,86 %.
2. Obtuvo un F1 macro superior: 63,88 % frente a aproximadamente 58 %.
3. Mejoró el reconocimiento de reseñas neutrales: 12 de 26 frente a 6 de 26.
4. Ofrece probabilidades por categoría, útiles para presentar los resultados en la interfaz web.

La selección prioriza el equilibrio entre las tres clases en lugar de considerar únicamente la exactitud.

Naive Bayes se conserva como modelo alternativo y referencia comparativa.

## 9. Conclusiones

La experimentación permitió comprobar que diferentes algoritmos pueden obtener la misma exactitud y, aun así, presentar comportamientos distintos según la categoría analizada.

En SaborIA, regresión logística demostró un mejor equilibrio general entre sentimientos positivos, negativos y neutrales, por lo que fue seleccionada para la integración con FastAPI y el frontend.

Como mejoras futuras se propone ampliar el dataset con reseñas reales, revisar la calidad de las etiquetas y experimentar con estrategias adicionales para mejorar la clasificación neutral.

Los resultados obtenidos corresponden a un prototipo académico y no deben interpretarse como garantía de rendimiento en todos los restaurantes o contextos de uso.