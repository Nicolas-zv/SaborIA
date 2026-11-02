# Análisis de errores del modelo Naive Bayes — SaborIA

## 1. Objetivo

Analizar las predicciones incorrectas del modelo Multinomial Naive Bayes desarrollado para SaborIA, identificando patrones de error y limitaciones en la clasificación automática de reseñas de restaurantes en español.

## 2. Descripción del experimento

Se utilizó un dataset de 730 reseñas sintéticas de restaurantes en español, clasificadas en tres categorías: positivo, negativo y neutral.

Los datos se dividieron mediante una partición estratificada:

- Entrenamiento: 584 reseñas (80%).
- Prueba: 146 reseñas (20%).
- Semilla aleatoria: 42.

Se implementó un pipeline con TF-IDF y Multinomial Naive Bayes.

Para seleccionar los hiperparámetros se utilizó GridSearchCV con cinco particiones de validación cruzada y la métrica F1 macro.

## 3. Hiperparámetros seleccionados

Los mejores hiperparámetros según validación cruzada fueron:

- `alpha = 0.1`
- `fit_prior = False`
- `ngram_range = (1, 2)`

El mejor F1 macro promedio en validación cruzada fue de 0.5918.

## 4. Resultados de evaluación

El modelo seleccionado obtuvo los siguientes resultados sobre el conjunto de prueba:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.6986 |
| Precision macro | 0.59 |
| Recall macro | 0.58 |
| F1 macro | 0.58 |
| F1 ponderado | 0.68 |

### Resultados por sentimiento

| Sentimiento | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negativo | 0.65 | 0.62 | 0.63 |
| Neutral | 0.30 | 0.23 | 0.26 |
| Positivo | 0.81 | 0.90 | 0.85 |

La clase positiva presentó el mejor desempeño, mientras que la clase neutral resultó la más difícil de identificar.

## 5. Matriz de confusión

La matriz de confusión obtenida fue:

| Clase real / Predicha | Negativo | Neutral | Positivo |
|---|---:|---:|---:|
| Negativo | 26 | 10 | 6 |
| Neutral | 10 | 6 | 10 |
| Positivo | 4 | 4 | 70 |

El modelo clasificó correctamente 102 reseñas y cometió 44 errores.

La clase neutral fue confundida con las clases negativa y positiva en 20 de sus 26 ejemplos.

## 6. Análisis cualitativo de errores

Se revisaron los primeros diez errores generados por el modelo.

### 6.1. Lenguaje figurado

**Reseña:** «La carne estaba tan tierna que sospecho que fue masajeada por ángeles antes de llegar a la mesa».

- Sentimiento real: positivo.
- Sentimiento predicho: negativo.

La reseña utiliza una expresión figurada para transmitir una valoración favorable. El modelo no identificó correctamente el sentido positivo de la frase.

### 6.2. Opiniones mixtas

**Reseña:** «Presentaciones increíbles. Algo ruidoso el comedor».

- Sentimiento real: positivo.
- Sentimiento predicho: neutral.

La reseña contiene un elogio y una observación negativa. La combinación de ambas valoraciones puede dificultar la clasificación.

### 6.3. Contrastes dentro de una oración

**Reseña:** «Ambiente sofisticado, pero el trato del personal de sala fue un tanto distante».

- Sentimiento real: neutral.
- Sentimiento predicho: positivo.

La reseña presenta un aspecto positivo seguido de una crítica. El modelo pudo haber dado mayor importancia a las palabras positivas que al contraste introducido por «pero».

### 6.4. Críticas indirectas

**Reseña:** «Las copas para el maridaje eran mediocres. Para el precio que cobran, deberían usar cristalería de mayor calidad que potencie el vino».

- Sentimiento real: negativo.
- Sentimiento predicho: neutral.

La crítica se expresa mediante una valoración desfavorable y una recomendación de mejora. El modelo no logró identificar correctamente la etiqueta negativa.

### 6.5. Intensidad de las críticas

**Reseña:** «El pescado estaba ligeramente pasado de cocción. Un error imperdonable a este nivel».

- Sentimiento real: neutral.
- Sentimiento predicho: negativo.

Aunque la etiqueta del dataset es neutral, el texto contiene una expresión de desaprobación intensa. Esto evidencia que la clasificación basada en estrellas puede no coincidir completamente con el tono aparente del texto.

## 7. Principales dificultades identificadas

A partir de los ejemplos revisados se identificaron las siguientes dificultades:

1. Interpretación de lenguaje figurado.
2. Clasificación de opiniones mixtas.
3. Reconocimiento de contrastes entre aspectos positivos y negativos.
4. Identificación de críticas indirectas.
5. Diferencias entre el tono del texto y la etiqueta derivada de estrellas.
6. Bajo rendimiento en la categoría neutral.

Estas observaciones corresponden a un análisis cualitativo de diez errores y no representan una medición estadística de la frecuencia de cada causa.

## 8. Posibles mejoras

Para futuras versiones de SaborIA se recomienda:

- Comparar el modelo Naive Bayes con regresión logística.
- Evaluar diferentes configuraciones de TF-IDF.
- Experimentar con técnicas de normalización, stemming y lematización.
- Analizar estrategias para mejorar la clasificación de reseñas neutrales.
- Considerar modelos capaces de capturar mejor el contexto de las oraciones.
- Evaluar el sistema con reseñas reales y etiquetas revisadas manualmente.

## 9. Limitaciones del experimento

El dataset utilizado contiene reseñas sintéticas, por lo que los resultados pueden no representar el desempeño del sistema con opiniones reales.

Las etiquetas de sentimiento se derivaron de las calificaciones por estrellas.

Además, algunas configuraciones se exploraron observando repetidamente el mismo conjunto de prueba. Por ello, sus resultados no deben interpretarse como una evaluación final completamente independiente.

La selección mediante GridSearchCV se realizó utilizando únicamente el conjunto de entrenamiento.

## 10. Conclusión

El modelo Multinomial Naive Bayes alcanzó un accuracy de 69.86% y un F1 macro aproximado de 0.58 sobre el conjunto de prueba.

El análisis permitió identificar que la principal dificultad del modelo se encuentra en la clasificación de reseñas neutrales y en textos con valoraciones mixtas o expresiones complejas.

Los resultados constituyen una referencia inicial para comparar el desempeño de otros modelos de Machine Learning dentro del proyecto SaborIA.