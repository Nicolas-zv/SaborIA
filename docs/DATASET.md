# Documentación del Dataset — SaborIA

## 1. Nombre del dataset

HorecaReviews-ES.

## 2. Fuente de los datos

Hugging Face: https://huggingface.co/datasets/mrcsgh/horeca-spanish-reviews

## 3. Descripción

El dataset contiene reseñas sintéticas en español de distintos establecimientos gastronómicos, como restaurantes, cafeterías, mesones y cervecerías.

Para el proyecto SaborIA se seleccionaron únicamente los registros correspondientes a restaurantes.

## 4. Cantidad de registros

- Registros originales: 5.717.
- Reseñas seleccionadas de restaurantes: 730.
- Idioma: español.
- Categorías de sentimiento: positivo, negativo y neutral.

## 5. Variables

| Variable | Descripción |
|---|---|
| negocio | Tipo de establecimiento gastronómico. |
| texto | Contenido de la reseña. |
| estrellas | Calificación numérica del establecimiento. |
| estilo | Estilo de redacción de la reseña. |
| sentimiento | Etiqueta de sentimiento generada a partir de las estrellas. |

## 6. Criterio de clasificación

- 1 y 2 estrellas: negativo.
- 3 estrellas: neutral.
- 4 y 5 estrellas: positivo.

## 7. Distribución de sentimientos

| Sentimiento | Cantidad | Porcentaje |
|---|---:|---:|
| Positivo | 391 | 53,6 % |
| Negativo | 209 | 28,6 % |
| Neutral | 130 | 17,8 % |
| **Total** | **730** | **100 %** |

## 8. Procesamiento realizado

Se descargó el dataset desde Hugging Face y se interpretaron los campos separados por tabulaciones. Posteriormente se seleccionaron las reseñas identificadas como restaurantes, se eliminaron registros incompletos y se generó la columna de sentimiento.

El resultado se almacenó en `data/resenas_restaurantes.csv`.

## 9. Limitaciones

El dataset contiene reseñas sintéticas, por lo que no representa necesariamente las opiniones de clientes reales.

Las etiquetas de sentimiento se generan a partir de la calificación en estrellas y no de una anotación humana independiente.

Existe un desbalance entre las clases, principalmente por la mayor cantidad de reseñas positivas.

Estas limitaciones deberán considerarse al evaluar e interpretar los resultados de los modelos.

## 10. Uso dentro del proyecto

Este dataset se utilizará para el análisis exploratorio, el preprocesamiento de lenguaje natural, la vectorización de textos y el entrenamiento y evaluación de los modelos de clasificación supervisada de SaborIA.