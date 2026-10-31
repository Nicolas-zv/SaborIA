# Comparación de Stemming y Lematización — SaborIA

## 1. Introducción

El Procesamiento de Lenguaje Natural (NLP) permite analizar y transformar textos escritos en lenguaje humano para facilitar su procesamiento mediante algoritmos de Machine Learning.

En el proyecto **SaborIA**, un sistema inteligente de análisis de sentimientos de reseñas de restaurantes, se implementaron y compararon dos técnicas de procesamiento lingüístico: **stemming y lematización**.

Ambas técnicas permiten reducir las variaciones de las palabras, aunque utilizan procedimientos diferentes y pueden producir resultados distintos.

## 2. Objetivo

Comparar el funcionamiento del stemming y la lematización sobre textos en español, utilizando ejemplos manuales y reseñas del dataset de SaborIA, para identificar sus características, ventajas y limitaciones antes del entrenamiento de modelos de clasificación de sentimientos.

## 3. Herramientas utilizadas

- **Python:** lenguaje de programación utilizado para implementar el procesamiento.
- **NLTK:** biblioteca utilizada para aplicar stemming mediante `SnowballStemmer("spanish")`.
- **spaCy:** biblioteca utilizada para realizar lematización.
- **es_core_news_sm:** modelo lingüístico de spaCy para español.
- **Pandas:** biblioteca utilizada para cargar y consultar el dataset.

El código fue desarrollado en el archivo `src/comparar_stemming_lematizacion.py`.

## 4. Stemming

El stemming es una técnica de NLP que reduce las palabras a raíces aproximadas mediante reglas de transformación.

No necesariamente produce palabras válidas del idioma, ya que su finalidad es agrupar diferentes formas de una palabra bajo una representación común.

Por ejemplo:

| Palabra original | Resultado |
|---|---|
| clientes | client |
| comiendo | com |
| comidas | com |
| camareros | camarer |
| atendieron | atend |
| excelentes | excelent |

### Ventajas

- Permite reducir variaciones del vocabulario.
- Produce representaciones compactas de las palabras.
- Puede resultar útil para modelos de clasificación basados en características textuales.

### Limitaciones

- Puede generar raíces que no son palabras completas.
- Puede perder información morfológica.
- Algunas transformaciones pueden dificultar la interpretación de los resultados.

## 5. Lematización

La lematización es una técnica que busca transformar las palabras a su forma base o lema, utilizando información lingüística.

A diferencia del stemming, intenta producir palabras interpretables del idioma.

Por ejemplo:

| Palabra original | Resultado |
|---|---|
| clientes | cliente |
| comiendo | comer |
| comidas | comida |
| camareros | camarero |
| atendieron | atender |
| excelentes | excelente |

### Ventajas

- Produce resultados generalmente más interpretables.
- Permite relacionar diferentes formas de una misma palabra.
- Considera información lingüística durante el procesamiento.

### Limitaciones

- Requiere recursos y modelos lingüísticos.
- Puede demandar más procesamiento computacional.
- No garantiza resultados correctos para todas las palabras.

## 6. Metodología de implementación

Para desarrollar la comparación se realizaron las siguientes actividades:

1. Se instaló la biblioteca spaCy y el modelo `es_core_news_sm`.
2. Se configuró el algoritmo `SnowballStemmer` de NLTK para español.
3. Se implementó la función `aplicar_stemming(texto)`.
4. Se implementó la función `aplicar_lematizacion(texto)`.
5. Se realizaron pruebas con cuatro oraciones de ejemplo.
6. Se cargó el dataset `data/resenas_restaurantes.csv`.
7. Se seleccionaron cinco reseñas del dataset y se aplicaron ambas técnicas.
8. Se compararon los resultados obtenidos.

En ambos procedimientos se utilizaron palabras en minúsculas y se excluyeron los tokens que no fueran alfabéticos.

## 7. Resultados con ejemplos manuales

Se utilizaron cuatro oraciones relacionadas con restaurantes para observar las diferencias entre ambas técnicas.

### Ejemplo 1

**Texto original:**

Los clientes estaban comiendo deliciosas comidas.

**Stemming:**

`los client estab com delici com`

**Lematización:**

`el cliente estar comer deliciosa comida`

Se observa que el stemming reduce las palabras a raíces, mientras que la lematización transforma las palabras a formas base más interpretables.

### Ejemplo 2

**Texto original:**

Los camareros atendieron rápidamente las mesas.

**Stemming:**

`los camarer atend rapid las mes`

**Lematización:**

`el camarero atender rápidamente el mesa`

El stemming reduce palabras como `camareros` y `atendieron` a `camarer` y `atend`. La lematización produce `camarero` y `atender`.

### Ejemplo 3

**Texto original:**

Nunca volvería porque los platos estaban fríos.

**Stemming:**

`nunc volv porqu los plat estab fri`

**Lematización:**

`nunca volver porque el plato estar frío`

En este ejemplo se observa que el stemming transforma `nunca` en `nunc`, mientras que la lematización conserva `nunca`.

Esto es relevante para el análisis de sentimientos, ya que las negaciones pueden modificar el significado de una reseña.

### Ejemplo 4

**Texto original:**

El restaurante ofrece comidas excelentes.

**Stemming:**

`el restaur ofrec com excelent`

**Lematización:**

`el restaurante ofrecer comida excelente`

La lematización conserva palabras más fáciles de interpretar, mientras que el stemming utiliza raíces aproximadas.

## 8. Resultados con el dataset de SaborIA

Se seleccionaron las primeras cinco reseñas del archivo `resenas_restaurantes.csv`, que contiene 730 reseñas sintéticas de restaurantes en español.

### Reseña 1 — Sentimiento negativo

**Original:**

Mucho minimalismo pero las sillas son insufribles para una cena de dos horas. La ergonomía no debería sacrificarse por el diseño.

**Stemming:**

`much minimal per las sill son insufr par una cen de dos hor la ergonom no deb sacrific por el diseñ`

**Lematización:**

`mucho minimalismo pero el silla ser insufribl para uno cena de dos hora el ergonomía no deber sacrificar él por el diseño`

### Reseña 2 — Sentimiento positivo

**Original:**

Acústica muy cuidada, se puede hablar sin problemas. La música ambiente está a un volumen ideal.

**Stemming:**

`acust muy cuid se pued habl sin problem la music ambient esta a un volum ideal`

**Lematización:**

`acústicar mucho cuidado él poder hablar sin problema el música ambiente estar a uno volumen ideal`

### Reseña 3 — Sentimiento positivo

**Original:**

Incluso el servicio de pan y aceites es una experiencia gourmet por derecho propio.

**Stemming:**

`inclus el servici de pan y aceit es una experient gourmet por derech propi`

**Lematización:**

`incluso el servicio de pan y aceite ser uno experiencia gourmet por derecho propio`

### Reseña 4 — Sentimiento positivo

**Original:**

Atmósfera selecta y cuidada. La secuencia de aperitivos fue el punto álgido del menú.

**Stemming:**

`atmosfer select y cuid la secuenci de aperit fue el punt algid del menu`

**Lematización:**

`atmósferar selecta y cuidado el secuencia de aperitivo ser el punto álgido del menú`

### Reseña 5 — Sentimiento positivo

**Original:**

La limpieza, el orden y la estética del local son simplemente impecables.

**Stemming:**

`la limpiez el orden y la estet del local son simplement impec`

**Lematización:**

`el limpieza el orden y el estética del local ser simplemente impecable`

## 9. Análisis comparativo

Los resultados permiten identificar diferencias entre las técnicas:

| Criterio | Stemming | Lematización |
|---|---|---|
| Herramienta | NLTK | spaCy |
| Método | Reducción mediante reglas | Análisis lingüístico |
| Resultado | Raíces aproximadas | Lemas |
| Interpretabilidad | Menor | Generalmente mayor |
| Palabras completas | No siempre | Generalmente |
| Posibles errores | Reducción excesiva | Lemas incorrectos |

Durante las pruebas se observaron transformaciones adecuadas, como `comiendo` → `comer` y `excelentes` → `excelente`.

Sin embargo, también se identificaron errores de lematización, como:

- `Acústica` → `acústicar`.
- `Atmósfera` → `atmósferar`.
- `insufribles` → `insufribl`.

Estos resultados demuestran que la lematización no siempre produce formas lingüísticamente correctas.

También se observó que ambas técnicas pueden conservar negaciones, aunque el stemming puede modificar su forma, como sucede con `nunca` → `nunc`.

## 10. Importancia para el análisis de sentimientos

En un sistema de análisis de sentimientos es importante conservar información que permita distinguir opiniones positivas, negativas y neutrales.

Por ejemplo, las expresiones `bueno` y `no bueno` pueden representar valoraciones diferentes.

Por esta razón, SaborIA también implementa un módulo de eliminación de stop words que conserva negaciones relevantes, como `no`, `nunca`, `sin` y `tampoco`.

La integración de las técnicas deberá considerar el orden del preprocesamiento y sus posibles efectos sobre el significado de las reseñas.

## 11. Aplicación dentro de SaborIA

Las funciones implementadas permiten procesar textos mediante stemming o lematización.

Estas funciones podrán utilizarse posteriormente en experimentos de clasificación supervisada, junto con técnicas de vectorización como TF-IDF.

En las siguientes etapas se entrenarán y evaluarán modelos Naive Bayes y regresión logística.

La comparación del rendimiento permitirá determinar qué configuración de preprocesamiento resulta más adecuada para el dataset utilizado.

## 12. Conclusiones

La implementación permitió comprobar que el stemming y la lematización son técnicas diferentes para reducir las variaciones de las palabras en español.

El stemming genera raíces aproximadas, mientras que la lematización produce resultados generalmente más interpretables.

Las pruebas también evidenciaron que el modelo lingüístico de spaCy puede cometer errores al determinar algunos lemas.

Por lo tanto, no es posible afirmar únicamente con estas pruebas que una técnica produzca mejores resultados de clasificación que la otra.

**Se concluye que ambas técnicas deben evaluarse experimentalmente durante el entrenamiento de los modelos de Machine Learning**, utilizando métricas como accuracy, precisión, recall y F1-score.

## 13. Referencias

- NLTK. Documentación oficial: https://www.nltk.org/
- spaCy. Documentación oficial: https://spacy.io/
- Dataset HorecaReviews-ES: https://huggingface.co/datasets/mrcsgh/horeca-spanish-reviews
- Scikit-learn. Documentación oficial: https://scikit-learn.org/