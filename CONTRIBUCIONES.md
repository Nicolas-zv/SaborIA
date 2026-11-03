# Registro de Contribuciones y Flujo de Colaboración
**Proyecto:** SaborIA – Sistema Inteligente de Análisis de Sentimientos de Reseñas de Restaurantes  
**Materia:** Tópicos y Programación Avanzada  

---

## 1. Integrantes del Equipo

| Nombre | Usuario GitHub / Git | Correo | Rol Principal |
| :--- | :--- | :--- | :--- |
| **Nicolás** | `@Nicolas-zv` | *(Configurado en repo)* | Administrador del Repositorio / Datos, NLP & Backend |
| **Roberto** | `@Roberto` | `torrezroberto9021@gmail.com` | NLP, Modelado ML & Frontend |

---

## 2. Flujo de Trabajo y Buenas Prácticas de Git

Para garantizar la trazabilidad exigida en la rúbrica de evaluación (20 puntos individuales de aporte en Git):

1. **Ramas independientes:** Ningún integrante hace push directo a `main`. Cada funcionalidad se desarrolla en ramas específicas (`robert-dev` o `feat/...`).
2. **Pull Requests (PR):**
   - Roberto desarrolla en su rama y crea Pull Requests hacia `main`.
   - Nicolás revisa los cambios propuestos y realiza la integración (*merge*) a `main`.
3. **Formato de Commits (Conventional Commits):**
   - `feat:` Nuevas funcionalidades o transformadores de datos/modelos.
   - `fix:` Corrección de errores o validaciones.
   - `docs:` Modificaciones en documentación o informes.
   - `test:` Pruebas unitarias o de integración.
   - `chore:` Tareas de configuración, dependencias o herramientas de entorno.
4. **Commits atómicos y frecuentes:** Cada commit debe representar un avance claro y puntual con su explicación.
5. **Pair Programming:** Cuando se trabaje en conjunto en una sesión, se incluirá `Co-authored-by: Nombre <correo>` en el mensaje del commit.

---

## 3. Plan de Trabajo Distribuido por Etapas

| Etapa | Responsable: Nicolás | Responsable: Roberto |
| :--- | :--- | :--- |
| **Etapa 1: Configuración Inicial** | • Estructura base de carpetas y `.gitignore`<br>• Creación de `README.md` y `requirements.txt` | • Clonado y configuración del entorno virtual local (`.venv`)<br>• Creación de `CONTRIBUCIONES.md` y guía de colaboración |
| **Etapa 2: Dataset y EDA** | • Selección y documentación del dataset gastronómico<br>• Análisis exploratorio (EDA): clases, longitud y nulos | • Implementación de la normalización de textos<br>• Implementación de la tokenización y casos difíciles |
| **Etapa 3: Pipeline de NLP** | • Configuración de stop words (conservando negaciones)<br>• Comparación experimental de stemming vs. lematización | • Implementación de vectorización TF-IDF documentada<br>• Comparación experimental de unigramas vs. bigramas |
| **Etapa 4: Machine Learning** | • Entrenamiento y evaluación de Naive Bayes<br>• Análisis cualitativo de errores (falsos positivos/negativos) | • Entrenamiento y evaluación de Regresión Logística<br>• Ajuste de hiperparámetros con validación cruzada (`GridSearchCV`) |
| **Etapa 5: Aplicación** | • Script de entrenamiento reproducible y serialización (`joblib`)<br>• Backend API REST con FastAPI (`/predict`, `/preprocess`) | • Diseño del formulario web para reseñas en Frontend<br>• Visualizador paso a paso del pipeline de NLP |
| **Etapa 6: Integración y Tests** | • Validaciones de entrada en la API<br>• Pruebas automatizadas del backend con `pytest` | • Conexión del Frontend con la API REST<br>• Dashboard de métricas del modelo y comparativa |
| **Etapa 7: Entrega y Defensa** | • Redacción del informe técnico final en `docs/`<br>• Verificación de reproducibilidad de instalación | • Actualización final del registro en `CONTRIBUCIONES.md`<br>• Preparación de diapositivas y guion de defensa |

---

## 4. Historial Detallado de Commits por Integrante

### Contribuciones de Nicolás (`@Nicolas-zv`)
- `Initial commit` / `Crear estructura inicial` (`f086a39`, `a039071` - *Etapa 1*)
- `docs: agregar documentacion inicial y dependencias` (`cacab7d` - *Etapa 1*)
- `Merge PR #1: feature/estructura-inicial` (`fc31c01` - *Etapa 1*)
- `data: incorporar y documentar dataset de restaurantes` (`bedf7b2` - *Etapa 2*)
- `feat: agregar analisis exploratorio` (`3923752` - *Etapa 2*)
- `Merge PR #2: feature/dataset-eda` (`6a9dd96` - *Etapa 2*)
- `feat: configurar stopwords en español` (`c976215` - *Etapa 3*)
- `feat: comparar stemming y lematizacion` (`e91d0c4` - *Etapa 3*)
- `Merge PR #3: feature/nlp-stopwords-stemming-4-` (`1930ce2` - *Etapa 3*)
- `feat: entrenar y evaluar modelo naive bayes` (`d693a0b` - *Etapa 4*)
- `Merge PR #4: feature/modelo-naive-bayes` (`d378a43` - *Etapa 4*)
- `feat: implementar backend FastAPI y pruebas automatizadas` (`1317a50` - *Etapa 5*)
- `Merge PR #5: feature/backend-fastapi` (`dbe4c91` - *Etapa 5*)
- `Merge PR #6: integracion de robert-dev` (`2eb4550` - *Etapa 6*)
- `feat: integrar regresion logistica con frontend y API` (`92cbc93` - *Etapa 6*)
- `Merge PR #7: feature/integracion-frontend-backend` (`7901090` - *Etapa 6*)
- `docs: agregar comparacion de modelos Naive Bayes y Regresion Logistica` (`1bb7e3b` - *Etapa 7*)
- `Merge PR #8: docs/comparacion-modelos` (`c74d668` - *Etapa 7*)

### Contribuciones de Roberto (`@Roberto`)
- `chore: agregar enunciado de proyecto al .gitignore` (`867422c` - *Etapa 1*)
- `docs: documentar flujo de colaboracion y contribuciones` (`2ffff52` - *Etapa 1*)
- `feat: implementar normalizacion de reseñas` (`3cf1347` - *Etapa 2*)
- `feat: agregar tokenizacion de reseñas` (`8f2432f` - *Etapa 2*)
- `feat: implementar vectorizacion tfidf` (`474c311` - *Etapa 3*)
- `feat: comparar unigramas y bigramas` (`9d3072d` - *Etapa 3*)
- `refactor(nlp): estandarizar clases y funciones al espanol` (`8f1e8ac` - *Etapa 3*)
- `feat(nlp): integrar pipeline nlp unificado y desglose paso a paso` (`5fdbcf4` - *Etapa 3*)
- `feat(ml): implementar regresion logistica con gridsearchcv, metricas y serializacion` (`9167c14` - *Etapa 4*)
- `feat(frontend): crear interfaz web interactiva, visualizador nlp y dashboard` (`8b2565f` - *Etapa 5 y 6*)
- `style(frontend): refinar diseno visual con estetica moderna, animacion en loop continuo y eliminar distorsion 3d` (`d63b162`, `b5e1265` - *Etapa 5 y 6*)
- `docs: actualizar README con guia exhaustiva de instalacion, arquitectura y ejecucion` (`03eb6fa` - *Etapa 7*)
- `refactor: delimitar aportes estrictamente al rol asignado (nlp, ml regresion logistica y frontend)` (`1fb3be2` - *Etapa 7*)
- `docs: actualizar registro de contribuciones y preparar material de defensa para etapa 7` (`8ae36ad` - *Etapa 7*)
