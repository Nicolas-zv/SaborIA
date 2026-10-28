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
- `chore: crear estructura inicial de SaborIA`
- `docs: agregar documentacion inicial y dependencias`

### Contribuciones de Roberto (`@Roberto`)
- `chore: agregar enunciado de proyecto al .gitignore` (`867422c`)
- `docs: documentar flujo de colaboracion y contribuciones` (`2ffff52` - *Etapa 1*)
- `feat: implementar normalizacion de reseñas` (`3cf1347` - *Etapa 2*)
- `feat: agregar tokenizacion de reseñas` (`8f2432f` - *Etapa 2*)
- `feat: implementar vectorizacion tfidf` (*Etapa 3*)
