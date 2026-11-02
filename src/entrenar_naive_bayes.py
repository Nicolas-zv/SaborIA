
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Ruta del dataset
RUTA_DATASET = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "resenas_restaurantes.csv"
)


def main():
    print("=" * 60)
    print("SABORIA - ENTRENAMIENTO NAIVE BAYES")
    print("=" * 60)

    # 1. Cargar dataset
    df = pd.read_csv(RUTA_DATASET)

    print("\nTotal de reseñas:", len(df))
    print("\nDistribución de sentimientos:")
    print(df["sentimiento"].value_counts())

    # 2. Separar textos y etiquetas
    X = df["texto"].astype(str)
    y = df["sentimiento"]

    # 3. División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nReseñas para entrenamiento:", len(X_train))
    print("Reseñas para prueba:", len(X_test))

    # 4. Pipeline TF-IDF + Naive Bayes
    modelo = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 1),
            ),
        ),
        ("naive_bayes", MultinomialNB(alpha=0.1, fit_prior=False)),
    ])


    # 5. Buscar mejores hiperparámetros mediante validación cruzada
    print("\nBuscando mejores hiperparámetros...")

    parametros = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "naive_bayes__alpha": [0.01, 0.1, 0.5, 1.0],
        "naive_bayes__fit_prior": [True, False],
    }

    busqueda = GridSearchCV(
        estimator=modelo,
        param_grid=parametros,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
    )

    busqueda.fit(X_train, y_train)

    print("\nMejores parámetros:")
    print(busqueda.best_params_)

    print(
        "\nMejor F1 macro en validación cruzada:",
        round(busqueda.best_score_, 4),
    )

    # Modelo seleccionado mediante validación cruzada
    modelo = busqueda.best_estimator_

    # 6. Realizar predicciones
    predicciones = modelo.predict(X_test)

    # 7. Evaluar resultados
    accuracy = accuracy_score(y_test, predicciones)

    print("\n" + "=" * 60)
    print("RESULTADOS DEL MODELO")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_test,
            predicciones,
            zero_division=0,
        )
    )

    etiquetas = ["negativo", "neutral", "positivo"]

    print("\nMatriz de confusión:")
    print(
        confusion_matrix(
            y_test,
            predicciones,
            labels=etiquetas,
        )
    )

    print("\nOrden de las etiquetas:", etiquetas)


    # 8. Análisis de errores
    resultados = pd.DataFrame({
        "texto": X_test.values,
        "sentimiento_real": y_test.values,
        "sentimiento_predicho": predicciones,
    })

    errores = resultados[
        resultados["sentimiento_real"]
        != resultados["sentimiento_predicho"]
    ].copy()

    print("\n" + "=" * 60)
    print("ANÁLISIS DE ERRORES")
    print("=" * 60)

    print("\nTotal de predicciones incorrectas:", len(errores))

    print("\nPrimeros 10 errores:")
    for _, fila in errores.head(10).iterrows():
        print("\n" + "-" * 60)
        print("RESEÑA:", fila["texto"])
        print("REAL:", fila["sentimiento_real"])
        print("PREDICHO:", fila["sentimiento_predicho"])

    # Guardar errores para análisis posterior
    ruta_errores = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "errores_naive_bayes.csv"
    )

    errores.to_csv(ruta_errores, index=False, encoding="utf-8-sig")

    print("\nErrores guardados en:", ruta_errores)


    # 9. Guardar el modelo entrenado
    carpeta_modelos = (
        Path(__file__).resolve().parent.parent
        / "models"
    )

    carpeta_modelos.mkdir(parents=True, exist_ok=True)

    ruta_modelo = carpeta_modelos / "naive_bayes.joblib"

    joblib.dump(modelo, ruta_modelo)

    print("\n" + "=" * 60)
    print("MODELO GUARDADO CORRECTAMENTE")
    print("=" * 60)
    print("Ubicación:", ruta_modelo)

    # 10. Comprobar que el modelo se puede cargar
    modelo_cargado = joblib.load(ruta_modelo)

    ejemplo = ["La comida estaba deliciosa y la atención fue excelente."]
    prediccion = modelo_cargado.predict(ejemplo)

    print("\nPRUEBA DEL MODELO GUARDADO")
    print("Reseña:", ejemplo[0])
    print("Sentimiento predicho:", prediccion[0])


if __name__ == "__main__":
    main()
