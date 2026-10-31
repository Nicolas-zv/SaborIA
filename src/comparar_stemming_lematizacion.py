
from pathlib import Path
import pandas as pd

import spacy
from nltk.stem.snowball import SnowballStemmer

# Configurar stemming en español
stemmer = SnowballStemmer("spanish")

# Cargar modelo de español de spaCy
nlp = spacy.load("es_core_news_sm")


def aplicar_stemming(texto):
    """Reduce las palabras a raíces aproximadas."""
    documento = nlp(texto.lower())

    return [
        stemmer.stem(token.text)
        for token in documento
        if token.is_alpha
    ]


def aplicar_lematizacion(texto):
    """Obtiene la forma base de las palabras."""
    documento = nlp(texto.lower())

    return [
        token.lemma_
        for token in documento
        if token.is_alpha
    ]



if __name__ == "__main__":
    # PRIMERA PARTE: Ejemplos manuales
    ejemplos = [
        "Los clientes estaban comiendo deliciosas comidas.",
        "Los camareros atendieron rápidamente las mesas.",
        "Nunca volvería porque los platos estaban fríos.",
        "El restaurante ofrece comidas excelentes."
    ]

    print("COMPARACIÓN CON EJEMPLOS MANUALES")

    for texto in ejemplos:
        print("\n" + "=" * 60)
        print("ORIGINAL:", texto)
        print("STEMMING:", " ".join(aplicar_stemming(texto)))
        print("LEMATIZACIÓN:", " ".join(aplicar_lematizacion(texto)))

    # SEGUNDA PARTE: Reseñas reales del dataset
    print("\n" + "=" * 60)
    print("COMPARACIÓN CON RESEÑAS DEL DATASET")

    ruta = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "resenas_restaurantes.csv"
    )

    df = pd.read_csv(ruta)

    for _, fila in df.head(5).iterrows():
        texto = fila["texto"]

        print("\n" + "-" * 60)
        print("SENTIMIENTO:", fila["sentimiento"])
        print("ORIGINAL:", texto)
        print("STEMMING:", " ".join(aplicar_stemming(texto)))
        print("LEMATIZACIÓN:", " ".join(aplicar_lematizacion(texto)))

