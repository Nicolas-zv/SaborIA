
import re
from nltk.corpus import stopwords

# Cargar palabras vacías del español
STOPWORDS_ES = set(stopwords.words("spanish"))

# Estas palabras son importantes para detectar sentimientos
NEGACIONES = {
    "no",
    "nunca",
    "jamás",
    "jamas",
    "ni",
    "sin",
    "tampoco"
}

# Eliminar de la lista de stop words las negaciones
STOPWORDS_PERSONALIZADAS = STOPWORDS_ES - NEGACIONES


def eliminar_stopwords(texto):
    """
    Elimina palabras vacías de una reseña,
    conservando las negaciones importantes.
    """

    if not isinstance(texto, str):
        raise TypeError("El texto debe ser una cadena.")

    # Convertir el texto a minúsculas
    texto = texto.lower()

    # Separar palabras conservando letras españolas
    palabras = re.findall(r"[a-záéíóúüñ]+", texto)

    # Eliminar palabras vacías
    palabras_filtradas = [
        palabra
        for palabra in palabras
        if palabra not in STOPWORDS_PERSONALIZADAS
    ]

    return " ".join(palabras_filtradas)


if __name__ == "__main__":
    ejemplos = [
        "La comida estaba muy buena y el servicio fue excelente.",
        "La comida no estaba buena y el servicio fue lento.",
        "Nunca volvería a este restaurante.",
        "El restaurante es bonito, pero la atención no fue buena."
    ]

    for texto in ejemplos:
        print("\nOriginal:", texto)
        print("Procesado:", eliminar_stopwords(texto))
