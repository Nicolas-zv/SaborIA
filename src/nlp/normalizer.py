"""
Módulo de Normalización de Texto para Reseñas Gastronómicas (SaborIA).
Aplica técnicas de limpieza, estandarización de caracteres, manejo de emojis
y tratamiento de diacríticos/acentos con justificación lingüística.
"""
import re
import unicodedata
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin

# Diccionario de emojis y emoticonos frecuentes en reseñas gastronómicas con su carga de polaridad
EMOJI_SENTIMENT_MAP: Dict[str, str] = {
    # Emociones positivas / delicia
    "😋": " emoji_delicioso ",
    "🤤": " emoji_delicioso ",
    "😍": " emoji_encanto ",
    "🥰": " emoji_amor ",
    "❤️": " emoji_amor ",
    "👏": " emoji_aplauso ",
    "👌": " emoji_excelente ",
    "🔥": " emoji_fuego ",
    "✨": " emoji_brillante ",
    "👍": " emoji_positivo ",
    "😊": " emoji_feliz ",
    "😁": " emoji_feliz ",
    ":)": " emoji_feliz ",
    ":-)": " emoji_feliz ",
    ":D": " emoji_muyfeliz ",
    # Emociones negativas / disgusto
    "🤮": " emoji_vomito ",
    "🤢": " emoji_asco ",
    "😡": " emoji_enojo ",
    "🤬": " emoji_furia ",
    "👎": " emoji_negativo ",
    "😞": " emoji_triste ",
    "😢": " emoji_triste ",
    "😭": " emoji_llanto ",
    "💀": " emoji_pesimo ",
    ":(": " emoji_triste ",
    ":-(": " emoji_triste ",
}


class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    Transformador de normalización de texto compatible con Scikit-learn Pipeline.

    Parámetros:
    -----------
    to_lower : bool, default=True
        Convierte todo el texto a minúsculas para unificar variantes ortográficas.
    remove_accents : bool, default=True
        Elimina tildes y diacríticos (ej. 'pésimo' -> 'pesimo') para evitar
        dispersión del vocabulario debido a faltas ortográficas comunes en reseñas web.
    handle_emojis : bool, default=True
        Traduce emojis y emoticonos a tokens semánticos descriptivos
        (ej. '😋' -> 'emoji_delicioso') para no perder la fuerte señal de polaridad.
    normalize_punctuation : bool, default=True
        Reduce signos de exclamación/interrogación repetidos (ej. '!!!!!' -> '!!')
        para conservar el énfasis emocional sin crear tokens infinitos.
    remove_urls_mentions : bool, default=True
        Remueve URLs, correos y menciones (@usuario) que representan ruido.
    remove_numbers : bool, default=True
        Elimina dígitos numéricos no informativos.
    """

    def __init__(
        self,
        to_lower: bool = True,
        remove_accents: bool = True,
        handle_emojis: bool = True,
        normalize_punctuation: bool = True,
        remove_urls_mentions: bool = True,
        remove_numbers: bool = True,
    ):
        self.to_lower = to_lower
        self.remove_accents = remove_accents
        self.handle_emojis = handle_emojis
        self.normalize_punctuation = normalize_punctuation
        self.remove_urls_mentions = remove_urls_mentions
        self.remove_numbers = remove_numbers

    def fit(self, X, y=None):
        """Método fit compatible con Scikit-Learn (sin estado aprendido)."""
        return self

    def transform(self, X, y=None) -> List[str]:
        """Aplica la normalización a una lista o Serie de textos."""
        return [self.normalize(text) for text in X]

    def normalize(self, text: str) -> str:
        """
        Normaliza una única cadena de texto aplicando la secuencia ordenada de reglas.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Tratamiento o traducción de emojis antes de que sean destruidos por filtros de caracteres
        if self.handle_emojis:
            for emoji_char, token in EMOJI_SENTIMENT_MAP.items():
                text = text.replace(emoji_char, token)

        # 2. Eliminación de URLs y menciones (@usuario)
        if self.remove_urls_mentions:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)
            text = re.sub(r"@\w+", " ", text)
            text = re.sub(r"\S+@\S+", " ", text)  # correos

        # 3. Conversión a minúsculas
        if self.to_lower:
            text = text.lower()

        # 4. Normalización de puntuación repetida (conserva hasta dos repeticiones para capturar énfasis)
        if self.normalize_punctuation:
            text = re.sub(r"!{2,}", " !! ", text)
            text = re.sub(r"\?{2,}", " ?? ", text)

        # 5. Tratamiento de tildes / diacríticos
        if self.remove_accents:
            text = unicodedata.normalize("NFKD", text)
            text = "".join(c for c in text if not unicodedata.combining(c))

        # 6. Eliminación de números
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)

        # 7. Limpieza de caracteres no alfanuméricos restantes (permitiendo letras y signos de énfasis)
        text = re.sub(r"[^\w\s!?]", " ", text)

        # 8. Reducción de espacios en blanco múltiples
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def explain(self, text: str) -> Dict[str, str]:
        """
        Devuelve el paso a paso de la normalización con fines didácticos,
        defensa ante el docente o consumo desde el frontend.
        """
        steps = {"original": text}

        # Paso 1: Emojis
        curr = text
        if self.handle_emojis:
            for emoji_char, token in EMOJI_SENTIMENT_MAP.items():
                curr = curr.replace(emoji_char, token)
        steps["emojis_mapeados"] = curr

        # Paso 2: Sin URLs / Menciones
        if self.remove_urls_mentions:
            curr = re.sub(r"https?://\S+|www\.\S+", " ", curr)
            curr = re.sub(r"@\w+", " ", curr)
            curr = re.sub(r"\S+@\S+", " ", curr)
        steps["sin_urls_menciones"] = curr

        # Paso 3: Minúsculas
        if self.to_lower:
            curr = curr.lower()
        steps["minusculas"] = curr

        # Paso 4: Puntuación de énfasis
        if self.normalize_punctuation:
            curr = re.sub(r"!{2,}", " !! ", curr)
            curr = re.sub(r"\?{2,}", " ?? ", curr)
        steps["puntuacion_reducida"] = curr

        # Paso 5: Sin tildes
        if self.remove_accents:
            curr = unicodedata.normalize("NFKD", curr)
            curr = "".join(c for c in curr if not unicodedata.combining(c))
        steps["sin_tildes"] = curr

        # Paso 6: Sin números
        if self.remove_numbers:
            curr = re.sub(r"\d+", " ", curr)
        steps["sin_numeros"] = curr

        # Paso 7: Caracteres especiales y espacios
        curr = re.sub(r"[^\w\s!?]", " ", curr)
        curr = re.sub(r"\s+", " ", curr).strip()
        steps["resultado_final"] = curr

        return steps


def normalize_review(text: str, **kwargs) -> str:
    """Función de utilidad directa para normalizar una reseña."""
    normalizer = TextNormalizer(**kwargs)
    return normalizer.normalize(text)
