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
MAPA_SENTIMIENTO_EMOJIS: Dict[str, str] = {
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


class NormalizadorTexto(BaseEstimator, TransformerMixin):
    """
    Transformador de normalización de texto compatible con Scikit-learn Pipeline.

    Parámetros:
    -----------
    a_minusculas : bool, default=True
        Convierte todo el texto a minúsculas para unificar variantes ortográficas.
    eliminar_tildes : bool, default=True
        Elimina tildes y diacríticos (ej. 'pésimo' -> 'pesimo') para evitar
        dispersión del vocabulario debido a faltas ortográficas comunes en reseñas web.
    manejar_emojis : bool, default=True
        Traduce emojis y emoticonos a tokens semánticos descriptivos
        (ej. '😋' -> 'emoji_delicioso') para no perder la fuerte señal de polaridad.
    normalizar_puntuacion : bool, default=True
        Reduce signos de exclamación/interrogación repetidos (ej. '!!!!!' -> '!!')
        para conservar el énfasis emocional sin crear tokens infinitos.
    eliminar_urls_menciones : bool, default=True
        Remueve URLs, correos y menciones (@usuario) que representan ruido.
    eliminar_numeros : bool, default=True
        Elimina dígitos numéricos no informativos.
    """

    def __init__(
        self,
        a_minusculas: bool = True,
        eliminar_tildes: bool = True,
        manejar_emojis: bool = True,
        normalizar_puntuacion: bool = True,
        eliminar_urls_menciones: bool = True,
        eliminar_numeros: bool = True,
        # Parámetros en inglés para retrocompatibilidad
        to_lower: Optional[bool] = None,
        remove_accents: Optional[bool] = None,
        handle_emojis: Optional[bool] = None,
        normalize_punctuation: Optional[bool] = None,
        remove_urls_mentions: Optional[bool] = None,
        remove_numbers: Optional[bool] = None,
    ):
        self.a_minusculas = to_lower if to_lower is not None else a_minusculas
        self.eliminar_tildes = remove_accents if remove_accents is not None else eliminar_tildes
        self.manejar_emojis = handle_emojis if handle_emojis is not None else manejar_emojis
        self.normalizar_puntuacion = normalize_punctuation if normalize_punctuation is not None else normalizar_puntuacion
        self.eliminar_urls_menciones = remove_urls_mentions if remove_urls_mentions is not None else eliminar_urls_menciones
        self.eliminar_numeros = remove_numbers if remove_numbers is not None else eliminar_numeros

    def fit(self, X, y=None):
        """Método fit compatible con Scikit-Learn (sin estado aprendido)."""
        return self

    def transform(self, X, y=None) -> List[str]:
        """Aplica la normalización a una lista o Serie de textos."""
        return [self.normalizar(texto) for texto in X]

    def normalizar(self, texto: str) -> str:
        """
        Normaliza una única cadena de texto aplicando la secuencia ordenada de reglas.
        """
        if not isinstance(texto, str) or not texto.strip():
            return ""

        # 1. Tratamiento o traducción de emojis antes de que sean destruidos por filtros de caracteres
        if self.manejar_emojis:
            for emoji_char, token in MAPA_SENTIMIENTO_EMOJIS.items():
                texto = texto.replace(emoji_char, token)

        # 2. Eliminación de URLs y menciones (@usuario)
        if self.eliminar_urls_menciones:
            texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
            texto = re.sub(r"@\w+", " ", texto)
            texto = re.sub(r"\S+@\S+", " ", texto)

        # 3. Conversión a minúsculas
        if self.a_minusculas:
            texto = texto.lower()

        # 4. Normalización de puntuación repetida
        if self.normalizar_puntuacion:
            texto = re.sub(r"!{2,}", " !! ", texto)
            texto = re.sub(r"\?{2,}", " ?? ", texto)

        # 5. Tratamiento de tildes / diacríticos
        if self.eliminar_tildes:
            texto = unicodedata.normalize("NFKD", texto)
            texto = "".join(c for c in texto if not unicodedata.combining(c))

        # 6. Eliminación de números
        if self.eliminar_numeros:
            texto = re.sub(r"\d+", " ", texto)

        # 7. Limpieza de caracteres no alfanuméricos restantes
        texto = re.sub(r"[^\w\s!?]", " ", texto)

        # 8. Reducción de espacios en blanco múltiples
        texto = re.sub(r"\s+", " ", texto).strip()

        return texto

    # Alias en inglés para normalizar()
    normalize = normalizar

    def explicar_pasos(self, texto: str) -> Dict[str, str]:
        """
        Devuelve el paso a paso de la normalización con fines didácticos,
        defensa ante el docente o consumo desde el frontend.
        """
        pasos = {"original": texto}

        # Paso 1: Emojis
        curr = texto
        if self.manejar_emojis:
            for emoji_char, token in MAPA_SENTIMIENTO_EMOJIS.items():
                curr = curr.replace(emoji_char, token)
        pasos["emojis_mapeados"] = curr

        # Paso 2: Sin URLs / Menciones
        if self.eliminar_urls_menciones:
            curr = re.sub(r"https?://\S+|www\.\S+", " ", curr)
            curr = re.sub(r"@\w+", " ", curr)
            curr = re.sub(r"\S+@\S+", " ", curr)
        pasos["sin_urls_menciones"] = curr

        # Paso 3: Minúsculas
        if self.a_minusculas:
            curr = curr.lower()
        pasos["minusculas"] = curr

        # Paso 4: Puntuación de énfasis
        if self.normalizar_puntuacion:
            curr = re.sub(r"!{2,}", " !! ", curr)
            curr = re.sub(r"\?{2,}", " ?? ", curr)
        pasos["puntuacion_reducida"] = curr

        # Paso 5: Sin tildes
        if self.eliminar_tildes:
            curr = unicodedata.normalize("NFKD", curr)
            curr = "".join(c for c in curr if not unicodedata.combining(c))
        pasos["sin_tildes"] = curr

        # Paso 6: Sin números
        if self.eliminar_numeros:
            curr = re.sub(r"\d+", " ", curr)
        pasos["sin_numeros"] = curr

        # Paso 7: Caracteres especiales y espacios
        curr = re.sub(r"[^\w\s!?]", " ", curr)
        curr = re.sub(r"\s+", " ", curr).strip()
        pasos["resultado_final"] = curr

        return pasos

    # Alias en inglés para explicar_pasos()
    explain = explicar_pasos


def normalizar_resena(texto: str, **kwargs) -> str:
    """Función de utilidad directa para normalizar una reseña."""
    normalizador = NormalizadorTexto(**kwargs)
    return normalizador.normalizar(texto)


# Alias para retrocompatibilidad
TextNormalizer = NormalizadorTexto
normalize_review = normalizar_resena
EMOJI_SENTIMENT_MAP = MAPA_SENTIMIENTO_EMOJIS
