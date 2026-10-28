"""
Módulo de Tokenización Especializada para Reseñas Gastronómicas (SaborIA).
Diseñado para manejar casos difíciles del dominio:
- Contracciones ('al', 'del')
- Jerga gastronómica y términos compuestos ('tex-mex', 'all-you-can-eat', 'fast-food', 't-bone')
- Puntuación de énfasis pegada a palabras ('delicioso!!')
- Mantenimiento de tokens especiales de sentimiento ('emoji_delicioso', 'emoji_vomito')
"""
import re
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin

# Términos compuestos del dominio gastronómico que no deben fragmentarse
COMPUESTOS_GASTRONOMICOS = {
    "tex-mex", "fast-food", "all-you-can-eat", "t-bone", "fondue",
    "take-away", "drive-thru", "fine-dining", "coca-cola", "gluten-free",
    "buffet-libre", "a-la-carta", "post-cena"
}

# Expansión de contracciones gramaticales frecuentes en español
MAPA_CONTRACCIONES: Dict[str, List[str]] = {
    "al": ["a", "el"],
    "del": ["de", "el"],
}


class TokenizadorResenas(BaseEstimator, TransformerMixin):
    """
    Tokenizador especializado para reseñas de restaurantes.
    Compatible con Scikit-Learn y como callable para TfidfVectorizer.

    Parámetros:
    -----------
    expandir_contracciones : bool, default=True
        Expande contracciones como 'al' -> ['a', 'el'] y 'del' -> ['de', 'el'].
    preservar_compuestos : bool, default=True
        Preserva términos compuestos como 'tex-mex' o 'gluten-free' como un solo token.
    preservar_enfasis : bool, default=True
        Separa signos de énfasis ('!!', '??') como tokens independientes de polaridad.
    longitud_minima_token : int, default=1
        Longitud mínima permitida para tokens alfabéticos (evita caracteres vacíos).
    """

    def __init__(
        self,
        expandir_contracciones: bool = True,
        preservar_compuestos: bool = True,
        preservar_enfasis: bool = True,
        longitud_minima_token: int = 1,
        # Parámetros retrocompatibles en inglés
        expand_contractions: Optional[bool] = None,
        preserve_compounds: Optional[bool] = None,
        preserve_emphasis: Optional[bool] = None,
        min_token_length: Optional[int] = None,
    ):
        self.expandir_contracciones = expand_contractions if expand_contractions is not None else expandir_contracciones
        self.preservar_compuestos = preserve_compounds if preserve_compounds is not None else preservar_compuestos
        self.preservar_enfasis = preserve_emphasis if preserve_emphasis is not None else preservar_enfasis
        self.longitud_minima_token = min_token_length if min_token_length is not None else longitud_minima_token

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> List[List[str]]:
        return [self.tokenizar(texto) for texto in X]

    def tokenizar(self, texto: str) -> List[str]:
        """
        Tokeniza un texto normalizado preservando la semántica gastronómica y de polaridad.
        """
        if not isinstance(texto, str) or not texto.strip():
            return []

        tokens: List[str] = []

        # Expresión regular que captura:
        # 1. Tokens de emojis estructurados (emoji_\w+)
        # 2. Términos con guión (ej. tex-mex, gluten-free)
        # 3. Signos de énfasis dobles (!! o ??)
        # 4. Palabras alfanuméricas estándar
        patron = r"(emoji_\w+|[a-zA-ZáéíóúñÁÉÍÓÚÑ]+-[a-zA-ZáéíóúñÁÉÍÓÚÑ]+|!!|\?\?|[a-zA-ZáéíóúñÁÉÍÓÚÑ\d]+)"

        tokens_brutos = re.findall(patron, texto)

        for tok in tokens_brutos:
            tok_min = tok.lower()

            # Caso 1: Signos de énfasis
            if tok in {"!!", "??"}:
                if self.preservar_enfasis:
                    tokens.append(tok)
                continue

            # Caso 2: Tokens de emoji
            if tok_min.startswith("emoji_"):
                tokens.append(tok_min)
                continue

            # Caso 3: Términos compuestos gastronómicos
            if "-" in tok_min and self.preservar_compuestos:
                tokens.append(tok_min)
                continue

            # Caso 4: Expansión de contracciones
            if self.expandir_contracciones and tok_min in MAPA_CONTRACCIONES:
                tokens.extend(MAPA_CONTRACCIONES[tok_min])
                continue

            # Caso estándar
            if len(tok_min) >= self.longitud_minima_token:
                tokens.append(tok_min)

        return tokens

    # Alias en inglés para tokenizar()
    tokenize = tokenizar

    def explicar_casos_dificiles(self, texto: str) -> Dict[str, object]:
        """
        Analiza e identifica casos difíciles encontrados en el texto:
        - Contracciones
        - Términos compuestos
        - Énfasis y emojis
        Útil para la justificación de NLP ante el docente y la vista de frontend.
        """
        tokens = self.tokenizar(texto)
        casos_detectados = {
            "tokens_totales": len(tokens),
            "tokens": tokens,
            "contracciones_detectadas": [],
            "terminos_compuestos": [],
            "emojis_o_enfasis": [],
        }

        palabras = texto.lower().split()
        for p in palabras:
            p_limpia = re.sub(r"[^\w-]", "", p)
            if p_limpia in MAPA_CONTRACCIONES:
                casos_detectados["contracciones_detectadas"].append(
                    f"'{p_limpia}' -> {MAPA_CONTRACCIONES[p_limpia]}"
                )
            if "-" in p_limpia:
                casos_detectados["terminos_compuestos"].append(p_limpia)
            if "emoji_" in p or "!!" in p or "??" in p:
                casos_detectados["emojis_o_enfasis"].append(p)

        return casos_detectados

    # Alias en inglés para explicar_casos_dificiles()
    explain_difficult_cases = explicar_casos_dificiles


def tokenizar_resena(texto: str, **kwargs) -> List[str]:
    """Función de utilidad directa para tokenizar una reseña."""
    tokenizador = TokenizadorResenas(**kwargs)
    return tokenizador.tokenizar(texto)


# Alias para retrocompatibilidad
ReviewTokenizer = TokenizadorResenas
tokenize_review = tokenizar_resena
GASTRONOMIC_COMPOUNDS = COMPUESTOS_GASTRONOMICOS
CONTRACTIONS_MAP = MAPA_CONTRACCIONES
