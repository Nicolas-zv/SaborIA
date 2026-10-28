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
GASTRONOMIC_COMPOUNDS = {
    "tex-mex", "fast-food", "all-you-can-eat", "t-bone", "fondue",
    "take-away", "drive-thru", "fine-dining", "coca-cola", "gluten-free",
    "buffet-libre", "a-la-carta", "post-cena"
}

# Expansión de contracciones gramaticales frecuentes en español
CONTRACTIONS_MAP: Dict[str, List[str]] = {
    "al": ["a", "el"],
    "del": ["de", "el"],
}


class ReviewTokenizer(BaseEstimator, TransformerMixin):
    """
    Tokenizador especializado para reseñas de restaurantes.
    Compatible con Scikit-Learn y como callable para TfidfVectorizer.

    Parámetros:
    -----------
    expand_contractions : bool, default=True
        Expande contracciones como 'al' -> ['a', 'el'] y 'del' -> ['de', 'el'].
    preserve_compounds : bool, default=True
        Preserva términos compuestos como 'tex-mex' o 'gluten-free' como un solo token.
    preserve_emphasis : bool, default=True
        Separa signos de énfasis ('!!', '??') como tokens independientes de polaridad.
    min_token_length : int, default=1
        Longitud mínima permitida para tokens alfabéticos (evita caracteres vacíos).
    """

    def __init__(
        self,
        expand_contractions: bool = True,
        preserve_compounds: bool = True,
        preserve_emphasis: bool = True,
        min_token_length: int = 1,
    ):
        self.expand_contractions = expand_contractions
        self.preserve_compounds = preserve_compounds
        self.preserve_emphasis = preserve_emphasis
        self.min_token_length = min_token_length

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> List[List[str]]:
        return [self.tokenize(text) for text in X]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza un texto normalizado preservando la semántica gastronómica y de polaridad.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        tokens: List[str] = []

        # Expresión regular que captura:
        # 1. Tokens de emojis estructurados (emoji_\w+)
        # 2. Términos con guión (ej. tex-mex, gluten-free)
        # 3. Signos de énfasis dobles (!! o ??)
        # 4. Palabras alfanuméricas estándar
        pattern = r"(emoji_\w+|[a-zA-ZáéíóúñÁÉÍÓÚÑ]+-[a-zA-ZáéíóúñÁÉÍÓÚÑ]+|!!|\?\?|[a-zA-ZáéíóúñÁÉÍÓÚÑ\d]+)"

        raw_tokens = re.findall(pattern, text)

        for tok in raw_tokens:
            tok_lower = tok.lower()

            # Caso 1: Signos de énfasis
            if tok in {"!!", "??"}:
                if self.preserve_emphasis:
                    tokens.append(tok)
                continue

            # Caso 2: Tokens de emoji
            if tok_lower.startswith("emoji_"):
                tokens.append(tok_lower)
                continue

            # Caso 3: Términos compuestos gastronómicos
            if "-" in tok_lower and self.preserve_compounds:
                tokens.append(tok_lower)
                continue

            # Caso 4: Expansión de contracciones
            if self.expand_contractions and tok_lower in CONTRACTIONS_MAP:
                tokens.extend(CONTRACTIONS_MAP[tok_lower])
                continue

            # Caso estándar
            if len(tok_lower) >= self.min_token_length:
                tokens.append(tok_lower)

        return tokens

    def explain_difficult_cases(self, text: str) -> Dict[str, object]:
        """
        Analiza e identifica casos difíciles encontrados en el texto:
        - Contracciones
        - Términos compuestos
        - Énfasis y emojis
        Útil para la justificación de NLP ante el docente y la vista de frontend.
        """
        tokens = self.tokenize(text)
        detected_cases = {
            "tokens_totales": len(tokens),
            "tokens": tokens,
            "contracciones_detectadas": [],
            "terminos_compuestos": [],
            "emojis_o_enfasis": [],
        }

        words = text.lower().split()
        for w in words:
            clean_w = re.sub(r"[^\w-]", "", w)
            if clean_w in CONTRACTIONS_MAP:
                detected_cases["contracciones_detectadas"].append(
                    f"'{clean_w}' -> {CONTRACTIONS_MAP[clean_w]}"
                )
            if "-" in clean_w:
                detected_cases["terminos_compuestos"].append(clean_w)
            if "emoji_" in w or "!!" in w or "??" in w:
                detected_cases["emojis_o_enfasis"].append(w)

        return detected_cases


def tokenize_review(text: str, **kwargs) -> List[str]:
    """Función de utilidad directa para tokenizar una reseña."""
    tokenizer = ReviewTokenizer(**kwargs)
    return tokenizer.tokenize(text)
