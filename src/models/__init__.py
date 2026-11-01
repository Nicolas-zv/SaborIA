"""
Módulo de Modelos de Machine Learning — SaborIA.
"""
from .logistic_regression import ClasificadorRegresionLogistica
from .train import entrenar_modelo_completo

__all__ = [
    "ClasificadorRegresionLogistica",
    "entrenar_modelo_completo",
]
