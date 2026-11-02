"""
Pruebas unitarias para el PipelineNLP unificado de SaborIA.
"""
import pytest
from src.nlp.pipeline import PipelineNLP, preprocesar_resena_completa


def test_pipeline_procesar_texto():
    pipeline = PipelineNLP(metodo_reduccion="lematizacion")
    texto = "¡¡¡La comida no estaba buena, pésimo servicio!!! 😡"
    procesado = pipeline.procesar(texto)
    assert isinstance(procesado, str)
    assert len(procesado) > 0
    # La negación "no" debe conservarse
    assert "no" in procesado.split()


def test_pipeline_desglose_paso_a_paso():
    pipeline = PipelineNLP(metodo_reduccion="lematizacion")
    texto = "Excelente atención pero las sillas no eran cómodas."
    resultado = pipeline.procesar_paso_a_paso(texto)

    assert "texto_original" in resultado
    assert "paso_1_normalizacion" in resultado
    assert "paso_2_tokenizacion" in resultado
    assert "paso_3_sin_stopwords" in resultado
    assert "paso_4_reduccion" in resultado
    assert "paso_5_texto_final" in resultado
    assert "estadisticas" in resultado

    assert isinstance(resultado["paso_2_tokenizacion"], list)
    assert isinstance(resultado["paso_3_sin_stopwords"], list)
    assert resultado["estadisticas"]["total_tokens_iniciales"] >= len(resultado["paso_3_sin_stopwords"])


def test_pipeline_alternar_stemming():
    pipeline = PipelineNLP(metodo_reduccion="stemming")
    texto = "Los camareros atendieron muy bien."
    procesado = pipeline.procesar(texto)
    assert isinstance(procesado, str)
    assert len(procesado) > 0


def test_preprocesar_resena_completa_helper():
    texto = "Me encantó el postre 😍"
    resultado = preprocesar_resena_completa(texto)
    assert isinstance(resultado, str)
    assert len(resultado) > 0
