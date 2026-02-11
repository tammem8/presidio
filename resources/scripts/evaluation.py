"""Core evaluation logic."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from presidio_analyzer import AnalyzerEngine
from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import SpanEvaluator
from presidio_evaluator.models import PresidioAnalyzerWrapper

from .utils import get_entity_counts
from .models import EvaluationMetrics, EvaluationOutput

CONFIG_DIR = Path(__file__).parent.parent / "config"


def build_entities_mapping(analyzer_engine: AnalyzerEngine) -> Dict[str, str]:
    """Build a complete entity type mapping from all sources.

    Combines mappings from:
    - PresidioAnalyzerWrapper default mappings
    - NLP config (spaCy NER label -> Presidio entity)
    - Analyzer engine supported entities (predefined + custom recognizers)
    - Recognizers config (custom recognizer entities)

    Args:
        analyzer_engine: Configured AnalyzerEngine to query for supported entities.

    Returns:
        Dict mapping entity labels to Presidio entity types.
    """
    entities_mapping = dict(PresidioAnalyzerWrapper.presidio_entities_map)

    # Add mappings from NLP config (spaCy NER label -> Presidio entity)
    nlp_config_path = CONFIG_DIR / "nlp-config.yaml"
    with open(nlp_config_path) as f:
        nlp_config = yaml.safe_load(f)
    for entity, presidio_entity in nlp_config["ner_model_configuration"]["model_to_presidio_entity_mapping"].items():
        entities_mapping[entity] = presidio_entity

    # Add identity mappings for all recognizer entities (predefined + custom)
    for supported_entity in analyzer_engine.get_supported_entities():
        entities_mapping.setdefault(supported_entity, supported_entity)

    # Add custom recognizer entity mappings from recognizers config
    recognizers_config_path = CONFIG_DIR / "recognizers-config.yaml"
    with open(recognizers_config_path) as f:
        recognizers_config = yaml.safe_load(f)
    for recognizer in recognizers_config.get("recognizers", []):
        if "supported_entity" in recognizer:
            entities_mapping.setdefault(recognizer["supported_entity"], recognizer["supported_entity"])

    return entities_mapping


def run_evaluation(
    dataset: List[InputSample],
    analyzer_engine: AnalyzerEngine,
    language: str = "en",
) -> Tuple[Any, Any]:
    """Run the evaluation and return results.

    Args:
        dataset: List of InputSample objects to evaluate.
        analyzer_engine: Configured Presidio AnalyzerEngine.
        language: Language code for analysis (e.g. 'en', 'de').

    Returns:
        Tuple of (aggregated results, individual evaluation results).
    """
    print(f"Running evaluation (language={language})...")
    model = PresidioAnalyzerWrapper(
        analyzer_engine=analyzer_engine,
        score_threshold=analyzer_engine.default_score_threshold,
        language=language,
    )
    evaluator = SpanEvaluator(
        model=model,
        skip_words=[],
    )
    evaluation_results = evaluator.evaluate_all(dataset)
    results = evaluator.calculate_score(evaluation_results)

    return results, evaluation_results


def create_evaluation_output(
    results: Any,
    dataset: List[InputSample],
    samples_discarded: int,
    fps: pd.DataFrame,
    fns: pd.DataFrame,
) -> EvaluationOutput:
    """Create the evaluation output with metrics and error dataframes.

    Args:
        results: Aggregated evaluation results.
        dataset: List of evaluated InputSample objects.
        samples_discarded: Number of samples that were discarded.
        fps: DataFrame of false positives.
        fns: DataFrame of false negatives.
        analyzer_config: Optional analyzer configuration.
        allow_missing_mappings: Whether missing entity mappings were allowed.
        score_threshold: Score threshold used for detections.

    Returns:
        EvaluationOutput containing metrics and error dataframes.
    """
    entity_counts = get_entity_counts(dataset)

    metrics = EvaluationMetrics(
        pii_precision=results.pii_precision,
        pii_recall=results.pii_recall,
        pii_f1_score=results.pii_f,
        entity_precision_dict=results.entity_precision_dict,
        entity_recall_dict=results.entity_recall_dict,
        total_samples=len(dataset) + samples_discarded,
        total_entities=sum(entity_counts.values()),
        samples_evaluated=len(dataset),
        samples_discarded=samples_discarded
    )

    return EvaluationOutput(metrics=metrics, fps=fps, fns=fns)
