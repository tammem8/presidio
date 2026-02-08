"""Core evaluation logic."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import SpanEvaluator

from .utils import get_entity_counts
from .models import AnalyzerConfig, EvaluationMetrics, EvaluationOutput


def run_evaluation(
    dataset: List[InputSample],
    analyzer_engine: AnalyzerEngine,
    score_threshold: float = 0.4,
) -> Tuple[Any, Any]:
    """Run the evaluation and return results.

    Args:
        dataset: List of InputSample objects to evaluate.
        analyzer_engine: Configured Presidio AnalyzerEngine.
        score_threshold: Minimum score threshold for detections.

    Returns:
        Tuple of (aggregated results, individual evaluation results).
    """
    print("Running evaluation...")
    evaluator = SpanEvaluator(model=analyzer_engine, skip_words=[])
    evaluation_results = evaluator.evaluate_all(
        dataset, score_threshold=score_threshold
    )
    results = evaluator.calculate_score(evaluation_results)

    return results, evaluation_results


def create_evaluation_output(
    results: Any,
    dataset: List[InputSample],
    samples_discarded: int,
    fps: pd.DataFrame,
    fns: pd.DataFrame,
    analyzer_config: Optional[AnalyzerConfig] = None,
    allow_missing_mappings: bool = True,
    score_threshold: float = 0.4,
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

    # Combine configs for output
    config_dict: Optional[Dict[str, Any]] = None
    if analyzer_config:
        config_dict = {
            "analyzer": analyzer_config.model_dump(),
            "evaluation": {
                "allow_missing_mappings": allow_missing_mappings,
                "score_threshold": score_threshold,
            },
        }

    metrics = EvaluationMetrics(
        pii_precision=results.pii_precision,
        pii_recall=results.pii_recall,
        pii_f1_score=results.pii_f,
        entity_precision_dict=results.entity_precision_dict,
        entity_recall_dict=results.entity_recall_dict,
        total_samples=len(dataset) + samples_discarded,
        total_entities=sum(entity_counts.values()),
        samples_evaluated=len(dataset),
        samples_discarded=samples_discarded,
        config=config_dict,
    )

    return EvaluationOutput(metrics=metrics, fps=fps, fns=fns)
