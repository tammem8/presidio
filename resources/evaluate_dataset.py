#!/usr/bin/env python3
"""
PII Detection Evaluation Script.

Evaluates a JSON input dataset against Presidio Analyzer and outputs:
- A JSON file with recall, precision, and F-score metrics
- A CSV file with false positives and false negatives

Usage:
    python evaluate.py --input data.json --output ./results
    python evaluate.py --input data.json --output ./results --config config.json
"""

import argparse
from typing import Optional, Tuple

from presidio_evaluator.evaluation import ModelError, SpanEvaluator

from .scripts.models import AnalyzerConfig, EvaluationOutput
from .scripts.analyzer import create_analyzer_engine, get_entities_mapping
from .scripts.utils import (
    load_dataset,
    get_entity_counts,
    print_dataset_stats,
    filter_mapped_samples,
    ensure_output_dir,
    save_metrics_json,
    save_errors_csv,
    plot_confusion_matrix,
    print_results_summary,
    load_analyzer_config,
)
from .scripts.evaluation import run_evaluation, create_evaluation_output


def evaluate(
    input_path: str,
    output_path: str,
    analyzer_config: Optional[AnalyzerConfig] = None,
    allow_missing_mappings: bool = True,
    score_threshold: float = 0.4,
) -> EvaluationOutput:
    """
    Main evaluation function.

    Args:
        input_path: Path to input JSON dataset
        output_path: Path to output directory
        analyzer_config: Optional analyzer configuration
        allow_missing_mappings: Whether to allow missing entity mappings
        score_threshold: Minimum score threshold for detections

    Returns:
        EvaluationOutput with metrics and errors
    """
    # Load analyzer configuration if not provided
    if analyzer_config is None:
        analyzer_config = load_analyzer_config(None)

    # Load dataset
    dataset = load_dataset(input_path)
    print_dataset_stats(dataset)

    # Create analyzer engine
    analyzer_engine = create_analyzer_engine(analyzer_config)

    # Get entity mappings
    entities_mapping = get_entities_mapping(analyzer_config)

    # Align entity types
    dataset = SpanEvaluator.align_entity_types(
        dataset,
        entities_mapping=entities_mapping,
        allow_missing_mappings=allow_missing_mappings,
    )

    # Print aligned entity counts
    aligned_counts = get_entity_counts(dataset)
    print("Count per entity after alignment:")
    for entity, count in aligned_counts.most_common():
        print(f"  {entity}: {count}")
    print(f"Total tags after alignment: {sum(aligned_counts.values())}")

    # Filter to only mapped samples
    dataset_mapped, dataset_not_mapped = filter_mapped_samples(
        dataset, entities_mapping
    )

    # Run evaluation
    results, evaluation_results = run_evaluation(
        dataset_mapped, analyzer_engine, score_threshold=score_threshold
    )

    # Extract errors
    fps = ModelError.get_fps_dataframe(results.model_errors)
    fns = ModelError.get_fns_dataframe(results.model_errors)

    # Create output
    output = create_evaluation_output(
        results=results,
        dataset=dataset_mapped,
        samples_discarded=len(dataset_not_mapped),
        fps=fps,
        fns=fns,
        analyzer_config=analyzer_config,
        allow_missing_mappings=allow_missing_mappings,
        score_threshold=score_threshold,
    )

    # Print summary
    print_results_summary(output)

    # Save outputs
    output_dir = ensure_output_dir(output_path)
    save_metrics_json(output, output_dir)
    save_errors_csv(output, output_dir)

    # Save confusion matrix as HTML (no kaleido dependency required)
    fig = plot_confusion_matrix(results)
    confusion_matrix_path = output_dir / "confusion_matrix.html"
    fig.write_html(str(confusion_matrix_path))
    print(f"Saved confusion matrix to: {confusion_matrix_path}")

    return output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate PII detection on a JSON dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate.py --input data.json --output ./results
    python evaluate.py --input data.json --output ./results --config config.json
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input JSON dataset file",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output directory for results",
    )
    parser.add_argument(
        "--analyzer-config",
        "-c",
        help="Path to analyzer configuration JSON file",
    )
    parser.add_argument(
        "--allow-missing-mappings",
        action="store_true",
        default=True,
        help="Allow missing entity mappings (default: True)",
    )
    parser.add_argument(
        "--no-allow-missing-mappings",
        action="store_false",
        dest="allow_missing_mappings",
        help="Disallow missing entity mappings",
    )
    parser.add_argument(
        "--score-threshold",
        "-t",
        type=float,
        default=0.4,
        help="Minimum score threshold for detections (default: 0.4)",
    )

    args = parser.parse_args()

    # Load analyzer config
    analyzer_config = load_analyzer_config(args.analyzer_config)

    # Run evaluation
    evaluate(
        input_path=args.input,
        output_path=args.output,
        analyzer_config=analyzer_config,
        allow_missing_mappings=args.allow_missing_mappings,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
