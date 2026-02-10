#!/usr/bin/env python3
"""
PII Detection Evaluation Script.

Evaluates a JSON input dataset against Presidio Analyzer and outputs:
- A JSON file with recall, precision, and F-score metrics
- A CSV file with false positives and false negatives

Usage:
    python evaluate.py --input data.json --output ./results
"""

import argparse
from pathlib import Path

import pandas as pd
from presidio_evaluator.evaluation import ModelError, SpanEvaluator
from .scripts.models import EvaluationOutput
from .scripts.analyzer import create_analyzer_engine
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
)
from .scripts.evaluation import build_entities_mapping, run_evaluation, create_evaluation_output


def evaluate(
    input_path: str,
    output_path: str,
) -> EvaluationOutput:
    """
    Main evaluation function.

    Args:
        input_path: Path to input JSON dataset
        output_path: Path to output directory for results

    Returns:
        EvaluationOutput with metrics and errors
    """

    # Load dataset
    dataset = load_dataset(input_path)
    print_dataset_stats(dataset)

    # Create analyzer engine using YAML config files
    analyzer_engine = create_analyzer_engine()

    # Build entity mappings from all sources
    entities_mapping = build_entities_mapping(analyzer_engine)
        
    # Align entity types
    dataset = SpanEvaluator.align_entity_types(
        dataset,
        entities_mapping=entities_mapping,
        allow_missing_mappings=True,
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
    results, _ = run_evaluation(
        dataset_mapped, analyzer_engine
    )

    # Extract errors (these methods return None when no errors found)
    fps_result = ModelError.get_fps_dataframe(results.model_errors)
    fns_result = ModelError.get_fns_dataframe(results.model_errors)
    fps = fps_result if fps_result is not None else pd.DataFrame()
    fns = fns_result if fns_result is not None else pd.DataFrame()

    # Create output
    output = create_evaluation_output(
        results=results,
        dataset=dataset_mapped,
        samples_discarded=len(dataset_not_mapped),
        fps=fps,
        fns=fns
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
    python evaluate.py --input data.json --output ./results-
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

    args = parser.parse_args()

    # Run evaluation
    evaluate(
        input_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
