"""Utility functions for dataset loading, configuration, and output handling."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from presidio_evaluator import InputSample

from .models import AnalyzerConfig, EvaluationOutput


# =============================================================================
# Configuration Loading Utilities
# =============================================================================


def load_analyzer_config(config_path: Optional[str]) -> AnalyzerConfig:
    """Load analyzer configuration from file or use Pydantic defaults.

    Args:
        config_path: Optional path to analyzer configuration JSON file.
            If None, looks for analyzer_config.json in the config folder.

    Returns:
        AnalyzerConfig instance.
    """
    if config_path:
        print(f"Loading analyzer configuration from: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return AnalyzerConfig(**config_dict)

    # Look for default config file in config folder
    default_config_path = Path(__file__).parent.parent / "config" / "analyzer_config.json"
    if default_config_path.exists():
        print(f"Loading analyzer configuration from default: {default_config_path}")
        with open(default_config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return AnalyzerConfig(**config_dict)

    print("Using default analyzer configuration")
    return AnalyzerConfig()


# =============================================================================
# Dataset Loading and Processing Utilities
# =============================================================================


def load_dataset(input_path: str) -> List[InputSample]:
    """Load dataset from JSON file.

    Args:
        input_path: Path to the JSON dataset file.

    Returns:
        List of InputSample objects.
    """
    print(f"Loading dataset from: {input_path}")
    dataset = InputSample.read_dataset_json(input_path)
    print(f"Loaded {len(dataset)} samples")
    return dataset


def get_entity_counts(dataset: List[InputSample]) -> Counter:
    """Return a counter per entity type.

    Args:
        dataset: List of InputSample objects.

    Returns:
        Counter with entity type counts.
    """
    entity_counter = Counter()
    for sample in dataset:
        for tag in sample.tags:
            entity_counter[tag] += 1
    return entity_counter


def print_dataset_stats(dataset: List[InputSample]) -> None:
    """Print dataset statistics.

    Args:
        dataset: List of InputSample objects.
    """
    entity_counts = get_entity_counts(dataset)

    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(dataset)}")
    print("")
    print("Count per entity:")
    for entity, count in entity_counts.most_common():
        print(f"  {entity}: {count}")

    if dataset:
        token_lengths = [
            len(sample.tokens) for sample in dataset if sample.tokens is not None
        ]
        text_lengths = [len(sample.full_text) for sample in dataset]
        print("")
        print(f"Token count - Min: {min(token_lengths)}, Max: {max(token_lengths)}")
        print(f"Text length - Min: {min(text_lengths)}, Max: {max(text_lengths)}")
    print("=" * 60)


def check_entity_mappings(
    dataset: List[InputSample],
    entities_mapping: Dict[str, str],
) -> Set[str]:
    """Check if all dataset entities are mapped to recognizers.

    Args:
        dataset: List of InputSample objects.
        entities_mapping: Dictionary of entity type mappings.

    Returns:
        Set of unmapped entity types.
    """
    mapped_entities = set(entities_mapping.keys())
    dataset_entities = set()

    for sample in dataset:
        for span in sample.spans:
            dataset_entities.add(span.entity_type)

    unmapped = dataset_entities - mapped_entities
    if unmapped:
        print(f"WARNING: Unmapped entities found: {unmapped}")
        for entity in unmapped:
            print(f"  Entity '{entity}' is not mapped to a recognizer.")

    return unmapped


def filter_mapped_samples(
    dataset: List[InputSample],
    entities_mapping: Dict[str, str],
) -> Tuple[List[InputSample], List[InputSample]]:
    """Filter dataset to only include samples with mapped entities.

    Args:
        dataset: List of InputSample objects.
        entities_mapping: Dictionary of entity type mappings.

    Returns:
        Tuple of (mapped samples, unmapped samples).
    """
    mapped_values = set(entities_mapping.values())

    dataset_mapped = []
    dataset_not_mapped = []

    for sample in dataset:
        if all(span.entity_type in mapped_values for span in sample.spans):
            dataset_mapped.append(sample)
        else:
            dataset_not_mapped.append(sample)

    print(
        f"{len(dataset_not_mapped)} records with unmapped entities will be discarded. "
        f"{len(dataset_mapped)} records with mapped entities."
    )

    return dataset_mapped, dataset_not_mapped


# =============================================================================
# Output Saving Utilities
# =============================================================================


def ensure_output_dir(output_path: str) -> Path:
    """Ensure output directory exists.

    Args:
        output_path: Path to the output directory.

    Returns:
        Path object for the output directory.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_metrics_json(output: EvaluationOutput, output_dir: Path) -> str:
    """Save metrics to JSON file.

    Args:
        output: EvaluationOutput containing metrics.
        output_dir: Path to the output directory.

    Returns:
        Path to the saved metrics file.
    """
    metrics_path = output_dir / "metrics.json"

    metrics_dict = {
        "precision": output.metrics.pii_precision,
        "recall": output.metrics.pii_recall,
        "f1_score": output.metrics.pii_f1_score,
        "details": output.metrics.model_dump(),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False, default=str)

    print(f"Saved metrics to: {metrics_path}")
    return str(metrics_path)


def save_errors_csv(output: EvaluationOutput, output_dir: Path) -> Tuple[str, str]:
    """Save false positives and false negatives to separate CSV files.

    Args:
        output: EvaluationOutput containing error dataframes.
        output_dir: Path to the output directory.

    Returns:
        Tuple of (false positives path, false negatives path).
    """
    fps_path = output_dir / "false_positives.csv"
    fns_path = output_dir / "false_negatives.csv"

    output.fps.to_csv(fps_path, index=False, encoding="utf-8")
    output.fns.to_csv(fns_path, index=False, encoding="utf-8")

    print(f"Saved {len(output.fps)} false positives to: {fps_path}")
    print(f"Saved {len(output.fns)} false negatives to: {fns_path}")
    return str(fps_path), str(fns_path)


def plot_confusion_matrix(
    results: Any, title: str = "PII Detection Confusion Matrix"
) -> Any:
    """Plot a confusion matrix using Plotly.

    Args:
        results: The evaluation results object from SpanEvaluator.calculate_score().
        title: Title for the confusion matrix plot.

    Returns:
        Plotly Figure object.
    """
    import plotly.figure_factory as ff

    # Get confusion matrix from results
    confusion_matrix = results.to_confusion_matrix()

    # Get labels (entity types)
    labels, z = confusion_matrix

    # Create text annotations showing the counts
    z_text = [[str(int(val)) for val in row] for row in z]

    # Create the heatmap using figure_factory
    fig = ff.create_annotated_heatmap(
        z=z,
        x=labels,
        y=labels,
        annotation_text=z_text,
        colorscale="Blues",
        showscale=True,
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual", autorange="reversed"),
        width=800,
        height=700,
    )

    # Update xaxis to be at the bottom
    fig.update_xaxes(side="bottom")

    return fig


def print_results_summary(output: EvaluationOutput) -> None:
    """Print results summary to console.

    Args:
        output: EvaluationOutput containing metrics and errors.
    """
    metrics = output.metrics

    print("")
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"PII Precision: {metrics.pii_precision:.4f}")
    print(f"PII Recall:    {metrics.pii_recall:.4f}")
    print(f"PII F1 Score:  {metrics.pii_f1_score:.4f}")
    print("")
    print(f"Samples evaluated: {metrics.samples_evaluated}")
    print(f"Samples discarded: {metrics.samples_discarded}")
    print(f"Total entities:    {metrics.total_entities}")
    print("")

    print(f"False Positives: {len(output.fps)}")
    print(f"False Negatives: {len(output.fns)}")
    print("=" * 60)
