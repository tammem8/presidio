"""Benchmark script to measure Presidio Analyzer performance across multiple dimensions.

Tests:
1. Prompt length impact (10, 100, 1000 tokens)
2. Number of recognizers impact
3. Entity filtering impact
4. spaCy model size impact (sm vs lg)
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import SpacyNlpEngine, NerModelConfiguration

from .scripts.analyzer import create_analyzer_engine

PROMPTS_FILE = Path(__file__).parent / "data" / "benchmark_prompts.yaml"


def load_prompts() -> Dict[str, str]:
    """Load benchmark prompts from YAML file."""
    with open(PROMPTS_FILE) as f:
        prompts = yaml.safe_load(f)
    return {
        "short": prompts["short"].strip(),
        "medium": prompts["medium"].strip(),
        "long": prompts["long"].strip(),
    }


def time_analysis(
    analyzer: AnalyzerEngine,
    text: str,
    iterations: int,
    language: str = "en",
    entities: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Run analyzer.analyze() multiple times and return timing stats."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        analyzer.analyze(text=text, language=language, entities=entities)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
    }


def print_table(title: str, rows: List[Dict], columns: List[str]) -> None:
    """Print a formatted results table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    # Header
    header = "  ".join(f"{col:<{widths[col]}}" for col in columns)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    # Rows
    for row in rows:
        line = "  ".join(f"{str(row.get(col, '')):<{widths[col]}}" for col in columns)
        print(f"  {line}")
    print()


def create_engine_with_recognizers(
    recognizer_names: List[str],
    analyzer: AnalyzerEngine,
) -> AnalyzerEngine:
    """Create an analyzer engine with only the specified recognizers."""
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()

    # Get all recognizers and filter
    all_recognizers = list(registry.recognizers)
    filtered = [r for r in all_recognizers if type(r).__name__ in recognizer_names]

    new_registry = RecognizerRegistry()
    for r in filtered:
        new_registry.add_recognizer(r)

    return AnalyzerEngine(
        nlp_engine=analyzer.nlp_engine,
        registry=new_registry,
    )


def create_engine_with_model(model_name: str) -> AnalyzerEngine:
    """Create an analyzer engine with a specific spaCy model."""
    ner_config = NerModelConfiguration()
    nlp_engine = SpacyNlpEngine(
        models=[{"lang_code": "en", "model_name": model_name}],
        ner_model_configuration=ner_config,
    )
    return AnalyzerEngine(nlp_engine=nlp_engine)


def benchmark_prompt_length(analyzer: AnalyzerEngine, prompts: Dict[str, str], iterations: int) -> None:
    """Test 1: Impact of prompt length."""
    rows = []
    for label, text in prompts.items():
        stats = time_analysis(analyzer, text, iterations)
        rows.append({
            "Prompt": label,
            "Chars": str(len(text)),
            "Mean (s)": f"{stats['mean']:.4f}",
            "Std (s)": f"{stats['std']:.4f}",
            "Min (s)": f"{stats['min']:.4f}",
            "Max (s)": f"{stats['max']:.4f}",
        })
    print_table(
        "Test 1: Prompt Length Impact",
        rows,
        ["Prompt", "Chars", "Mean (s)", "Std (s)", "Min (s)", "Max (s)"],
    )


def benchmark_recognizer_count(analyzer: AnalyzerEngine, prompts: Dict[str, str], iterations: int) -> None:
    """Test 2: Impact of number of recognizers."""
    configs = [
        ("All recognizers", None),
        ("Predefined only (5)", ["SpacyRecognizer", "EmailRecognizer", "PhoneRecognizer", "CreditCardRecognizer", "DateRecognizer"]),
        ("Minimal (2)", ["SpacyRecognizer", "EmailRecognizer"]),
    ]

    text = prompts["medium"]
    rows = []
    for label, recognizer_names in configs:
        if recognizer_names is None:
            engine = analyzer
        else:
            engine = create_engine_with_recognizers(recognizer_names, analyzer)
        stats = time_analysis(engine, text, iterations)
        n_recognizers = len(list(engine.registry.recognizers))
        rows.append({
            "Config": label,
            "# Recognizers": str(n_recognizers),
            "Mean (s)": f"{stats['mean']:.4f}",
            "Std (s)": f"{stats['std']:.4f}",
            "Min (s)": f"{stats['min']:.4f}",
            "Max (s)": f"{stats['max']:.4f}",
        })
    print_table(
        "Test 2: Number of Recognizers Impact (using ~100 token prompt)",
        rows,
        ["Config", "# Recognizers", "Mean (s)", "Std (s)", "Min (s)", "Max (s)"],
    )


def benchmark_entity_filtering(analyzer: AnalyzerEngine, prompts: Dict[str, str], iterations: int) -> None:
    """Test 3: Impact of entity filtering via the entities parameter."""
    text = prompts["medium"]
    configs = [
        ("All entities (default)", None),
        ("PERSON only", ["PERSON"]),
        ("PERSON + EMAIL", ["PERSON", "EMAIL_ADDRESS"]),
        ("PERSON + EMAIL + PHONE", ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]),
    ]

    rows = []
    for label, entities in configs:
        stats = time_analysis(analyzer, text, iterations, entities=entities)
        rows.append({
            "Entities": label,
            "Mean (s)": f"{stats['mean']:.4f}",
            "Std (s)": f"{stats['std']:.4f}",
            "Min (s)": f"{stats['min']:.4f}",
            "Max (s)": f"{stats['max']:.4f}",
        })
    print_table(
        "Test 3: Entity Filtering Impact (using ~100 token prompt)",
        rows,
        ["Entities", "Mean (s)", "Std (s)", "Min (s)", "Max (s)"],
    )


def benchmark_model_size(models: List[str], prompts: Dict[str, str], iterations: int) -> None:
    """Test 4: Impact of spaCy model size."""
    text = prompts["medium"]
    rows = []
    for model_name in models:
        try:
            engine = create_engine_with_model(model_name)
            # Warmup
            engine.analyze(text=text, language="en")
            stats = time_analysis(engine, text, iterations)
            rows.append({
                "Model": model_name,
                "Mean (s)": f"{stats['mean']:.4f}",
                "Std (s)": f"{stats['std']:.4f}",
                "Min (s)": f"{stats['min']:.4f}",
                "Max (s)": f"{stats['max']:.4f}",
            })
        except OSError:
            rows.append({
                "Model": model_name,
                "Mean (s)": "N/A (not installed)",
                "Std (s)": "",
                "Min (s)": "",
                "Max (s)": "",
            })
    print_table(
        "Test 4: spaCy Model Size Impact (using ~100 token prompt)",
        rows,
        ["Model", "Mean (s)", "Std (s)", "Min (s)", "Max (s)"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Presidio Analyzer performance across multiple dimensions."
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=5,
        help="Number of iterations per test (default: 5)",
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        default="en_core_web_sm",
        help="Comma-separated spaCy model names to compare (default: en_core_web_sm)",
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    print(f"Running benchmarks with {args.iterations} iterations each...\n")

    # Load prompts
    prompts = load_prompts()

    # Create default analyzer (with all config)
    print("Loading default analyzer engine...")
    analyzer = create_analyzer_engine()

    # Warmup
    analyzer.analyze(text="warmup text", language="en")

    # Run benchmarks
    benchmark_prompt_length(analyzer, prompts, args.iterations)
    benchmark_recognizer_count(analyzer, prompts, args.iterations)
    benchmark_entity_filtering(analyzer, prompts, args.iterations)
    benchmark_model_size(models, prompts, args.iterations)

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
