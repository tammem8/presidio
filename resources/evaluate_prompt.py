"""Script to analyze text input for PII entities using Presidio Analyzer."""

import argparse
import time

from .scripts.analyzer import create_analyzer_engine
from .scripts.models import PromptEvaluationInput


def analyze_text(prompt_input: PromptEvaluationInput) -> None:
    """Analyze text for PII entities and print results.

    Args:
        prompt_input: Input configuration containing text and language.
    """
    # Create analyzer engine using YAML config files
    analyzer = create_analyzer_engine()

    # Analyze the text with timing
    start_time = time.perf_counter()
    results = analyzer.analyze(
        text=prompt_input.text,
        language=prompt_input.language,
    )
    elapsed_time = time.perf_counter() - start_time

    # Print results
    print("\n" + "=" * 60)
    print("PII ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nInput Text: {prompt_input.text}")
    print(f"Language: {prompt_input.language}")
    print(f"Analysis Time: {elapsed_time:.4f} seconds")
    print(f"\nDetected Entities ({len(results)}):")
    print("-" * 40)

    if not results:
        print("No PII entities detected.")
    else:
        for result in results:
            entity_text = prompt_input.text[result.start : result.end]
            print(f"  • {result.entity_type}")
            print(f"    Text: '{entity_text}'")
            print(f"    Position: [{result.start}:{result.end}]")
            print(f"    Score: {result.score:.2f}")
            print()


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze text for PII entities using Presidio Analyzer."
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        required=True,
        help="The text to analyze for PII entities.",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="en",
        help="Language code for the text (default: en).",
    )
    args = parser.parse_args()

    # Create input model
    prompt_input = PromptEvaluationInput(
        text=args.text,
        language=args.language,
    )

    # Run analysis
    analyze_text(prompt_input)


if __name__ == "__main__":
    main()
