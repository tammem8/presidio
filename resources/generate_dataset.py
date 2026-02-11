#!/usr/bin/env python3
"""
Script to generate fake PII data using Presidio Evaluator.

Usage:
    python generate_fake_data.py --config config.yaml --output ./output
"""

import argparse
import os
import random

import yaml
from faker.providers import BaseProvider
from presidio_evaluator import InputSample
from presidio_evaluator.data_generator import PresidioSentenceFaker
from collections import Counter

# Custom Providers
class FingerprintProvider(BaseProvider):
    """Provider for generating fake fingerprint hashes."""

    def fingerprint_hash(self):
        """Generate a fake fingerprint hash (e.g., SSH/TLS fingerprint format)."""
        hex_pairs = [format(random.randint(0, 255), "02X") for _ in range(16)]
        fingerprint = ":".join(hex_pairs)
        return f"SHA256:{fingerprint}"


# Registry of custom providers
CUSTOM_PROVIDERS = {
    "fingerprint_hash": FingerprintProvider
}


def load_config(config_path: str, language: str) -> dict:
    """Load language-specific configuration from YAML file.

    Args:
        config_path: Path to YAML config file.
        language: Language key to look up (e.g. 'en_US').

    Returns:
        Dict with 'sentence_templates' and 'sentence_mapping' for the language.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "languages" not in config:
        raise ValueError("Config file must contain a 'languages' key")

    if language not in config["languages"]:
        available = ", ".join(config["languages"].keys())
        raise ValueError(
            f"Language '{language}' not found in config. Available: {available}"
        )

    lang_config = config["languages"][language]

    required_fields = ["sentence_templates", "sentence_mapping"]
    for field in required_fields:
        if field not in lang_config:
            raise ValueError(
                f"Missing required field '{field}' for language '{language}'"
            )

    return lang_config


def create_sentence_faker(
    language: str,
    sentence_templates: list[str],
    sentence_mapping: dict[str, str],
) -> PresidioSentenceFaker:
    """Create and configure a PresidioSentenceFaker instance."""
    sentence_faker = PresidioSentenceFaker(
        language,
        sentence_templates=sentence_templates,
        entity_type_mapping=sentence_mapping,
        lower_case_ratio=0.0,
    )

    # Add custom providers based on entities used in templates
    for entity in sentence_mapping.keys():
        if entity in CUSTOM_PROVIDERS:
            sentence_faker.add_provider(CUSTOM_PROVIDERS[entity])  # pyright: ignore[reportArgumentType]

    return sentence_faker


def generate_fake_data(
    config_path: str,
    output_folder: str,
    prefix: str,
    num_samples: int,
    language: str,
) -> str:
    """
    Generate fake data based on configuration.

    Args:
        config_path: Path to JSON config file
        output_folder: Folder to save output JSON file
        prefix: Prefix for the output filename
        num_samples: Number of samples to generate
        language: Language/locale to use

    Returns:
        Path to the generated output file
    """
    # Load language-specific config
    config = load_config(config_path, language)

    sentence_templates = config["sentence_templates"]
    sentence_mapping = config["sentence_mapping"]

    # Create faker
    sentence_faker = create_sentence_faker(
        language=language,
        sentence_templates=sentence_templates,
        sentence_mapping=sentence_mapping,
    )

    # Generate fake records
    print(f"Generating {num_samples} fake samples...")
    fake_records = sentence_faker.generate_new_fake_sentences(num_samples)

    # Prepare output
    os.makedirs(output_folder, exist_ok=True)
    output_filename = f"{prefix}_{num_samples}_{language}.json"
    output_path = os.path.join(output_folder, output_filename)

    # Save to JSON
    InputSample.to_json(dataset=fake_records, output_file=output_path)
    print(f"Generated data saved to: {output_path}")

    # Print summary
    count_per_entity = Counter()
    for record in fake_records:
        count_per_entity.update(Counter([span.entity_type for span in record.spans]))
    print("\nEntity counts:")
    for entity, count in sorted(count_per_entity.items()):
        print(f"  {entity}: {count}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate fake PII data using Presidio Evaluator"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML config file containing sentence_templates and other settings",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output folder path where to write the generated JSON file",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        required=True,
        help="Prefix for the output filename (output: prefix_size_language.json)",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        required=True,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        required=True,
        help="Language/locale to use, e.g., 'en_US', 'de_DE'",
    )

    args = parser.parse_args()

    generate_fake_data(
        config_path=args.config,
        output_folder=args.output,
        prefix=args.prefix,
        num_samples=args.num_samples,
        language=args.language,
    )


if __name__ == "__main__":
    main()
