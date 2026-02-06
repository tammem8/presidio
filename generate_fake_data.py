#!/usr/bin/env python3
"""
Script to generate fake PII data using Presidio Evaluator.

Usage:
    python generate_fake_data.py --config config.json --output ./output
"""

import argparse
import json
import os
import random
import re
import string
from pathlib import Path

from faker.providers import BaseProvider
from presidio_evaluator import InputSample
from presidio_evaluator.data_generator import PresidioSentenceFaker


# Custom Providers
class SSLProvider(BaseProvider):
    """Provider for generating fake SSL/TLS private keys."""
    
    def ssl_key(self):
        """Generate a fake SSL/TLS private key in PEM format."""
        key_length = 1704  # Approximate length for a 2048-bit RSA key
        chars = string.ascii_letters + string.digits + '+/'
        key_content = ''.join(random.choice(chars) for _ in range(key_length))
        lines = [key_content[i:i+64] for i in range(0, len(key_content), 64)]
        key_body = '\n'.join(lines)
        return f"-----BEGIN RSA PRIVATE KEY-----\n{key_body}\n-----END RSA PRIVATE KEY-----"


class FingerprintProvider(BaseProvider):
    """Provider for generating fake fingerprint hashes."""
    
    def fingerprint_hash(self):
        """Generate a fake fingerprint hash (e.g., SSH/TLS fingerprint format)."""
        hex_pairs = [format(random.randint(0, 255), '02X') for _ in range(16)]
        fingerprint = ':'.join(hex_pairs)
        return f"SHA256:{fingerprint}"


# Registry of custom providers
CUSTOM_PROVIDERS = {
    "ssl_key": SSLProvider,
    "fingerprint_hash": FingerprintProvider,
}


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['sentence_templates']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    return config


def extract_entities(sentence_templates: list[str]) -> list[str]:
    """Extract entity names from sentence templates."""
    all_templates = ' '.join(sentence_templates)
    entities = list(set(re.findall(r'\{\{(\w+)\}\}', all_templates)))
    return entities


def create_sentence_faker(
    language: str,
    sentence_templates: list[str],
    lower_case_ratio: float = 0.05,
) -> PresidioSentenceFaker:
    """Create and configure a PresidioSentenceFaker instance."""
    entities = extract_entities(sentence_templates)
    
    sentence_faker = PresidioSentenceFaker(
        language,
        lower_case_ratio=lower_case_ratio,
        sentence_templates=sentence_templates,
        entity_type_mapping={e: e.upper() for e in entities},
    )
    
    # Add custom providers based on entities used in templates
    for entity in entities:
        if entity in CUSTOM_PROVIDERS:
            sentence_faker.add_provider(CUSTOM_PROVIDERS[entity])
    
    return sentence_faker


def generate_fake_data(
    config_path: str,
    output_folder: str,
    num_samples: int | None = None,
    language: str | None = None,
) -> str:
    """
    Generate fake data based on configuration.
    
    Args:
        config_path: Path to JSON config file
        output_folder: Folder to save output JSON file
        num_samples: Number of samples to generate (overrides config)
        language: Language/locale to use (overrides config)
    
    Returns:
        Path to the generated output file
    """
    # Load config
    config = load_config(config_path)
    
    # Get parameters (CLI args override config values)
    sentence_templates = config['sentence_templates']
    final_num_samples = num_samples or config.get('number_of_samples', 100)
    final_language = language or config.get('language', 'en_US')
    lower_case_ratio = config.get('lower_case_ratio', 0.05)
    output_name = config.get('output_name', 'generated_data')
    
    # Create faker
    sentence_faker = create_sentence_faker(
        language=final_language,
        sentence_templates=sentence_templates,
        lower_case_ratio=lower_case_ratio,
    )
    
    # Generate fake records
    print(f"Generating {final_num_samples} fake samples...")
    fake_records = sentence_faker.generate_new_fake_sentences(final_num_samples)
    
    # Prepare output
    os.makedirs(output_folder, exist_ok=True)
    output_filename = f"{output_name}_size_{final_num_samples}_{final_language}.json"
    output_path = os.path.join(output_folder, output_filename)
    
    # Save to JSON
    InputSample.to_json(dataset=fake_records, output_file=output_path)
    print(f"Generated data saved to: {output_path}")
    
    # Print summary
    from collections import Counter
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
        "--config", "-c",
        type=str,
        required=True,
        help="Path to JSON config file containing sentence_templates and other settings"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output folder path where to write the generated JSON file"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=None,
        help="Number of samples to generate (overrides config value)"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Language/locale to use, e.g., 'en_US', 'de_DE' (overrides config value)"
    )
    
    args = parser.parse_args()
    
    generate_fake_data(
        config_path=args.config,
        output_folder=args.output,
        num_samples=args.num_samples,
        language=args.language,
    )


if __name__ == "__main__":
    main()
