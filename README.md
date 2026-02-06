# Presidio POC - Fake PII Data Generator

A tool for generating fake PII (Personally Identifiable Information) data using [Presidio Evaluator](https://github.com/microsoft/presidio-research).

## Features

- Generate fake data from customizable sentence templates
- Support for multiple locales/languages
- Custom providers for SSL keys and fingerprint hashes
- JSON output compatible with Presidio Evaluator

## Installation

```bash
uv sync
```

## Usage

### Command Line

```bash
python generate_fake_data.py --config <config_path> --output <output_folder> [options]
```

#### Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--config` | `-c` | Yes | Path to JSON config file containing sentence_templates and other settings |
| `--output` | `-o` | Yes | Output folder path where to write the generated JSON file |
| `--num-samples` | `-n` | No | Number of samples to generate (overrides config value) |
| `--language` | `-l` | No | Language/locale to use, e.g., `en_US`, `de_DE` (overrides config value) |

#### Examples

```bash
# Basic usage
python generate_fake_data.py --config resources/config/sample_config.json --output ./resources/data

# Generate 500 samples
python generate_fake_data.py -c resources/config/sample_config.json -o ./output -n 500

# Generate German data
python generate_fake_data.py -c resources/config/sample_config.json -o ./output -l de_DE
```

## Configuration

Create a JSON config file with the following structure:

```json
{
    "sentence_templates": [
        "My name is {{name}}",
        "Please send it to {{address}}",
        "My phone number is {{phone_number}}",
        "Use this card number {{credit_card}} to pay for it"
    ],
    "language": "en_US",
    "number_of_samples": 200,
    "lower_case_ratio": 0.05,
    "output_name": "fake_pii_data"
}
```

### Config Options

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `sentence_templates` | Yes | - | List of sentence templates with `{{entity}}` placeholders |
| `language` | No | `en_US` | Faker locale for data generation |
| `number_of_samples` | No | `100` | Number of samples to generate |
| `lower_case_ratio` | No | `0.05` | Ratio of lowercase text in output |
| `output_name` | No | `generated_data` | Prefix for output filename |

### Supported Entities

Standard Faker entities plus custom providers:

- **Standard**: `name`, `address`, `city`, `country`, `phone_number`, `credit_card`, `iban`, `email`, `date_of_birth`, etc.
- **Custom**: `ssl_key`, `fingerprint_hash`

## Output

The script generates a JSON file in the Presidio Evaluator `InputSample` format, which can be used for training and evaluating PII detection models.
