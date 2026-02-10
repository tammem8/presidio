# Presidio POC - Fake PII Data Generator

A tool for generating fake PII (Personally Identifiable Information) data using [Presidio Evaluator](https://github.com/microsoft/presidio-research).

## Features

- Generate fake data from customizable sentence templates and custom providers
- Support for multiple locales/languages
- Analyzer and custom recognizers
- Dataset evaluation with Presidio's framework, metrics and insights
- Prompt evaluation 

## Installation

```bash
task install
```
Or
```bash
uv sync
```

## Usage

### Task Commands

| Command | Description |
|---------|-------------|
| `task lint` | Run linter with auto-fix |
| `task format` | Format code |
| `task test` | Run tests |

### Generate Fake PII Data

Generate fake PII data using customizable sentence templates:

```bash
task detector:generate
```

#### Data generation configuration

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

The script generates a JSON file in the Presidio Evaluator `InputSample` format, which can be used for training and evaluating PII detection models.

### Analyzer Configuration

Create a JSON config file with the following structure:

| Field | Required | Description |
|-------|----------|-------------|
| `nlp_engine_name` | Yes | NLP engine to use (e.g., `spacy`) |
| `model_name` | Yes | Model name (e.g., `en_core_web_sm`, `en_core_web_lg`) |
| `language` | Yes | Language code (e.g., `en`) |
| `recognizers_to_keep` | No | List of built-in recognizers to enable (e.g., `SpacyRecognizer`, `EmailRecognizer`) |
| `custom_recognizers` | No | List of custom pattern-based recognizers |
| `entity_mappings` | No | Mappings from source entity types to target types |
| `allow_missing_mappings` | No | If `true`, entities without explicit mappings keep their original type. If `false`, unmapped entities raise an error. Default: `true` |
| `score_threshold` | No | Minimum confidence score for detections (0.0 - 1.0) |

### Custom Recognizers

Define custom pattern-based recognizers for entities not covered by built-in recognizers:

```json
{
    "entity_type": "SSL",
    "patterns": [
        {
            "name": "ssl_key_pattern",
            "regex": "-----BEGIN\\s+(RSA\\s+)?PRIVATE\\s+KEY-----[\\s\\S]*?-----END\\s+(RSA\\s+)?PRIVATE\\s+KEY-----",
            "score": 0.6
        }
    ]
}
```

### Evaluate Dataset

Evaluate Presidio's PII detection accuracy against a JSON dataset containing labeled PII samples. This command runs the analyzer on the dataset, compares predictions against ground truth, and outputs:
- Precision, recall, and F-score metrics (`metrics.json`)
- False positives and false negatives (`false_positives.csv`, `false_negatives.csv`)
- Confusion matrix visualization (`confusion_matrix.html`)

```bash
task detector:evaluate
```

### Evaluate Prompt

Analyze a single text input for PII entities in real-time. This is useful for quick testing and debugging of the analyzer configuration. The command displays detected entities with their type, position, confidence score, and analysis time.

```bash
task detector:prompt_evaluate
```