# Presidio PII Detection Toolkit

This project evaluates PII (Personally Identifiable Information) detection using [Microsoft Presidio](https://microsoft.github.io/presidio/). It generates synthetic datasets, runs Presidio Analyzer against them, and produces precision/recall/F1 metrics.

## Table of Contents

- [Analyzer](./docs/analyzer.md) - How Presidio Analyzer works, recognition pipeline, entity detection flow
- [Configuration](./docs/configuration.md) - YAML config files for analyzer, NLP engine, and recognizers
- [Data Generation](./docs/data-generation.md) - Generating synthetic PII datasets with Faker
- [Evaluation](./docs/evaluation.md) - Running evaluations, understanding metrics and outputs
- [CLI Reference](./docs/cli-reference.md) - All available commands and arguments

## Quick Start

```bash
# Install dependencies
task install

# Generate a dataset
task detector:generate

# Evaluate
task detector:evaluate

# Quick single-text analysis
task detector:prompt_evaluate
```

## Supported Languages

| Language | Locale | spaCy Model | Status |
|----------|--------|-------------|--------|
| English | `en_US` | `en_core_web_lg` | Supported |
| German | `de_DE` | `de_core_news_lg` | Supported |
| French | `fr_FR` | `fr_core_news_lg` | Supported |

## Supported Entity Types

| Entity | Detection Method | Example |
|--------|-----------------|---------|
| `PERSON` | spaCy NER | John Doe |
| `LOCATION` | spaCy NER | 123 Main St, Berlin |
| `DATE_TIME` | spaCy NER + DateRecognizer | 1990-01-15 |
| `PHONE_NUMBER` | PhoneRecognizer + custom regex | +49 30 1234567 |
| `EMAIL_ADDRESS` | EmailRecognizer | user@example.com |
| `CREDIT_CARD` | CreditCardRecognizer + custom regex | 4111 1111 1111 1111 |
| `IBAN_CODE` | IbanRecognizer | DE89 3704 0044 0532 0130 00 |
| `BIOMEDICAL` | Custom regex (fingerprint hash) | SHA256:AA:BB:CC:... |
