# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PII (Personally Identifiable Information) detection evaluation toolkit built on top of Microsoft's [Presidio](https://github.com/microsoft/presidio) and [Presidio Evaluator](https://github.com/microsoft/presidio-research). It generates fake PII datasets, runs Presidio Analyzer against them, and produces precision/recall/F1 metrics with error analysis.

## Commands

Uses [Task](https://taskfile.dev/) as a task runner with `uv` for Python package management.

```bash
task install          # Install dependencies (uv sync)
task lint             # Lint with ruff --fix
task format           # Format with ruff
task test             # Run tests (uv run pytest)
task detector:generate        # Generate fake PII dataset
task detector:evaluate        # Evaluate analyzer on dataset
task detector:prompt_evaluate # Analyze a single text prompt
```

Run a single test: `uv run pytest tests/test_file.py::test_name`

## Architecture

The project has three CLI entry points in `resources/`:

- **`generate_fake_data.py`** — Generates fake PII data using `PresidioSentenceFaker`. Takes a JSON config with sentence templates (e.g. `"My name is {{name}}"`) and a `sentence_mapping` dict. Outputs Presidio `InputSample` JSON. Custom Faker providers (SSL keys, fingerprints) are registered in `CUSTOM_PROVIDERS`.

- **`evaluate_dataset.py`** — Runs `SpanEvaluator` against a labeled dataset. Handles entity type alignment via mappings, filters unmapped samples, and outputs metrics JSON, FP/FN CSVs, and a Plotly confusion matrix HTML.

- **`evaluate_prompt.py`** — Quick single-text analysis. Creates an analyzer engine and prints detected entities with scores and timing.

Shared logic lives in `resources/scripts/`:
- **`analyzer.py`** — Creates `AnalyzerEngine` via `AnalyzerEngineProvider` using three YAML config files from `resources/config/` (analyzer, NLP engine, recognizer registry).
- **`models.py`** — Pydantic models: `AnalyzerConfig`, `EvaluationMetrics`, `EvaluationOutput`, `PromptEvaluationInput`, and recognizer/pattern configs.
- **`evaluation.py`** — Core evaluation logic wrapping `SpanEvaluator`.
- **`utils.py`** — Dataset loading, entity counting, output saving (metrics JSON, error CSVs, confusion matrix).

## Configuration

Analyzer config uses YAML files in `resources/config/`:
- `analyzer-config.yaml` — Analyzer engine settings
- `nlp-config.yaml` — NLP engine/model config (spaCy)
- `recognizers-config.yaml` — Built-in and custom recognizer registry

The JSON config (`analyzer_config.json`) defines entity mappings and recognizer selection for evaluation. The data generation config (`sample_config.json`) defines sentence templates and entity-to-faker mappings.

## Key Dependencies

- Python 3.13, managed via `uv`
- `presidio-evaluator` — evaluation framework (includes `presidio-analyzer`)
- `en_core_web_sm` — spaCy English model (installed from GitHub releases URL)
- `plotly` — confusion matrix visualization
- `ruff` — linting and formatting