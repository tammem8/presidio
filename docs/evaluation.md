# Evaluation

## Overview

The evaluation pipeline runs Presidio Analyzer against a labeled dataset and measures detection accuracy using span-based matching.

```
Labeled Dataset ──> PresidioAnalyzerWrapper ──> SpanEvaluator ──> Metrics + Errors
     (JSON)              (analyze each           (compare         (precision,
                          sample)                 spans)           recall, F1)
```

## Span-Based Evaluation (IoU)

The `SpanEvaluator` compares predicted entity spans against ground truth using **Intersection over Union (IoU)** at the character level.

```
Ground truth:  "Contact John Smith at john@example.com"
                        |---------|    character positions 8-17
Prediction:    "Contact John Smith at john@example.com"
                        |-------|      character positions 8-15

IoU = intersection / union = 8 / 10 = 0.80
```

A prediction counts as a **true positive** only if:
- IoU >= threshold (default: 0.9)
- Entity type matches the annotation

Otherwise it's classified as a false positive, false negative, or wrong entity type.

## Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Precision** | TP / (TP + FP) | Of all detections, how many were correct |
| **Recall** | TP / (TP + FN) | Of all actual PII, how much was found |
| **F1 Score** | F-beta with beta=2 | Weighted harmonic mean (recall-heavy) |

Metrics are computed both **globally** (all PII vs non-PII) and **per entity type**.

## Evaluation Outputs

The evaluation produces four files in the output directory:

| File | Content |
|------|---------|
| `metrics.json` | Precision, recall, F1, per-entity breakdowns |
| `false_positives.csv` | Text flagged as PII that wasn't (FP errors) |
| `false_negatives.csv` | PII that was missed (FN errors) |
| `confusion_matrix.html` | Interactive Plotly heatmap of predicted vs actual entities |

### metrics.json structure

```json
{
  "precision": 0.76,
  "recall": 0.64,
  "f1_score": 0.66,
  "details": {
    "pii_precision": 0.76,
    "pii_recall": 0.64,
    "pii_f1_score": 0.66,
    "entity_precision_dict": { "PERSON": 0.85, "LOCATION": 0.70, ... },
    "entity_recall_dict": { "PERSON": 0.90, "LOCATION": 0.55, ... },
    "total_samples": 200,
    "samples_evaluated": 200,
    "samples_discarded": 0
  }
}
```

## Entity Alignment

Before evaluation, entity types from the dataset are aligned to Presidio's entity names using mappings from:

1. `PresidioAnalyzerWrapper.presidio_entities_map` (default mappings)
2. `nlp-config.yaml` (spaCy NER label → Presidio entity)
3. `recognizers-config.yaml` (custom recognizer entities)
4. `AnalyzerEngine.get_supported_entities()` (identity mappings)

Samples with unmapped entity types are discarded and reported.

## Language Handling

The `--language` flag controls which language the analyzer uses at inference time:
- Use `en`, `de`
- This is different from Faker locales used during generation (`en_US`, `de_DE`)

The evaluation wraps the analyzer in a `PresidioAnalyzerWrapper` with the specified language, bypassing the default behavior which always uses the first language in the config.
