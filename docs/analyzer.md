# Analyzer

## How Presidio Analyzer Works

Presidio Analyzer detects PII entities in text using a pipeline of **recognizers**. Each recognizer specializes in one or more entity types.

```
                    ┌─────────────────────────────────────┐
  Input Text ──>    │         AnalyzerEngine              │
                    │                                     │
                    │  1. NLP Engine (spaCy)              │
                    │     tokenization, NER labels        │
                    │              │                      │
                    │              ▼                      │
                    │  2. Recognizers (parallel)          │
                    │     ┌────────────────────────── ┐   │
                    │     │ SpacyRecognizer   (NER)   │   │
                    │     │ EmailRecognizer   (regex) │   │
                    │     │ PhoneRecognizer   (regex) │   │
                    │     │ CreditCardRecognizer (✓)  │   │
                    │     │ Custom patterns   (regex) │   │
                    │     └────────────────────────── ┘   │
                    │              │                      │
                    │              ▼                      │
                    │  3. Context enhancement             │
                    │     boost scores near context words │
                    │              │                      │
                    │              ▼                      │
                    │  4. Dedup overlapping spans         │
                    │              │                      │
                    │              ▼                      │
                    │  5. Score threshold filter          │
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                       List[RecognizerResult]
```

### Detection Pipeline

1. **NLP Engine** processes the text through spaCy, producing tokens, lemmas, and NER labels
2. **Recognizers** run in parallel, each returning candidate spans with confidence scores:
   - **NLP-based** (`SpacyRecognizer`) - uses spaCy NER labels (PERSON, LOC, GPE, DATE, etc.)
   - **Pattern-based** - regex matching (phone numbers, credit cards, IBANs, etc.)
   - **Predefined** - built-in recognizers with validation logic (e.g., credit card checksum)
3. **Context enhancement** - scores are boosted when context words appear near a match
4. **Deduplication** - overlapping detections are resolved, keeping the highest score
5. **Score threshold filtering** - detections below `default_score_threshold` are discarded

#### Example

Input: `"Use card 4111111111111111 to pay, contact John Smith at john@acme.com"`

| Step | Recognizer | Span | Entity | Raw Score |
|------|-----------|------|--------|-----------|
| 2 | `CreditCardRecognizer` | `4111111111111111` | CREDIT_CARD | 1.0 (checksum valid) |
| 2 | `SpacyRecognizer` | `John Smith` | PERSON | 0.85 (NER default score) |
| 2 | `EmailRecognizer` | `john@acme.com` | EMAIL_ADDRESS | 1.0 (pattern match) |
| 2 | `SpacyRecognizer` | `John` | PERSON | 0.85 |

| Step | Action | Result |
|------|--------|--------|
| 4 | Dedup `John Smith` vs `John` | Keep `John Smith` (wider span, same score) |
| 5 | Filter by threshold (0.4) | All 3 detections pass |

#### Deduplication step
Dedup runs after all recognizers have returned their results, on the combined list.
- Note that a single recognizer can produce multiple overlapping spans.
- A span is removed only if:
  1. It's fully contained inside another span, AND
  2. They have the same entity type

These cases do survive dedup:
- Different entity types on the same span — e.g., SpacyRecognizer detects "John" as PERSON and CustomPhoneRecognizer matches overlapping digits as PHONE_NUMBER. Both are kept.
- Partially overlapping spans of the same type — e.g., regex matches "123 456" and "456 789" as PHONE_NUMBER. Neither fully contains the other, so both survive.

**Final output:** CREDIT_CARD (1.0), PERSON (0.85), EMAIL_ADDRESS (1.0)

If the threshold were 0.9, `PERSON` (0.85) would be filtered out.

#### Overriding the Score Threshold

The default threshold is set in `analyzer-config.yaml`:

```yaml
default_score_threshold: 0.4
```

It can be overridden per request:

```python
# Use the config default (0.4)
results = analyzer.analyze(text=text, language="en")

# Override for this request only
results = analyzer.analyze(text=text, language="en", score_threshold=0.7)
```

Lower thresholds increase recall (more detections, more false positives). Higher thresholds increase precision (fewer detections, more false negatives).

### Recognizer Types

| Type | How it works | Examples |
|------|-------------|----------|
| **Predefined** | Built into Presidio, enabled via config | `EmailRecognizer`, `CreditCardRecognizer`, `PhoneRecognizer`, `IbanRecognizer`, `DateRecognizer` |
| **NLP-based** | Uses spaCy NER model labels | `SpacyRecognizer` (detects PERSON, LOCATION, DATE_TIME) |
| **Custom pattern** | Regex patterns defined in YAML | `FingerprintRecognizer`, `CustomPhoneRecognizer` |

### Context-Aware Scoring

Recognizers can define **context words** that boost confidence when found near a match. For example, the `CreditCard19Recognizer` has context words `["credit", "card"]` — if these appear near a 19-digit number, the score increases from 0.6 (base) to ~0.95.

### Entity Mapping (spaCy to Presidio)

spaCy NER labels don't always match Presidio entity names. The NLP config maps them:

```
spaCy label    →    Presidio entity
───────────         ───────────────
PERSON         →    PERSON
LOC            →    LOCATION
GPE            →    LOCATION
DATE           →    DATE_TIME
TIME           →    DATE_TIME
```

### Production Considerations

The `AnalyzerEngine` is **stateless and thread-safe**. In production:
- Create it **once** at application startup
- Reuse it for all requests via `analyzer.analyze(text, language)`
- The heavy work (loading spaCy models, compiling regex, initializing recognizers) happens only at construction time
- Per-request cost is just the NLP pipeline + recognizer execution

### One Model Per Language

The NLP engine supports **one spaCy model per language**. Models are keyed by `lang_code` in the config. If you need to compare models (e.g., `de_core_news_sm` vs `de_core_news_lg`), you must swap the config and recreate the engine.
