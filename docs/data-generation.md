# Data Generation

Synthetic PII datasets are generated using [Presidio Evaluator](https://github.com/microsoft/presidio-research)'s `PresidioSentenceFaker`, which combines sentence templates with Faker-generated data.

## How It Works

```
Template:  "Contact {{person}} at {{email}}"
                 ↓ Faker
Output:    "Contact John Smith at john.smith@example.com"
Labels:            ^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^
                   PERSON         EMAIL_ADDRESS
```

1. Templates are loaded from `generate-dataset-config.yaml` for the specified language
2. `PresidioSentenceFaker` fills `{{placeholders}}` with locale-appropriate fake data
3. Each generated sample is an `InputSample` with text, tokens, spans (entity positions), and tags (per-token labels)
4. Output is saved as JSON (Presidio Evaluator's dataset format)

## Custom Faker Providers

A custom provider extend Faker for entities it doesn't natively support:

| Provider | Faker Method | Entity | Example Output |
|----------|-------------|--------|----------------|
| `FingerprintProvider` | `fingerprint_hash()` | Certificate fingerprint | `SHA256:AA:BB:CC:DD:...` |

To add a new custom provider:
1. Create a class extending `BaseProvider` in `generate_dataset.py`
2. Register it in the `CUSTOM_PROVIDERS` dict
3. Use the method name as a placeholder in templates and add it to `sentence_mapping`

## Output Format

Generated files are named `{prefix}_{num_samples}_{language}.json`:

```
fake_pii_data_200_en_US.json
fake_pii_data_200_de_DE.json
```

Each JSON file contains an array of `InputSample` objects with:
- `full_text` - the generated sentence
- `spans` - list of entity spans with type, start/end positions
- `tags` - per-token IO tags (e.g., `PERSON`, `O`, `EMAIL_ADDRESS`)
- `tokens` - tokenized text
