# ai_draft_bot

Baselines and utilities for training a Limited drafter model using 17L exports. The
package provides ingestion helpers, feature extraction, a logistic regression baseline,
and CLIs for training and simulation.

## Getting started

Install dependencies with an isolated virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Training a model

Prepare two files from https://www.17lands.com/:

- A JSONL draft log export containing pick sequences
- A CSV card metadata export for the same set

Then run the training command:

```bash
python scripts/train.py run --drafts-path path/to/drafts.jsonl \
    --metadata-path path/to/cards.csv \
    --output-path artifacts/model.joblib
```

The script logs basic progress, splits the dataset into train/validation partitions, and
saves a `joblib` artifact with the classifier and label encoder.

## Simulating picks

After training, use the CLI to score a pack:

```bash
python -m ai_draft_bot.cli simulate \
    --model-path artifacts/model.joblib \
    --metadata-path path/to/cards.csv \
    --pack "Card A" --pack "Card B" --pack "Card C"
```

The CLI prints the most likely pick per the model along with probabilities for each
candidate card.

## Evaluation notes

- The baseline model is intentionally simple: it uses card-level metadata plus pack
  averages to fit a multiclass logistic regression classifier.
- Feature extraction lives in `src/ai_draft_bot/features/draft_context.py`; tweak this
  module to experiment with richer signals such as seat position or wheel picks.
- The training split and solver parameters are configurable via CLI flags in
  `scripts/train.py`.

## Development tooling

Ruff and mypy configurations live in `pyproject.toml`. Run the linters and tests with:

```bash
ruff check .
mypy src
pytest
```
