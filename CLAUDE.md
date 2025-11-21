# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a baseline toolkit for training Limited draft models using 17Lands (17L) data exports. The package ingests JSONL draft logs and CSV card metadata, extracts features, trains a multiclass logistic regression classifier, and provides CLIs for training and simulation.

## Environment Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## Development Commands

**Linting and Type Checking:**
```bash
ruff check .              # Enforce line length 100, E/F/I/B rules
mypy src                  # Strict type checking on source only
```

**Testing:**
```bash
pytest                    # Run all tests
pytest -q                 # Quick iteration mode
```

**Training a Model:**
```bash
python scripts/train.py run \
    --drafts-path path/to/drafts.jsonl \
    --metadata-path path/to/cards.csv \
    --output-path artifacts/model.joblib \
    --test-size 0.2 \
    --max-iter 500 \
    --c 1.0 \
    --random-state 13
```

**Simulating Picks:**
```bash
python -m ai_draft_bot.cli simulate \
    --model-path artifacts/model.joblib \
    --metadata-path path/to/cards.csv \
    --pack "Card A" --pack "Card B" --pack "Card C"
```

## Architecture

### Data Pipeline Flow

1. **Ingestion** (`src/ai_draft_bot/data/ingest_17l.py`)
   - `load_jsonl()`: Streams JSONL entries to avoid loading large files into memory
   - `parse_draft_logs()`: Converts raw 17L JSONL exports into `PickRecord` objects
   - `parse_card_metadata()`: Parses CSV metadata into `CardMetadata` objects keyed by card name
   - `group_picks_by_event()`: Organizes picks by event ID for sequential analysis

2. **Feature Extraction** (`src/ai_draft_bot/features/draft_context.py`)
   - `build_pick_features()`: Generates feature vectors combining:
     - Card-level attributes (mana value, rarity, color one-hot encoding)
     - Pack-level aggregates (mean mana value, rarity, color distribution)
   - Features are concatenated: `[chosen_card_vector, pack_mean_vector]`
   - Returns `PickFeatures` dataclass with feature array and label (chosen card name)

3. **Model Training** (`src/ai_draft_bot/models/drafter.py`)
   - `train_model()`: Trains scikit-learn `LogisticRegression` with train/val split
   - Uses `LabelEncoder` to map card names to class indices
   - Returns `DraftModel` wrapper with `predict()` and `predict_proba()` methods
   - Serializes artifacts using joblib (model + label encoder)

4. **Entry Points**
   - `scripts/train.py`: Typer CLI for training workflow
   - `src/ai_draft_bot/cli.py`: Typer CLI for simulation and evaluation

### Key Design Decisions

- **Streaming ingestion**: 17L exports can be large; `load_jsonl()` yields records to keep memory usage low
- **Simple feature space**: Current features are intentionally basic (card metadata + pack averages) to establish a baseline; extend `draft_context.py` to add signals like seat position or wheel picks
- **Multiclass classification**: Each pick is a multiclass problem with card names as labels
- **Joblib serialization**: Training artifacts are saved as a dictionary containing both the scikit-learn model and label encoder

## Code Organization

```
src/ai_draft_bot/
├── data/
│   └── ingest_17l.py       # JSONL/CSV parsing, PickRecord/CardMetadata dataclasses
├── features/
│   └── draft_context.py    # Feature extraction, color encoding, pack aggregation
├── models/
│   └── drafter.py          # DraftModel wrapper, training logic, TrainConfig
└── cli.py                  # Simulation CLI

scripts/
└── train.py                # Training CLI entry point

tests/
└── (add test_*.py here)    # Mirror package structure; use small fixtures
```

## Coding Conventions

- **Python 3.10+** with strict mypy type hints
- **Naming**: modules/files `snake_case`, classes `PascalCase`, functions/vars `snake_case`, CLI options `kebab-case`
- **Fixtures**: Store under `tests/data/` to avoid polluting source directories
- **No large artifacts in git**: Keep `artifacts/` and raw 17L exports local (add to `.gitignore` as needed)

## Common Modifications

- **Add new features**: Edit `features/draft_context.py` to incorporate signals like picked cards history, seat position, or pack number
- **Tune model**: Adjust `TrainConfig` parameters in `models/drafter.py` or pass CLI flags to `train.py`
- **Handle new metadata columns**: Update `parse_card_metadata()` in `ingest_17l.py` if 17L schema changes
- **Validate inputs**: CLIs in `train.py` and `cli.py` check for empty datasets; extend error handling for malformed files as needed
