# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **advanced AI draft bot** for Magic: The Gathering Limited using 17Lands (17L) data exports. The system provides both baseline (logistic regression) and advanced (XGBoost/LightGBM) models with sophisticated feature engineering including win rates, draft context, synergy detection, and archetype awareness.

**Key Capabilities:**
- 75+ advanced features (vs 16 baseline)
- Win rate integration (GIH WR, IWD, ALSA)
- Draft context tracking (pick history, deck composition)
- Synergy scoring (color fit, mana curve, archetype coherence)
- State-of-the-art gradient boosting (XGBoost, LightGBM)
- Feature importance analysis

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

**Training a Baseline Model (Logistic Regression):**
```bash
python scripts/train.py run \
    --drafts-path path/to/drafts.jsonl \
    --metadata-path path/to/cards.csv \
    --output-path artifacts/baseline_model.joblib \
    --test-size 0.2 \
    --max-iter 500 \
    --c 1.0 \
    --random-state 13
```

**Training an ADVANCED Model (XGBoost - RECOMMENDED):**
```bash
python scripts/train.py advanced \
    --drafts-path path/to/drafts.jsonl \
    --metadata-path path/to/cards.csv \
    --output-path artifacts/advanced_model.joblib \
    --model-type xgboost \
    --n-estimators 500 \
    --max-depth 8 \
    --learning-rate 0.1 \
    --early-stopping 50 \
    --test-size 0.2
```

**Training with LightGBM (faster, lower memory):**
```bash
python scripts/train.py advanced \
    --drafts-path path/to/drafts.jsonl \
    --metadata-path path/to/cards.csv \
    --model-type lightgbm \
    --use-gpu  # Optional: GPU acceleration
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

2. **Feature Extraction** (`src/ai_draft_bot/features/`)

   **Baseline Features** (`draft_context.py::build_pick_features()`):
   - Card-level: mana value, rarity, color one-hot (8 features)
   - Pack-level: mean mana, rarity, color distribution (8 features)
   - Total: 16 features

   **Advanced Features** (`draft_context.py::build_advanced_pick_features()`):
   - Chosen card: basic stats + win rates (13 features)
   - Pack aggregates: mean, max, std dev (29 features)
   - Pack signals: size, bomb count, win rate distribution (5 features)
   - Context: pick/pack number (2 features)
   - Deck state: mana curve, color commitment, creature/spell ratio (23 features)
   - Synergy: color fit, curve fit, archetype coherence (6 features)
   - Total: 78 features

   **Draft State Tracking** (`draft_state.py`):
   - `DraftState`: Tracks cards picked chronologically
   - `DeckStats`: Computes deck composition metrics
   - Enables sequential feature extraction that mirrors real drafting

   **Synergy Detection** (`synergies.py`):
   - Color synergy: How well card fits deck colors
   - Mana curve synergy: Filling gaps vs oversaturation
   - Archetype synergy: Detect fliers/control/aggro/etc.
   - Creature/spell balance: Target ~60% creatures

3. **Model Training**

   **Baseline** (`models/drafter.py`):
   - `train_model()`: Logistic regression with scikit-learn
   - Fast, interpretable, good starting point

   **Advanced** (`models/advanced_drafter.py`):
   - `train_advanced_model()`: XGBoost or LightGBM
   - Gradient boosting with early stopping
   - Feature importance analysis
   - 500 estimators, max depth 8, learning rate 0.1
   - GPU support for faster training
   - Typically achieves 15-30% higher accuracy than baseline

4. **Entry Points**
   - `scripts/train.py`: Typer CLI for training workflow
   - `src/ai_draft_bot/cli.py`: Typer CLI for simulation and evaluation

### Key Design Decisions

- **Streaming ingestion**: 17L exports can be large; `load_jsonl()` yields records to keep memory usage low
- **Two-tier architecture**: Baseline (simple, fast) vs Advanced (sophisticated, accurate)
- **Win rate first**: GIH WR is the single most predictive feature - always use 17L exports with win rate columns
- **Sequential context**: Draft state tracking allows the model to understand what's been picked previously
- **Synergy-aware**: Not just picking "best card in pack" but "best card for this deck"
- **Gradient boosting**: XGBoost/LightGBM handle feature interactions non-linearly (unlike logistic regression)
- **Feature importance**: See which signals matter most (typically: GIH WR, color fit, deck synergy)
- **Multiclass classification**: Each pick is a multiclass problem with card names as labels
- **Joblib serialization**: Models saved with label encoders for easy deployment

## Code Organization

```
src/ai_draft_bot/
├── data/
│   └── ingest_17l.py           # JSONL/CSV parsing, PickRecord/CardMetadata (with win rates)
├── features/
│   ├── draft_context.py        # Feature extraction (baseline + advanced)
│   ├── draft_state.py          # Draft state tracking, deck statistics
│   └── synergies.py            # Synergy scoring, archetype detection
├── models/
│   ├── drafter.py              # Baseline model (Logistic Regression)
│   └── advanced_drafter.py     # Advanced models (XGBoost, LightGBM)
└── cli.py                      # Simulation CLI

scripts/
└── train.py                    # Training CLI (run | advanced commands)

tests/
└── (add test_*.py here)        # Mirror package structure; use small fixtures
```

## Coding Conventions

- **Python 3.10+** with strict mypy type hints
- **Naming**: modules/files `snake_case`, classes `PascalCase`, functions/vars `snake_case`, CLI options `kebab-case`
- **Fixtures**: Store under `tests/data/` to avoid polluting source directories
- **No large artifacts in git**: Keep `artifacts/` and raw 17L exports local (add to `.gitignore` as needed)

## Common Modifications

- **Add new features**:
  - Basic features: Edit `card_to_vector()` in `features/draft_context.py`
  - Advanced features: Extend `build_advanced_pick_features()` in `features/draft_context.py`
  - Synergy heuristics: Update `features/synergies.py` with set-specific mechanics

- **Tune hyperparameters**:
  - XGBoost: Adjust `n_estimators`, `max_depth`, `learning_rate` in CLI or `AdvancedTrainConfig`
  - Common tuning: Increase `max_depth` (8→12) for complex sets, decrease `learning_rate` (0.1→0.05) for stability

- **Handle new metadata columns**:
  - Update `parse_card_metadata()` in `ingest_17l.py` if 17L schema changes
  - Add new win rate columns to `winrate_column_map`

- **Improve synergy detection**:
  - Add set-specific archetype definitions in `synergies.py::COLOR_PAIR_ARCHETYPES`
  - Implement NLP-based card text analysis for mechanic synergies
  - Use pre-computed synergy matrices from domain experts

## Performance Expectations

Based on typical 17Lands datasets:

| Model | Features | Accuracy | Training Time | Use Case |
|-------|----------|----------|---------------|----------|
| Baseline LogReg | 16 | 35-45% | Fast (~1 min) | Quick prototyping |
| Advanced XGBoost | 78 | 55-70% | Medium (~10 min) | Production (recommended) |
| Advanced LightGBM | 78 | 55-70% | Fast (~5 min) | Large datasets |

**Notes:**
- Accuracy varies by set complexity and data quality
- Win rate features typically add +15-20% accuracy
- Draft context/synergy features add +5-10% accuracy
- Top human drafters achieve ~75-85% pick prediction accuracy
