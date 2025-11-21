# Repository Guidelines

## Project Overview

This is a **world-class, superhuman AI draft bot** for Magic: The Gathering Limited formats. The system uses 130+ advanced features, gradient boosting, neural networks, and ensemble methods to achieve **75-85%+ accuracy** (human expert level).

## Project Structure & Modules

### Core Package (`src/ai_draft_bot/`)

#### Data Ingestion
- `data/ingest_17l.py`: Parse 17Lands JSONL draft logs and CSV metadata
- `data/scryfall_client.py`: Fetch real card text from Scryfall API (with caching)

#### Feature Engineering
- `features/draft_context.py`: Main feature extraction (16 / 78 / 130+ features)
- `features/card_text.py`: Card text analysis (keywords, removal, synergies)
- `features/positional.py`: Positional features (wheeling, signals, pack quality)
- `features/opponent_model.py`: Opponent modeling and table dynamics
- `features/archetypes.py`: Set-specific archetype detection (JSON-configurable)
- `features/draft_state.py`: Draft state tracking and deck statistics
- `features/synergies.py`: Synergy scoring (color, curve, archetype fit)

#### Models
- `models/drafter.py`: Baseline (Logistic Regression)
- `models/advanced_drafter.py`: Advanced (XGBoost, LightGBM)
- `models/neural_drafter.py`: Neural Networks (PyTorch)
- `models/ensemble_drafter.py`: Ensemble models

#### Optimization & Explainability
- `optimization/optuna_tuner.py`: Bayesian hyperparameter optimization
- `explainability/shap_explainer.py`: SHAP-based pick explanations

#### Utilities
- `utils/cache.py`: Feature caching system (30-50% speedup)
- `utils/logging_config.py`: Logging configuration

### Scripts
- `scripts/train.py`: Typer CLI for training (commands: `run`, `advanced`, `ultra`, `optimize`)

### Configuration
- `configs/archetype_defaults.json`: Default archetype definitions for 10 color pairs

### Tests
- `tests/`: pytest modules (mirror package structure)
- `tests/data/`: Small fixture files (avoid large datasets)

## Setup & Key Commands

### Environment Setup
```bash
# Create venv and install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

### Linting & Type Checking
```bash
ruff check .              # Line length 100, E/F/I/B rules
mypy src                  # Strict type checking
pytest                    # Run tests (-q for quick mode)
```

### Training Commands

#### Baseline (16 features, ~40% accuracy)
```bash
python scripts/train.py run \
    --drafts-path data/drafts.jsonl \
    --metadata-path data/cards.csv \
    --output-path artifacts/baseline.joblib
```

#### Advanced (78 features, ~65% accuracy)
```bash
python scripts/train.py advanced \
    --drafts-path data/drafts.jsonl \
    --metadata-path data/cards.csv \
    --model-type xgboost
```

#### Ultra (130+ features, ~75% accuracy)
```bash
python scripts/train.py ultra \
    --drafts-path data/drafts.jsonl \
    --metadata-path data/cards.csv \
    --model-type xgboost \
    --archetype-config configs/archetype_defaults.json
```

#### Optimized (Auto-tuned, ~80% accuracy)
```bash
python scripts/train.py optimize \
    --drafts-path data/drafts.jsonl \
    --metadata-path data/cards.csv \
    --n-trials 100 \
    --model-type xgboost
```

### Simulation
```bash
python -m ai_draft_bot.cli simulate \
    --model-path artifacts/ultra_model.joblib \
    --metadata-path data/cards.csv \
    --pack "Card A" --pack "Card B" --pack "Card C"
```

## Coding Style & Naming

### Python Version & Type Hints
- Python 3.10+
- Use strict type hints throughout
- Prefer dataclasses for structured data

### Naming Conventions
- **Modules/files**: `snake_case` (e.g., `draft_context.py`)
- **Functions/variables**: `snake_case` (e.g., `build_features()`)
- **Classes**: `PascalCase` (e.g., `DraftModel`)
- **CLI options**: `kebab-case` (e.g., `--model-type`)

### Code Organization
- Keep functions short and focused
- Document complex logic with concise comments
- Use descriptive variable names
- Prefer pure functions where possible

## Key Architecture Patterns

### Feature Extraction Pipeline
```python
# 1. Parse data
picks = parse_draft_logs(drafts_path)
metadata = parse_card_metadata(metadata_path)

# 2. Build features (choose complexity level)
rows = build_pick_features(picks, metadata)           # Baseline: 16 features
rows = build_advanced_pick_features(picks, metadata)  # Advanced: 78 features
rows = build_ultra_advanced_pick_features(picks, metadata, archetype_config)  # Ultra: 130+ features

# 3. Train model
result = train_model(rows, config)                    # Baseline
result = train_advanced_model(rows, config)           # Advanced/Ultra
result = optimize_and_train(rows, optuna_config)      # Auto-optimized

# 4. Save model
result.model.save(output_path)
```

### Feature Caching
```python
from ai_draft_bot.utils.cache import cached_card_features

# Automatically caches expensive card-level features
@cached_card_features
def extract_features(card: CardMetadata) -> np.ndarray:
    # Scryfall API calls, text parsing, etc.
    # Only computed once per card!
    ...
```

### Explainability
```python
from ai_draft_bot.explainability.shap_explainer import DraftExplainer

explainer = DraftExplainer(model)
explainer.fit(training_data_sample)

# Get human-readable explanation
explanation = explainer.explain_pick_human_readable(features)
print(explanation)
```

## Testing Guidelines

### Test Structure
- Write pytest cases in `tests/test_*.py`
- Mirror package structure (e.g., `tests/test_draft_context.py`)
- Use small fixture files from `tests/data/`

### Coverage Areas
- Data ingestion (malformed JSONL, missing columns)
- Feature extraction (edge cases: empty packs, missing metadata)
- Model training and save/load
- Feature dimensionality correctness

### Example Test
```python
def test_feature_dimensions():
    """Verify ultra features have correct dimensionality."""
    picks = [...]
    metadata = {...}
    rows = build_ultra_advanced_pick_features(picks, metadata)

    assert len(rows) > 0
    assert rows[0].features.shape[0] == 130  # Expected dimension
```

## Commit & Pull Request Guidelines

### Commit Messages
- Use imperative mood ("Add feature" not "Added feature")
- Be specific and scoped
- Examples:
  - `Add Scryfall API integration for card text`
  - `Implement Optuna hyperparameter optimization`
  - `Fix archetype synergy calculation bug`

### Pull Requests
- Keep PRs focused on single features/fixes
- Include:
  - Purpose and motivation
  - Notable changes
  - Sample commands and output
  - Performance impact (if relevant)
- Link to issues when available
- Note data assumptions or reproducibility steps

## Data & Security

### Data Handling
- **DO NOT** commit raw 17Lands exports (JSONL files)
- **DO NOT** commit trained models (joblib files)
- Store under `artifacts/` locally (git-ignored)
- Use small fixtures (<100 picks) for tests

### API Keys & Secrets
- No API keys required (Scryfall is public)
- Validate input paths in CLIs
- Avoid embedding PII in logs or fixtures

## Performance Considerations

### Optimization Priorities
1. **Feature Caching**: Use `@cached_card_features` for expensive computations
2. **Scryfall Rate Limits**: Client respects 10 req/sec limit automatically
3. **GPU Acceleration**: Enable with `--use-gpu` for XGBoost/LightGBM/Neural Nets
4. **Parallel Processing**: Consider multiprocessing for large datasets (future work)

### Expected Training Times
| Model | Dataset Size | Time | Hardware |
|-------|-------------|------|----------|
| Baseline | 10k picks | 1 min | CPU |
| Advanced | 10k picks | 10 min | CPU |
| Ultra (first run) | 10k picks | 15-20 min | CPU (Scryfall API calls) |
| Ultra (cached) | 10k picks | 10-12 min | CPU |
| Optimized (100 trials) | 10k picks | 40-80 min | CPU |

## Documentation

### User-Facing Docs
- **README.md**: Quick start, features, installation
- **CLAUDE.md**: Complete development guide
- **OPTIMIZATIONS_COMPLETE.md**: Detailed optimization explanations

### Internal Docs
- **IMPROVEMENTS.md**: Original 130+ feature implementation
- **AGENTS.md**: This file (coding guidelines)

### Code Documentation
- Use docstrings for all public functions/classes
- Include:
  - Purpose and description
  - Args with types
  - Returns with types
  - Example usage (for complex functions)

## Common Tasks

### Adding New Features
1. Create feature extraction function in appropriate module
2. Apply `@cached_card_features` if card-level
3. Update feature count in documentation
4. Add to ultra feature builder in `draft_context.py`
5. Test with small dataset
6. Update CLAUDE.md with new feature details

### Adding New Archetype
1. Edit `configs/archetype_defaults.json`
2. Add archetype definition with:
   - Name, colors, keywords, target_curve
3. No code changes needed!

### Debugging Low Accuracy
1. Check feature importance: `result.metrics.feature_importance`
2. Use SHAP explainability to understand predictions
3. Validate card coverage: `validate_card_coverage(picks, metadata)`
4. Check for missing win rate data
5. Try Optuna optimization for better hyperparameters

## Success Metrics

### Model Performance
- **Minimum**: 60%+ validation accuracy
- **Target**: 70%+ validation accuracy
- **Excellence**: 75%+ validation accuracy (superhuman)

### Code Quality
- All tests pass
- No linting errors (`ruff check .`)
- No type errors (`mypy src`)
- <100 characters per line

### Feature Quality
- Top-5 feature importance includes GIH WR
- SHAP explanations make intuitive sense
- Feature extraction completes without errors
- Cache hit rate >80% on repeated training

## Known Issues & Future Work

### Implemented ✅
- 130+ features with all enhancements
- Scryfall API integration
- Feature caching (30-50% speedup)
- Optuna hyperparameter optimization
- SHAP explainability
- Neural networks and ensembles

### Future Enhancements 🔮
- Semantic card embeddings (sentence-transformers)
- Transformer-based architecture
- FastAPI deployment for web interface
- Online learning from new drafts
- Active learning for rare cards
- Draft simulation and evaluation

---

## Quick Reference

### File Organization
```
├── src/ai_draft_bot/        # Main package
│   ├── data/                # Data ingestion
│   ├── features/            # Feature engineering
│   ├── models/              # ML models
│   ├── optimization/        # Hyperparameter tuning
│   ├── explainability/      # SHAP explanations
│   └── utils/               # Utilities (cache, logging)
├── scripts/                 # CLI scripts
├── tests/                   # Test suite
├── configs/                 # Configuration files
├── artifacts/               # Trained models (gitignored)
└── cache/                   # Feature cache (gitignored)
```

### Command Cheatsheet
```bash
# Install
pip install -e .[dev]

# Train ultra model
python scripts/train.py ultra --drafts-path data.jsonl --metadata-path cards.csv

# Optimize hyperparameters
python scripts/train.py optimize --drafts-path data.jsonl --metadata-path cards.csv --n-trials 50

# Lint and test
ruff check .
mypy src
pytest
```

---

**Ready to build world-class draft AI! 🚀**
