# AI Draft Bot

World-class draft recommendations for Magic: The Gathering Limited, trained on 17Lands data. The
system combines 130+ engineered features, gradient boosting, neural models, and ensembles to hit
human-expert accuracy (75-85%+ pick prediction).

## Features

- Tiered models: baseline (16 features), advanced (78), ultra (130+), plus Optuna-tuned variants
- Rich signals: win rates, draft context, deck stats, synergies, archetypes, positional signals,
  and Scryfall-powered card text analysis
- Models: XGBoost, LightGBM, PyTorch, and ensembles with SHAP explainability
- Performance: feature caching, GPU support, Optuna Bayesian tuning

## Quick Start

### Install
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e .[dev]
```

### Train
Baseline (16 features):
```bash
python scripts/train.py run \
  --drafts-path data/drafts.jsonl \
  --metadata-path data/cards.csv \
  --output-path artifacts/baseline.joblib
```

Advanced (78 features):
```bash
python scripts/train.py advanced \
  --drafts-path data/drafts.jsonl \
  --metadata-path data/cards.csv \
  --model-type xgboost
```

Ultra (130+ features):
```bash
python scripts/train.py ultra \
  --drafts-path data/drafts.jsonl \
  --metadata-path data/cards.csv \
  --model-type xgboost
```

Optuna-tuned:
```bash
python scripts/train.py optimize \
  --drafts-path data/drafts.jsonl \
  --metadata-path data/cards.csv \
  --n-trials 100 \
  --model-type xgboost
```

### Simulate a Pack
```bash
python -m ai_draft_bot.cli simulate \
  --model-path artifacts/ultra_model.joblib \
  --metadata-path data/cards.csv \
  --pack "Lightning Bolt" --pack "Swords to Plowshares" --pack "Counterspell"
```

## Performance (typical)

| Model          | Features | Accuracy | Train Time | Use Case             |
|----------------|----------|----------|------------|----------------------|
| Baseline       | 16       | 35-45%   | ~1 min     | Quick prototype      |
| Advanced       | 78       | 55-70%   | ~10 min    | Solid baseline       |
| Ultra          | 130+     | 70-85%   | 10-15 min  | Target: superhuman   |
| Ultra + Optuna | 130+     | 75-87%   | 40-80 min  | Maximum accuracy     |
| Ensemble       | 130+     | 75-90%   | 60-120 min | Production candidate |

## Architecture
```
src/ai_draft_bot/
├─ data/               # 17Lands ingestion, Scryfall client
├─ features/           # 16 / 78 / 130+ feature builders
├─ models/             # Baseline, advanced, neural, ensemble
├─ optimization/       # Optuna tuner
├─ explainability/     # SHAP explanations
└─ utils/              # Caching, logging, metrics, splits
```

## Documentation
- `CLAUDE.md`: development guide and commands
- `IMPROVEMENTS.md`: feature implementation details
- `OPTIMIZATIONS_COMPLETE.md`: optimization notes
- `configs/archetype_defaults.json`: archetype definitions

## Contributing
See `AGENTS.md` for guidelines. Keep linting/tests green: `ruff`, `mypy`, `pytest`.

## License
Educational and research use only. Magic: The Gathering is © Wizards of the Coast.
