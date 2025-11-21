# AI Draft Bot 🤖🃏

An **advanced AI-powered draft bot** for Magic: The Gathering Limited formats, trained on [17Lands](https://www.17lands.com/) data. This project implements state-of-the-art machine learning techniques to build the world's most sophisticated draft pick predictor.

## 🌟 Features

### Two-Tier Architecture
- **Baseline Model**: Fast logistic regression (16 features, ~40% accuracy)
- **Advanced Model**: Gradient boosting with XGBoost/LightGBM (78 features, ~60-70% accuracy)

### Advanced Feature Engineering (78 Features)
- **Win Rate Integration**: GIH WR, OH WR, GD WR, IWD, ALSA from 17Lands
- **Draft Context**: Pick/pack number, cards picked so far, deck composition
- **Deck Statistics**: Mana curve, color commitment, creature/spell ratio, removal count
- **Synergy Detection**: Color fit, mana curve fit, archetype coherence
- **Pack Signals**: Bomb count, rare count, win rate distribution, pack size

### State-of-the-Art Models
- **XGBoost**: Extreme gradient boosting with early stopping
- **LightGBM**: Memory-efficient gradient boosting with GPU support
- **Feature Importance**: Understand which signals drive decisions

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai_draft_bot.git
cd ai_draft_bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .[dev]
```

### Training Advanced Model

```bash
python scripts/train.py advanced \
    --drafts-path data/drafts.jsonl \
    --metadata-path data/cards.csv \
    --output-path artifacts/advanced_model.joblib \
    --model-type xgboost
```

## 📊 Architecture

**78 Advanced Features:**
- Card win rates (GIH WR, IWD, ALSA)
- Draft context (pick/pack number)
- Deck composition (mana curve, colors)
- Synergy scores (color fit, archetype)

**Models:**
- XGBoost (recommended)
- LightGBM (faster, lower memory)

See [CLAUDE.md](CLAUDE.md) for full documentation.

---

**Happy Drafting! 🎉**
