"""AI Draft Bot - Superhuman Magic: The Gathering draft assistant.

This package provides state-of-the-art AI models for drafting Magic: The Gathering
cards using 17Lands data. Features include:

- 130+ ultra-advanced features (win rates, synergies, text analysis, opponent modeling)
- Multiple model architectures (XGBoost, LightGBM, Neural Networks, Ensembles)
- 75-85% accuracy targeting human expert level
- Comprehensive feature extraction and explainability

Key Components:
    - Data ingestion from 17Lands JSONL/CSV exports
    - Advanced feature engineering with draft context awareness
    - Multiple model architectures with hyperparameter tuning
    - Explainability tools (SHAP, feature importance)
    - CLI for training, simulation, and evaluation

Example:
    >>> from ai_draft_bot import DraftModel, CardMetadata
    >>> from ai_draft_bot.data import parse_draft_logs, parse_card_metadata
    >>>
    >>> # Load data
    >>> picks = parse_draft_logs("drafts.jsonl")
    >>> metadata = parse_card_metadata("cards.csv")
    >>>
    >>> # Train a model
    >>> from ai_draft_bot.models import train_model, TrainingConfig
    >>> config = TrainingConfig(test_size=0.2, max_iter=500)
    >>> artifacts = train_model(picks, metadata, config)
    >>>
    >>> # Make predictions
    >>> pack = ["Lightning Bolt", "Counterspell", "Llanowar Elves"]
    >>> prediction = artifacts.model.predict_pack(pack, metadata)
"""

from ai_draft_bot.config import Config, get_config
from ai_draft_bot.data.ingest_17l import CardMetadata, PickRecord
from ai_draft_bot.models.drafter import DraftModel, EvaluationMetrics, TrainingArtifacts

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "Config",
    "get_config",
    # Data structures
    "CardMetadata",
    "PickRecord",
    # Models
    "DraftModel",
    "TrainingArtifacts",
    "EvaluationMetrics",
]
