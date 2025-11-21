"""Ensemble models combining multiple draft prediction models.

This module provides sophisticated ensemble methods that combine:
- XGBoost
- LightGBM
- Neural Networks

For maximum accuracy through model diversity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ai_draft_bot.features.draft_context import PickFeatures
from ai_draft_bot.models.advanced_drafter import (
    AdvancedTrainConfig,
    ModelType,
    train_advanced_model,
)
from ai_draft_bot.models.neural_drafter import NeuralTrainConfig, train_neural_model

logger = logging.getLogger("ai_draft_bot.models.ensemble_drafter")


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training.

    Attributes:
        test_size: Validation split
        random_state: Random seed
        xgboost_config: Config for XGBoost model
        lightgbm_config: Config for LightGBM model
        neural_config: Config for Neural Net model
        ensemble_method: How to combine predictions ('voting', 'weighted', 'stacking')
        weights: Optional custom weights for weighted averaging
    """

    test_size: float = 0.2
    random_state: int = 13
    xgboost_config: AdvancedTrainConfig | None = None
    lightgbm_config: AdvancedTrainConfig | None = None
    neural_config: NeuralTrainConfig | None = None
    ensemble_method: str = "weighted"  # 'voting', 'weighted', 'stacking'
    weights: List[float] | None = None  # For weighted averaging


@dataclass
class EnsembleArtifacts:
    """Artifacts from ensemble training."""

    xgboost_model: object | None
    lightgbm_model: object | None
    neural_model: object | None
    label_encoder: LabelEncoder
    ensemble_method: str
    weights: List[float] | None


@dataclass
class EnsembleMetrics:
    """Evaluation metrics for ensemble."""

    accuracy: float
    train_samples: int
    validation_samples: int
    xgboost_accuracy: float
    lightgbm_accuracy: float
    neural_accuracy: float


class EnsembleDraftModel:
    """Ensemble draft model combining multiple base models."""

    def __init__(self, artifacts: EnsembleArtifacts):
        self.artifacts = artifacts

    def predict(self, features: np.ndarray) -> str:
        """Predict using ensemble of models.

        Args:
            features: Feature vector for a single pick

        Returns:
            Card name prediction
        """
        predictions = self.predict_proba(features)
        best_card = max(predictions.items(), key=lambda x: x[1])[0]
        return best_card

    def predict_proba(self, features: np.ndarray) -> Mapping[str, float]:
        """Predict probability distribution using ensemble.

        Args:
            features: Feature vector for a single pick

        Returns:
            Dictionary mapping card names to ensemble probabilities
        """
        proba_dists = []
        weights = self.artifacts.weights or []

        # Collect predictions from each model
        if self.artifacts.xgboost_model is not None:
            xgb_proba = self.artifacts.xgboost_model.predict_proba(features)
            proba_dists.append(xgb_proba)

        if self.artifacts.lightgbm_model is not None:
            lgb_proba = self.artifacts.lightgbm_model.predict_proba(features)
            proba_dists.append(lgb_proba)

        if self.artifacts.neural_model is not None:
            nn_proba = self.artifacts.neural_model.predict_proba(features)
            proba_dists.append(nn_proba)

        if not proba_dists:
            raise ValueError("No models in ensemble!")

        # Combine predictions
        if self.artifacts.ensemble_method == "voting":
            # Simple average
            return self._average_predictions(proba_dists)
        elif self.artifacts.ensemble_method == "weighted":
            # Weighted average
            return self._weighted_average(proba_dists, weights)
        else:
            # Default to simple average
            return self._average_predictions(proba_dists)

    def _average_predictions(
        self, proba_dists: List[Mapping[str, float]]
    ) -> Mapping[str, float]:
        """Simple average of probability distributions."""
        combined: dict[str, float] = {}
        card_names = set()

        for dist in proba_dists:
            card_names.update(dist.keys())

        for card in card_names:
            probs = [dist.get(card, 0.0) for dist in proba_dists]
            combined[card] = float(np.mean(probs))

        return combined

    def _weighted_average(
        self, proba_dists: List[Mapping[str, float]], weights: List[float]
    ) -> Mapping[str, float]:
        """Weighted average of probability distributions."""
        if not weights or len(weights) != len(proba_dists):
            # Fall back to simple average
            return self._average_predictions(proba_dists)

        # Normalize weights
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()

        combined: dict[str, float] = {}
        card_names = set()

        for dist in proba_dists:
            card_names.update(dist.keys())

        for card in card_names:
            probs = [dist.get(card, 0.0) for dist in proba_dists]
            weighted_prob = sum(p * w for p, w in zip(probs, weights_array))
            combined[card] = float(weighted_prob)

        return combined

    def save(self, path: Path | str) -> None:
        """Save ensemble to disk."""
        payload = {
            "xgboost_model": self.artifacts.xgboost_model,
            "lightgbm_model": self.artifacts.lightgbm_model,
            "neural_model": self.artifacts.neural_model,
            "label_encoder": self.artifacts.label_encoder,
            "ensemble_method": self.artifacts.ensemble_method,
            "weights": self.artifacts.weights,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "EnsembleDraftModel":
        """Load ensemble from disk."""
        payload = joblib.load(path)
        artifacts = EnsembleArtifacts(
            xgboost_model=payload.get("xgboost_model"),
            lightgbm_model=payload.get("lightgbm_model"),
            neural_model=payload.get("neural_model"),
            label_encoder=payload["label_encoder"],
            ensemble_method=payload.get("ensemble_method", "weighted"),
            weights=payload.get("weights"),
        )
        return cls(artifacts)


@dataclass
class EnsembleTrainResult:
    """Result of ensemble training."""

    model: EnsembleDraftModel
    metrics: EnsembleMetrics


def train_ensemble_model(
    rows: Sequence[PickFeatures], *, config: EnsembleConfig | None = None
) -> EnsembleTrainResult:
    """Train an ensemble of multiple models.

    Args:
        rows: Training data
        config: Ensemble configuration

    Returns:
        EnsembleTrainResult with trained ensemble and metrics
    """
    if config is None:
        config = EnsembleConfig()

    logger.info("=" * 60)
    logger.info("ENSEMBLE MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Dataset size: {len(rows)} picks")
    logger.info("Training models: XGBoost, LightGBM, Neural Network")

    # Extract labels for label encoder
    labels = [row.label for row in rows]
    encoder = LabelEncoder()
    encoder.fit(labels)

    # Split data
    train_rows, val_rows = train_test_split(
        rows, test_size=config.test_size, random_state=config.random_state
    )

    logger.info(f"Train/validation split: {len(train_rows)}/{len(val_rows)}")

    # Train XGBoost
    logger.info("\n🔧 Training XGBoost model...")
    xgb_config = config.xgboost_config or AdvancedTrainConfig(
        model_type=ModelType.XGBOOST,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    xgb_result = train_advanced_model(rows, config=xgb_config)
    xgb_accuracy = xgb_result.metrics.accuracy
    logger.info(f"✓ XGBoost accuracy: {xgb_accuracy:.4f}")

    # Train LightGBM
    logger.info("\n🔧 Training LightGBM model...")
    lgb_config = config.lightgbm_config or AdvancedTrainConfig(
        model_type=ModelType.LIGHTGBM,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    lgb_result = train_advanced_model(rows, config=lgb_config)
    lgb_accuracy = lgb_result.metrics.accuracy
    logger.info(f"✓ LightGBM accuracy: {lgb_accuracy:.4f}")

    # Train Neural Network
    logger.info("\n🔧 Training Neural Network model...")
    nn_config = config.neural_config or NeuralTrainConfig(
        test_size=config.test_size,
        random_state=config.random_state,
    )
    nn_result = train_neural_model(rows, config=nn_config)
    nn_accuracy = nn_result.metrics.accuracy
    logger.info(f"✓ Neural Network accuracy: {nn_accuracy:.4f}")

    # Determine ensemble weights based on validation accuracy
    if config.weights is None and config.ensemble_method == "weighted":
        # Weight by validation accuracy
        accuracies = np.array([xgb_accuracy, lgb_accuracy, nn_accuracy])
        weights = accuracies / accuracies.sum()
        config.weights = weights.tolist()
        logger.info(f"\n📊 Auto-computed weights: {config.weights}")

    # Evaluate ensemble on validation set
    logger.info("\n🎯 Evaluating ensemble...")
    artifacts = EnsembleArtifacts(
        xgboost_model=xgb_result.model,
        lightgbm_model=lgb_result.model,
        neural_model=nn_result.model,
        label_encoder=encoder,
        ensemble_method=config.ensemble_method,
        weights=config.weights,
    )

    ensemble_model = EnsembleDraftModel(artifacts)

    # Compute ensemble accuracy on validation set
    correct = 0
    for row in val_rows:
        pred = ensemble_model.predict(row.features)
        if pred == row.label:
            correct += 1

    ensemble_accuracy = correct / len(val_rows)
    logger.info(f"✓ Ensemble accuracy: {ensemble_accuracy:.4f}")

    # Build metrics
    metrics = EnsembleMetrics(
        accuracy=ensemble_accuracy,
        train_samples=len(train_rows),
        validation_samples=len(val_rows),
        xgboost_accuracy=xgb_accuracy,
        lightgbm_accuracy=lgb_accuracy,
        neural_accuracy=nn_accuracy,
    )

    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"XGBoost:  {xgb_accuracy:.4f}")
    logger.info(f"LightGBM: {lgb_accuracy:.4f}")
    logger.info(f"Neural:   {nn_accuracy:.4f}")
    logger.info(f"Ensemble: {ensemble_accuracy:.4f}")
    logger.info("=" * 60)

    return EnsembleTrainResult(model=ensemble_model, metrics=metrics)
