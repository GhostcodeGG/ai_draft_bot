"""Advanced draft models using gradient boosting and ensemble methods.

This module provides state-of-the-art models for draft pick prediction:
- XGBoost: Extreme gradient boosting (fast, accurate)
- LightGBM: Light gradient boosting (even faster, memory efficient)
- Ensemble: Combines multiple models for maximum performance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("ai_draft_bot.models.advanced_drafter")

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore

from ai_draft_bot.features.draft_context import PickFeatures
from ai_draft_bot.utils.metrics import ndcg_at_ks, topk_accuracies
from ai_draft_bot.utils.splits import train_val_split_by_event


class ModelType(str, Enum):
    """Available model architectures."""

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"


@dataclass
class AdvancedTrainConfig:
    """Configuration for advanced model training.

    Attributes:
        model_type: Which algorithm to use
        test_size: Fraction of data for validation
        random_state: Random seed for reproducibility
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage
        early_stopping_rounds: Stop if no improvement for N rounds
        use_gpu: Whether to use GPU acceleration (if available)
    """

    model_type: ModelType = ModelType.XGBOOST
    test_size: float = 0.2
    random_state: int = 13
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.1
    early_stopping_rounds: int = 50
    use_gpu: bool = False


@dataclass
class AdvancedTrainingArtifacts:
    """Training artifacts for advanced models."""

    model: object  # XGBoost or LightGBM model
    label_encoder: LabelEncoder
    model_type: ModelType
    feature_importance: np.ndarray | None = None


@dataclass
class AdvancedEvaluationMetrics:
    """Evaluation metrics for advanced models."""

    accuracy: float
    train_samples: int
    validation_samples: int
    feature_importance: Mapping[int, float] | None = None
    top_k: Mapping[int, float] | None = None
    ndcg: Mapping[int, float] | None = None


class AdvancedDraftModel:
    """Advanced draft model using gradient boosting."""

    def __init__(self, artifacts: AdvancedTrainingArtifacts):
        self.artifacts = artifacts

    def predict(self, features: np.ndarray) -> str:
        """Predict the most likely card pick.

        Args:
            features: Feature vector for a single pick

        Returns:
            Card name prediction
        """
        features_2d = features.reshape(1, -1)

        if self.artifacts.model_type == ModelType.XGBOOST:
            if xgb is None:
                raise ImportError("XGBoost not installed")
            # XGBoost returns class indices
            pred_idx = self.artifacts.model.predict(features_2d)[0]
            pred_idx = int(pred_idx)
        elif self.artifacts.model_type == ModelType.LIGHTGBM:
            if lgb is None:
                raise ImportError("LightGBM not installed")
            pred_idx = self.artifacts.model.predict(features_2d)[0]
            pred_idx = int(np.argmax(pred_idx))
        else:
            raise ValueError(f"Unknown model type: {self.artifacts.model_type}")

        return str(self.artifacts.label_encoder.inverse_transform([pred_idx])[0])

    def predict_proba(self, features: np.ndarray) -> Mapping[str, float]:
        """Predict probability distribution over all cards.

        Args:
            features: Feature vector for a single pick

        Returns:
            Dictionary mapping card names to probabilities
        """
        features_2d = features.reshape(1, -1)

        if self.artifacts.model_type == ModelType.XGBOOST:
            if xgb is None:
                raise ImportError("XGBoost not installed")
            # For multiclass, XGBoost returns probability matrix
            proba = self.artifacts.model.predict_proba(features_2d)[0]
        elif self.artifacts.model_type == ModelType.LIGHTGBM:
            if lgb is None:
                raise ImportError("LightGBM not installed")
            proba = self.artifacts.model.predict(features_2d)[0]
        else:
            raise ValueError(f"Unknown model type: {self.artifacts.model_type}")

        labels = self.artifacts.label_encoder.inverse_transform(np.arange(len(proba)))
        return {str(label): float(prob) for label, prob in zip(labels, proba)}

    def get_feature_importance(self) -> np.ndarray | None:
        """Get feature importance scores if available."""
        return self.artifacts.feature_importance

    def save(self, path: Path | str) -> None:
        """Save model to disk."""
        payload = {
            "model": self.artifacts.model,
            "label_encoder": self.artifacts.label_encoder,
            "model_type": self.artifacts.model_type.value,
            "feature_importance": self.artifacts.feature_importance,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "AdvancedDraftModel":
        """Load model from disk."""
        payload = joblib.load(path)
        artifacts = AdvancedTrainingArtifacts(
            model=payload["model"],
            label_encoder=payload["label_encoder"],
            model_type=ModelType(payload["model_type"]),
            feature_importance=payload.get("feature_importance"),
        )
        return cls(artifacts)


@dataclass
class AdvancedTrainResult:
    """Result of advanced model training."""

    model: AdvancedDraftModel
    metrics: AdvancedEvaluationMetrics


def _encode_dataset(
    rows: Sequence[PickFeatures], encoder: LabelEncoder | None = None
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Encode dataset into feature matrix and label vector."""
    features = np.vstack([row.features for row in rows])
    labels = [row.label for row in rows]
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    return features, encoded_labels, encoder


def train_xgboost_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: AdvancedTrainConfig,
) -> Tuple[object, float, np.ndarray, np.ndarray]:
    """Train XGBoost model.

    Returns:
        (model, validation_accuracy, feature_importance)
    """
    if xgb is None:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    num_classes = len(np.unique(y_train))

    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "random_state": config.random_state,
        "eval_metric": "mlogloss",
        "tree_method": "gpu_hist" if config.use_gpu else "hist",
    }

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)

    evals = [(dtrain, "train"), (dval, "validation")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=config.n_estimators,
        evals=evals,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )

    # Evaluate
    preds = model.predict(dval)
    pred_labels = np.argmax(preds, axis=1)
    accuracy = float(np.mean(pred_labels == y_val))

    # Feature importance
    importance = model.get_score(importance_type="gain")
    # Convert to array indexed by feature
    importance_array = np.zeros(x_train.shape[1])
    for feat_name, score in importance.items():
        feat_idx = int(feat_name[1:])  # Remove 'f' prefix
        importance_array[feat_idx] = score

    return model, accuracy, importance_array, preds


def train_lightgbm_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: AdvancedTrainConfig,
) -> Tuple[object, float, np.ndarray, np.ndarray]:
    """Train LightGBM model.

    Returns:
        (model, validation_accuracy, feature_importance)
    """
    if lgb is None:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")

    num_classes = len(np.unique(y_train))

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "random_state": config.random_state,
        "metric": "multi_logloss",
        "device": "gpu" if config.use_gpu else "cpu",
        "verbose": -1,
    }

    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=config.n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=["train", "validation"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=False)
        ],
    )

    # Evaluate
    preds = model.predict(x_val)
    pred_labels = np.argmax(preds, axis=1)
    accuracy = float(np.mean(pred_labels == y_val))

    # Feature importance
    importance_array = model.feature_importance(importance_type="gain")

    return model, accuracy, importance_array, preds


def train_advanced_model(
    rows: Sequence[PickFeatures], *, config: AdvancedTrainConfig | None = None
) -> AdvancedTrainResult:
    """Train an advanced draft model using gradient boosting.

    Args:
        rows: Training data (PickFeatures with labels)
        config: Training configuration

    Returns:
        AdvancedTrainResult with trained model and metrics
    """
    if config is None:
        config = AdvancedTrainConfig()

    logger.info("=" * 60)
    logger.info(f"ADVANCED MODEL TRAINING ({config.model_type.value.upper()})")
    logger.info("=" * 60)
    logger.info(f"Dataset size: {len(rows)} picks")
    if not rows:
        raise ValueError("No rows provided for training")

    # Draft-aware split
    split = train_val_split_by_event(rows, test_size=config.test_size, random_state=config.random_state)
    train_rows, val_rows = split.train, split.val
    if not val_rows:
        raise ValueError("Validation split is empty. Provide drafts from more than one event.")

    # Encode dataset with shared encoder (fit on all labels to avoid unseen classes)
    _, _, encoder = _encode_dataset(rows)
    x_train, y_train, _ = _encode_dataset(train_rows, encoder)
    x_val, y_val, _ = _encode_dataset(val_rows, encoder)

    logger.info(f"Feature dimension: {x_train.shape[1]}")
    logger.info(f"Unique labels (cards): {len(encoder.classes_)}")
    logger.info(
        f"Train/validation split (by draft): {len(x_train)}/{len(x_val)} "
        f"({config.test_size:.0%} validation)"
    )
    logger.info(
        f"Hyperparameters: n_estimators={config.n_estimators}, max_depth={config.max_depth}, "
        f"learning_rate={config.learning_rate}"
    )
    if config.use_gpu:
        logger.info("GPU acceleration: ENABLED")

    # Train model based on type
    logger.info(f"Training {config.model_type.value} model...")
    if config.model_type == ModelType.XGBOOST:
        model, accuracy, importance, val_probs = train_xgboost_model(
            x_train, y_train, x_val, y_val, config
        )
    elif config.model_type == ModelType.LIGHTGBM:
        model, accuracy, importance, val_probs = train_lightgbm_model(
            x_train, y_train, x_val, y_val, config
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    top_k = topk_accuracies(val_probs, y_val, ks=(1, 3, 5))
    ndcg = ndcg_at_ks(val_probs, y_val, ks=(1, 3, 5))

    logger.info(f"Validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Top-k accuracy: { {k: round(v, 3) for k, v in top_k.items()} }")
    logger.info(f"NDCG: { {k: round(v, 3) for k, v in ndcg.items()} }")

    # Create artifacts
    artifacts = AdvancedTrainingArtifacts(
        model=model,
        label_encoder=encoder,
        model_type=config.model_type,
        feature_importance=importance,
    )

    # Build metrics
    metrics = AdvancedEvaluationMetrics(
        accuracy=accuracy,
        train_samples=len(x_train),
        validation_samples=len(x_val),
        feature_importance={i: float(importance[i]) for i in range(len(importance))},
        top_k=top_k,
        ndcg=ndcg,
    )

    # Log feature importance summary
    if importance is not None:
        top_k = 10
        top_indices = np.argsort(importance)[-top_k:][::-1]
        logger.info(f"Top {top_k} most important features:")
        for idx in top_indices:
            logger.info(f"  Feature {idx}: {importance[idx]:.4f}")

    logger.info("Training complete!")
    return AdvancedTrainResult(model=AdvancedDraftModel(artifacts), metrics=metrics)
