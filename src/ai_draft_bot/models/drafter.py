"""Baseline draft model built on scikit-learn."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ai_draft_bot.features.draft_context import PickFeatures

logger = logging.getLogger("ai_draft_bot.models.drafter")


@dataclass
class TrainingArtifacts:
    model: LogisticRegression
    label_encoder: LabelEncoder


@dataclass
class EvaluationMetrics:
    accuracy: float
    train_samples: int
    validation_samples: int


class DraftModel:
    """Wrapper around a multiclass logistic regression model."""

    def __init__(self, artifacts: TrainingArtifacts):
        self.artifacts = artifacts

    def predict(self, features: np.ndarray) -> str:
        encoded = self.artifacts.model.predict(features.reshape(1, -1))
        return str(self.artifacts.label_encoder.inverse_transform(encoded)[0])

    def predict_proba(self, features: np.ndarray) -> Mapping[str, float]:
        scores = self.artifacts.model.predict_proba(features.reshape(1, -1))[0]
        labels = self.artifacts.label_encoder.inverse_transform(
            np.arange(len(scores))
        )
        return {label: float(score) for label, score in zip(labels, scores)}

    def save(self, path: Path | str) -> None:
        payload = {
            "model": self.artifacts.model,
            "label_encoder": self.artifacts.label_encoder,
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "DraftModel":
        logger.info(f"Loading model from {path}")
        payload = joblib.load(path)
        artifacts = TrainingArtifacts(
            model=payload["model"],
            label_encoder=payload["label_encoder"],
        )
        logger.info(f"Model loaded successfully ({len(artifacts.label_encoder.classes_)} classes)")
        return cls(artifacts)


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 13
    max_iter: int = 500
    C: float = 1.0


@dataclass
class TrainResult:
    model: DraftModel
    metrics: EvaluationMetrics


def _encode_dataset(rows: Sequence[PickFeatures]) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    features = np.vstack([row.features for row in rows])
    labels = [row.label for row in rows]
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    return features, encoded_labels, encoder


def train_model(rows: Sequence[PickFeatures], *, config: TrainConfig | None = None) -> TrainResult:
    """Train the baseline classifier and return metrics."""

    if config is None:
        config = TrainConfig()

    logger.info("=" * 60)
    logger.info("BASELINE MODEL TRAINING (Logistic Regression)")
    logger.info("=" * 60)
    logger.info(f"Dataset size: {len(rows)} picks")

    features, labels, encoder = _encode_dataset(rows)
    logger.info(f"Feature dimension: {features.shape[1]}")
    logger.info(f"Unique labels (cards): {len(encoder.classes_)}")

    x_train, x_val, y_train, y_val = train_test_split(
        features, labels, test_size=config.test_size, random_state=config.random_state
    )
    logger.info(f"Train/validation split: {len(x_train)}/{len(x_val)} ({config.test_size:.0%} validation)")
    logger.info(f"Hyperparameters: max_iter={config.max_iter}, C={config.C}")

    logger.info("Training logistic regression model...")
    clf = LogisticRegression(max_iter=config.max_iter, C=config.C, n_jobs=-1)
    clf.fit(x_train, y_train)

    accuracy = float(clf.score(x_val, y_val))
    logger.info(f"Validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    metrics = EvaluationMetrics(
        accuracy=accuracy,
        train_samples=int(len(x_train)),
        validation_samples=int(len(x_val)),
    )
    artifacts = TrainingArtifacts(model=clf, label_encoder=encoder)
    logger.info("Training complete!")
    return TrainResult(model=DraftModel(artifacts), metrics=metrics)


def batch_predict(model: DraftModel, features: Iterable[np.ndarray]) -> List[str]:
    return [model.predict(feature) for feature in features]
