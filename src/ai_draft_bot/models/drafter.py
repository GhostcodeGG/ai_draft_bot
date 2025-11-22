"""Baseline draft model built on scikit-learn."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from ai_draft_bot.features.draft_context import PickFeatures
from ai_draft_bot.utils.metrics import ndcg_at_ks, topk_accuracies
from ai_draft_bot.utils.splits import train_val_split_by_event

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
    top_k: Mapping[int, float] | None = None
    ndcg: Mapping[int, float] | None = None


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


def _fit_label_encoder(rows: Sequence[PickFeatures]) -> LabelEncoder:
    """Fit a label encoder on the provided rows."""
    encoder = LabelEncoder()
    encoder.fit([row.label for row in rows])
    return encoder


def _encode_with_encoder(
    rows: Sequence[PickFeatures], encoder: LabelEncoder, *, drop_unknown: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode rows with an existing encoder, optionally dropping unseen labels."""
    filtered_rows: list[PickFeatures] = []
    for row in rows:
        if row.label in encoder.classes_:
            filtered_rows.append(row)
        elif not drop_unknown:
            filtered_rows.append(row)

    if not filtered_rows:
        return np.empty((0, 0)), np.array([], dtype=int)

    features = np.vstack([row.features for row in filtered_rows])
    labels = encoder.transform([row.label for row in filtered_rows])
    return features, labels


def train_model(rows: Sequence[PickFeatures], *, config: TrainConfig | None = None) -> TrainResult:
    """Train the baseline classifier and return metrics."""

    if config is None:
        config = TrainConfig()

    logger.info("=" * 60)
    logger.info("BASELINE MODEL TRAINING (Logistic Regression)")
    logger.info("=" * 60)
    logger.info(f"Dataset size: {len(rows)} picks")
    if not rows:
        raise ValueError("No rows provided for training")

    # Draft-aware split to avoid leakage
    split = train_val_split_by_event(rows, test_size=config.test_size, random_state=config.random_state)
    train_rows, val_rows = split.train, split.val
    if not val_rows:
        raise ValueError("Validation split is empty. Provide drafts from more than one event.")
    logger.info(
        f"Train/validation split (by draft): {len(train_rows)}/{len(val_rows)} "
        f"({config.test_size:.0%} validation)"
    )

    encoder = _fit_label_encoder(train_rows)
    x_train, y_train = _encode_with_encoder(train_rows, encoder)
    x_val, y_val = _encode_with_encoder(val_rows, encoder, drop_unknown=True)

    # Drop validation rows with unseen labels to avoid class mismatch
    if x_val.size == 0 or y_val.size == 0:
        raise ValueError("Validation split contained only unseen labels. Provide more data.")

    logger.info(f"Feature dimension: {x_train.shape[1]}")
    logger.info(f"Unique labels (cards): {len(encoder.classes_)}")
    logger.info(f"Hyperparameters: max_iter={config.max_iter}, C={config.C}")

    logger.info("Training logistic regression model...")
    clf = LogisticRegression(max_iter=config.max_iter, C=config.C, n_jobs=-1)
    clf.fit(x_train, y_train)

    accuracy = float(clf.score(x_val, y_val))
    probs = clf.predict_proba(x_val)
    top_k = topk_accuracies(probs, y_val, ks=(1, 3, 5))
    ndcg = ndcg_at_ks(probs, y_val, ks=(1, 3, 5))

    logger.info(f"Validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Top-k accuracy: { {k: round(v, 3) for k, v in top_k.items()} }")
    logger.info(f"NDCG: { {k: round(v, 3) for k, v in ndcg.items()} }")

    metrics = EvaluationMetrics(
        accuracy=accuracy,
        train_samples=int(len(x_train)),
        validation_samples=int(len(x_val)),
        top_k=top_k,
        ndcg=ndcg,
    )
    artifacts = TrainingArtifacts(model=clf, label_encoder=encoder)
    logger.info("Training complete!")
    return TrainResult(model=DraftModel(artifacts), metrics=metrics)


def batch_predict(model: DraftModel, features: Iterable[np.ndarray]) -> List[str]:
    return [model.predict(feature) for feature in features]
