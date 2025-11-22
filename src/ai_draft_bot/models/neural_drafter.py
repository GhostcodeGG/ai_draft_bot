"""Neural network models for draft pick prediction using PyTorch.

This module provides deep learning models that can capture complex
non-linear interactions between features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from ai_draft_bot.features.draft_context import PickFeatures
from ai_draft_bot.utils.metrics import ndcg_at_ks, topk_accuracies
from ai_draft_bot.utils.splits import train_val_split_by_event

logger = logging.getLogger("ai_draft_bot.models.neural_drafter")


class DraftPickNetwork(nn.Module):
    """Deep neural network for draft pick prediction.

    Architecture:
    - Input layer (feature_dim)
    - Hidden layers with dropout and batch normalization
    - Output layer (num_classes)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3,
    ):
        """Initialize the network.

        Args:
            feature_dim: Number of input features
            num_classes: Number of output classes (card names)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        if hidden_dims is None:
            # Default architecture scales with input size
            hidden_dims = [feature_dim * 2, feature_dim, feature_dim // 2]

        layers = []
        prev_dim = feature_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


@dataclass
class NeuralTrainConfig:
    """Configuration for neural network training.

    Attributes:
        test_size: Validation split fraction
        random_state: Random seed
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout probability
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        num_epochs: Number of training epochs
        early_stopping_patience: Epochs to wait for improvement
        use_gpu: Whether to use GPU if available
    """

    test_size: float = 0.2
    random_state: int = 13
    hidden_dims: list[int] | None = None
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 128
    num_epochs: int = 100
    early_stopping_patience: int = 10
    use_gpu: bool = False


@dataclass
class NeuralTrainingArtifacts:
    """Training artifacts for neural models."""

    model: nn.Module
    label_encoder: LabelEncoder
    feature_dim: int
    num_classes: int


@dataclass
class NeuralEvaluationMetrics:
    """Evaluation metrics for neural models."""

    accuracy: float
    train_samples: int
    validation_samples: int
    best_epoch: int
    final_train_loss: float
    final_val_loss: float
    top_k: Mapping[int, float] | None = None
    ndcg: Mapping[int, float] | None = None


class NeuralDraftModel:
    """Neural network draft model."""

    def __init__(self, artifacts: NeuralTrainingArtifacts):
        self.artifacts = artifacts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.artifacts.model.to(self.device)
        self.artifacts.model.eval()

    def predict(self, features: np.ndarray) -> str:
        """Predict the most likely card pick.

        Args:
            features: Feature vector for a single pick

        Returns:
            Card name prediction
        """
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.artifacts.model(features_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()

        return str(self.artifacts.label_encoder.inverse_transform([pred_idx])[0])

    def predict_proba(self, features: np.ndarray) -> Mapping[str, float]:
        """Predict probability distribution over all cards.

        Args:
            features: Feature vector for a single pick

        Returns:
            Dictionary mapping card names to probabilities
        """
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.artifacts.model(features_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        labels = self.artifacts.label_encoder.inverse_transform(
            np.arange(len(probs))
        )
        return {str(label): float(prob) for label, prob in zip(labels, probs)}

    def save(self, path: Path | str) -> None:
        """Save model to disk."""
        payload = {
            "model_state_dict": self.artifacts.model.state_dict(),
            "label_encoder": self.artifacts.label_encoder,
            "feature_dim": self.artifacts.feature_dim,
            "num_classes": self.artifacts.num_classes,
            "hidden_dims": [
                layer.out_features
                for layer in self.artifacts.model.network
                if isinstance(layer, nn.Linear)
            ][:-1],  # Exclude output layer
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "NeuralDraftModel":
        """Load model from disk."""
        payload = joblib.load(path)

        # Reconstruct model
        model = DraftPickNetwork(
            feature_dim=payload["feature_dim"],
            num_classes=payload["num_classes"],
            hidden_dims=payload.get("hidden_dims"),
        )
        model.load_state_dict(payload["model_state_dict"])

        artifacts = NeuralTrainingArtifacts(
            model=model,
            label_encoder=payload["label_encoder"],
            feature_dim=payload["feature_dim"],
            num_classes=payload["num_classes"],
        )

        return cls(artifacts)


@dataclass
class NeuralTrainResult:
    """Result of neural model training."""

    model: NeuralDraftModel
    metrics: NeuralEvaluationMetrics


def _encode_dataset(
    rows: Sequence[PickFeatures],
    encoder: LabelEncoder | None = None,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Encode dataset into feature matrix and label vector."""
    features = np.vstack([row.features for row in rows])
    labels = [row.label for row in rows]
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    return features, encoded_labels, encoder


def train_neural_model(
    rows: Sequence[PickFeatures], *, config: NeuralTrainConfig | None = None
) -> NeuralTrainResult:
    """Train a neural network draft model.

    Args:
        rows: Training data (PickFeatures with labels)
        config: Training configuration

    Returns:
        NeuralTrainResult with trained model and metrics
    """
    if config is None:
        config = NeuralTrainConfig()

    logger.info("=" * 60)
    logger.info("NEURAL NETWORK TRAINING")
    logger.info("=" * 60)
    logger.info(f"Dataset size: {len(rows)} picks")
    if not rows:
        raise ValueError("No rows provided for training")

    # Draft-aware split
    split = train_val_split_by_event(rows, test_size=config.test_size, random_state=config.random_state)
    train_rows, val_rows = split.train, split.val
    if not val_rows:
        raise ValueError("Validation split is empty. Provide drafts from more than one event.")

    # Encode with shared encoder (fit on all labels to avoid unseen classes)
    _, _, encoder = _encode_dataset(rows)
    x_train, y_train, _ = _encode_dataset(train_rows, encoder)
    x_val, y_val, _ = _encode_dataset(val_rows, encoder)

    logger.info(f"Feature dimension: {x_train.shape[1]}")
    logger.info(f"Unique labels (cards): {len(encoder.classes_)}")
    logger.info(
        f"Train/validation split (by draft): {len(x_train)}/{len(x_val)} "
        f"({config.test_size:.0%} validation)"
    )

    # Convert to PyTorch tensors
    device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    train_dataset = TensorDataset(
        torch.FloatTensor(x_train), torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(x_val), torch.LongTensor(y_val)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    model = DraftPickNetwork(
        feature_dim=features.shape[1],
        num_classes=len(encoder.classes_),
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate,
    )
    model.to(device)

    logger.info(f"Model architecture: {model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Learning rate scheduling
        scheduler.step(val_accuracy)

        # Logging
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{config.num_epochs}] "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info(
        f"Training complete! Best validation accuracy: {best_val_accuracy:.4f} "
        f"at epoch {best_epoch}"
    )

    # Final evaluation on validation split
    model.eval()
    with torch.no_grad():
        val_outputs = model(torch.FloatTensor(x_val).to(device))
        val_probs = torch.softmax(val_outputs, dim=1).cpu().numpy()
        val_pred = np.argmax(val_probs, axis=1)
        final_val_accuracy = float(np.mean(val_pred == y_val))

    top_k = topk_accuracies(val_probs, y_val, ks=(1, 3, 5))
    ndcg = ndcg_at_ks(val_probs, y_val, ks=(1, 3, 5))

    logger.info(f"Final validation accuracy: {final_val_accuracy:.4f}")
    logger.info(f"Top-k accuracy: { {k: round(v, 3) for k, v in top_k.items()} }")
    logger.info(f"NDCG: { {k: round(v, 3) for k, v in ndcg.items()} }")

    # Create artifacts
    artifacts = NeuralTrainingArtifacts(
        model=model,
        label_encoder=encoder,
        feature_dim=x_train.shape[1],
        num_classes=len(encoder.classes_),
    )

    # Build metrics
    metrics = NeuralEvaluationMetrics(
        accuracy=final_val_accuracy,
        train_samples=len(x_train),
        validation_samples=len(x_val),
        best_epoch=best_epoch,
        final_train_loss=avg_train_loss,
        final_val_loss=avg_val_loss,
        top_k=top_k,
        ndcg=ndcg,
    )

    return NeuralTrainResult(model=NeuralDraftModel(artifacts), metrics=metrics)
