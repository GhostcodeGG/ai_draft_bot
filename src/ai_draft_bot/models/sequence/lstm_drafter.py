"""LSTM-based sequence model for draft pick prediction.

This model learns from entire draft sequences using LSTM layers to capture
temporal dependencies and draft progression patterns.

Key features:
- Learns "I'm building aggro" patterns from pick history
- Attention mechanism to focus on important picks
- Handles variable-length draft sequences
- Predicts best pick given pack and draft history
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

from ai_draft_bot.data.ingest_17l import PickRecord
from ai_draft_bot.models.sequence.embeddings import CardEmbedding, build_card_vocabulary
from ai_draft_bot.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LSTMDrafterConfig:
    """Configuration for LSTM draft model."""

    # Architecture
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    use_attention: bool = True

    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    early_stopping_patience: int = 10

    # Data
    max_sequence_length: int = 45  # Maximum picks in a draft
    test_size: float = 0.2
    random_state: int = 13


class AttentionLayer(nn.Module):
    """Attention mechanism to weight important picks in draft history."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self, lstm_output: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and weighted sum.

        Args:
            lstm_output: LSTM output (batch_size, seq_len, hidden_dim)
            mask: Optional mask for padding (batch_size, seq_len)

        Returns:
            (context_vector, attention_weights)
        """
        # Compute attention scores
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get weights
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(
            weights.unsqueeze(1), lstm_output
        ).squeeze(1)  # (batch, hidden_dim)

        return context, weights


class LSTMDraftNetwork(nn.Module):
    """LSTM network for draft pick prediction.

    Architecture:
    1. Card embeddings (learned dense vectors)
    2. LSTM layers (capture draft progression)
    3. Attention mechanism (focus on important picks)
    4. Output layer (predict best card from pack)
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        config: LSTMDrafterConfig,
    ):
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        # Card embedding layer
        self.embedding = CardEmbedding(
            num_cards=vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0,
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        # Attention layer
        lstm_output_dim = (
            config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        )
        if config.use_attention:
            self.attention = AttentionLayer(lstm_output_dim)

        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

        logger.info(f"Initialized LSTM network:")
        logger.info(f"  Vocab size: {vocab_size}")
        logger.info(f"  Num classes: {num_classes}")
        logger.info(f"  Embedding dim: {config.embedding_dim}")
        logger.info(f"  Hidden dim: {config.hidden_dim}")
        logger.info(f"  Num layers: {config.num_layers}")
        logger.info(f"  Bidirectional: {config.bidirectional}")
        logger.info(f"  Attention: {config.use_attention}")

    def forward(
        self,
        picked_cards: torch.Tensor,
        pack_cards: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            picked_cards: Cards picked so far (batch, seq_len)
            pack_cards: Cards in current pack (batch, pack_size)
            lengths: Actual sequence lengths (batch,) for padding

        Returns:
            Logits for each card in pack (batch, pack_size)
        """
        batch_size, seq_len = picked_cards.shape
        _, pack_size = pack_cards.shape

        # Embed picked cards
        embedded = self.embedding(picked_cards)  # (batch, seq_len, embed_dim)

        # Pass through LSTM
        if lengths is not None:
            # Pack padded sequences for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)

        # Get draft state representation
        if self.config.use_attention:
            # Use attention to weight important picks
            mask = None
            if lengths is not None:
                # Create mask for padding
                mask = torch.arange(seq_len, device=picked_cards.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
            draft_state, attn_weights = self.attention(lstm_out, mask)
        else:
            # Use last hidden state
            if self.config.bidirectional:
                # Concatenate forward and backward
                draft_state = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                draft_state = hidden[-1]

        # Embed pack cards
        pack_embedded = self.embedding(pack_cards)  # (batch, pack_size, embed_dim)

        # Expand draft state to compare with each pack card
        draft_state_expanded = draft_state.unsqueeze(1).expand(
            batch_size, pack_size, -1
        )  # (batch, pack_size, hidden_dim)

        # Concatenate draft state with each pack card
        combined = torch.cat(
            [pack_embedded, draft_state_expanded], dim=2
        )  # (batch, pack_size, embed_dim + hidden_dim)

        # Score each card in pack
        # Simple approach: project to hidden dim then to logit
        combined_features = self.dropout(combined)
        logits = self.fc(
            combined_features.view(batch_size * pack_size, -1)
        )  # (batch*pack_size, num_classes)
        logits = logits.view(batch_size, pack_size, -1)  # (batch, pack_size, num_classes)

        # For each pack, we want to select which card index is best
        # Take max over class dimension to get pack card scores
        pack_scores, _ = torch.max(logits, dim=2)  # (batch, pack_size)

        return pack_scores

    def predict(
        self, picked_cards: list[str], pack_cards: list[str], encoder: LabelEncoder
    ) -> str:
        """Predict best card from pack given draft history.

        Args:
            picked_cards: List of card names picked so far
            pack_cards: List of card names in current pack
            encoder: Card name encoder

        Returns:
            Predicted card name
        """
        self.eval()
        with torch.no_grad():
            # Encode cards
            picked_indices = [
                encoder.transform([card])[0] if card in encoder.classes_ else 1
                for card in picked_cards
            ]
            pack_indices = [
                encoder.transform([card])[0] if card in encoder.classes_ else 1
                for card in pack_cards
            ]

            # Convert to tensors
            picked_tensor = torch.LongTensor(picked_indices).unsqueeze(0)
            pack_tensor = torch.LongTensor(pack_indices).unsqueeze(0)
            lengths = torch.LongTensor([len(picked_indices)])

            # Get scores
            scores = self.forward(picked_tensor, pack_tensor, lengths)

            # Best card
            best_idx = torch.argmax(scores[0]).item()
            return pack_cards[best_idx]

    def save(self, path: Path) -> None:
        """Save model to disk."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
                "vocab_size": self.vocab_size,
                "num_classes": self.num_classes,
            },
            path,
        )
        logger.info(f"Saved LSTM model to {path}")

    @classmethod
    def load(cls, path: Path) -> LSTMDraftNetwork:
        """Load model from disk."""
        checkpoint = torch.load(path)
        model = cls(
            vocab_size=checkpoint["vocab_size"],
            num_classes=checkpoint["num_classes"],
            config=checkpoint["config"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded LSTM model from {path}")
        return model


def prepare_sequence_data(
    picks: Sequence[PickRecord], encoder: LabelEncoder, max_length: int = 45
) -> list[dict]:
    """Prepare draft sequences for LSTM training.

    Args:
        picks: All pick records
        encoder: Card name encoder
        max_length: Maximum sequence length (pad/truncate)

    Returns:
        List of training examples, each containing:
        - picked_so_far: Cards picked before this pick
        - pack_cards: Cards available in pack
        - label: Index of chosen card in pack
    """
    from ai_draft_bot.data.ingest_17l import group_picks_by_event

    grouped = group_picks_by_event(picks)
    examples = []

    for event_id, event_picks in grouped.items():
        picked_so_far = []

        for pick in event_picks:
            # Encode pack cards
            pack_cards = [
                int(encoder.transform([card])[0])
                if card in encoder.classes_
                else 1  # <UNK>
                for card in pick.pack_contents
            ]

            # Find label (index of chosen card in pack)
            try:
                label_idx = pick.pack_contents.index(pick.chosen_card)
            except ValueError:
                # Chosen card not in pack (data error)
                continue

            # Add example
            examples.append(
                {
                    "picked_so_far": picked_so_far.copy(),
                    "pack_cards": pack_cards,
                    "label": label_idx,
                    "event_id": event_id,
                }
            )

            # Update picked cards
            picked_idx = int(encoder.transform([pick.chosen_card])[0])
            picked_so_far.append(picked_idx)

    logger.info(f"Prepared {len(examples)} sequence examples from {len(grouped)} drafts")
    return examples
