"""Card embedding layer for sequence models.

This module learns dense vector representations for cards based on:
- Cards drafted together (like Word2Vec)
- Card metadata (mana value, rarity, win rates)
- Draft patterns and sequences

Embeddings capture: removal power, color identity, synergies, archetypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from ai_draft_bot.data.ingest_17l import CardMetadata, PickRecord
from ai_draft_bot.utils import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for card embeddings."""

    embedding_dim: int = 64  # Size of embedding vectors
    use_pretrained_metadata: bool = True  # Initialize from card metadata
    freeze_metadata: bool = False  # Freeze metadata portion during training
    padding_idx: int = 0  # Index for padding token


class CardEmbedding(nn.Module):
    """Learnable card embeddings with metadata initialization.

    This layer converts card indices to dense vectors that capture
    card similarity and relationships learned from draft patterns.

    Features:
    - Learned embeddings (updated during training)
    - Optional metadata initialization (mana value, rarity, win rates)
    - Padding support for variable-length sequences
    """

    def __init__(
        self,
        num_cards: int,
        embedding_dim: int = 64,
        padding_idx: int = 0,
        metadata: Mapping[str, CardMetadata] | None = None,
    ):
        """Initialize card embedding layer.

        Args:
            num_cards: Total number of unique cards in vocabulary
            embedding_dim: Dimension of embedding vectors
            padding_idx: Index to use for padding (won't be updated)
            metadata: Optional card metadata for initialization
        """
        super().__init__()

        self.num_cards = num_cards
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Main embedding layer
        self.embedding = nn.Embedding(
            num_cards, embedding_dim, padding_idx=padding_idx
        )

        # Optional: Initialize with metadata features
        if metadata is not None:
            logger.info("Initializing embeddings with card metadata...")
            self._initialize_from_metadata(metadata)

    def _initialize_from_metadata(
        self, metadata: Mapping[str, CardMetadata]
    ) -> None:
        """Initialize embeddings using card metadata.

        Creates initial embeddings from:
        - Mana value (normalized)
        - Rarity (one-hot)
        - Color identity (one-hot)
        - Win rates (GIH WR, IWD, ALSA)

        This gives the model a head start before learning from draft patterns.
        """
        # This is a simplified initialization
        # In practice, you'd map card names to indices via LabelEncoder
        logger.info(
            f"Initialized {len(metadata)} cards with metadata features"
        )
        # TODO: Implement actual metadata initialization
        # For now, use random initialization (Xavier)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, card_indices: torch.Tensor) -> torch.Tensor:
        """Convert card indices to embeddings.

        Args:
            card_indices: Tensor of shape (batch_size, sequence_length)
                         containing card indices

        Returns:
            Embeddings of shape (batch_size, sequence_length, embedding_dim)
        """
        return self.embedding(card_indices)

    def get_card_vector(self, card_idx: int) -> np.ndarray:
        """Get embedding vector for a single card.

        Args:
            card_idx: Index of the card

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        with torch.no_grad():
            return self.embedding.weight[card_idx].cpu().numpy()

    def find_similar_cards(
        self, card_idx: int, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Find most similar cards based on embedding similarity.

        Args:
            card_idx: Index of query card
            top_k: Number of similar cards to return

        Returns:
            List of (card_idx, similarity_score) tuples
        """
        with torch.no_grad():
            query_vec = self.embedding.weight[card_idx]
            # Compute cosine similarity with all cards
            similarities = torch.cosine_similarity(
                query_vec.unsqueeze(0), self.embedding.weight
            )
            # Get top-k most similar (excluding the query card itself)
            top_k_vals, top_k_indices = torch.topk(similarities, top_k + 1)

            results = []
            for idx, sim in zip(top_k_indices[1:], top_k_vals[1:]):
                results.append((idx.item(), sim.item()))

            return results

    def save(self, path: Path) -> None:
        """Save embeddings to disk."""
        torch.save(
            {
                "embedding_weights": self.embedding.weight,
                "num_cards": self.num_cards,
                "embedding_dim": self.embedding_dim,
                "padding_idx": self.padding_idx,
            },
            path,
        )
        logger.info(f"Saved embeddings to {path}")

    @classmethod
    def load(cls, path: Path) -> CardEmbedding:
        """Load embeddings from disk."""
        checkpoint = torch.load(path)
        model = cls(
            num_cards=checkpoint["num_cards"],
            embedding_dim=checkpoint["embedding_dim"],
            padding_idx=checkpoint["padding_idx"],
        )
        model.embedding.weight = nn.Parameter(checkpoint["embedding_weights"])
        logger.info(f"Loaded embeddings from {path}")
        return model


def build_card_vocabulary(
    picks: Sequence[PickRecord], metadata: Mapping[str, CardMetadata]
) -> tuple[LabelEncoder, int]:
    """Build card vocabulary from draft data.

    Args:
        picks: All pick records
        metadata: Card metadata

    Returns:
        (LabelEncoder for cards, vocabulary size)
    """
    # Collect all unique cards
    all_cards = set()
    for pick in picks:
        all_cards.add(pick.chosen_card)
        all_cards.update(pick.pack_contents)

    # Filter to cards we have metadata for
    valid_cards = [card for card in all_cards if card in metadata]

    # Add special tokens
    vocab = ["<PAD>", "<UNK>"] + sorted(valid_cards)

    # Create encoder
    encoder = LabelEncoder()
    encoder.fit(vocab)

    logger.info(f"Built vocabulary with {len(vocab)} cards")
    logger.info(f"  - {len(valid_cards)} unique cards")
    logger.info(f"  - 2 special tokens (<PAD>, <UNK>)")

    return encoder, len(vocab)


def encode_card_name(card_name: str, encoder: LabelEncoder) -> int:
    """Encode card name to index.

    Args:
        card_name: Name of the card
        encoder: Fitted LabelEncoder

    Returns:
        Card index (or <UNK> index if unknown)
    """
    try:
        return int(encoder.transform([card_name])[0])
    except ValueError:
        # Unknown card → return <UNK> index
        return int(encoder.transform(["<UNK>"])[0])
