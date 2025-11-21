"""Opponent modeling and table dynamics.

This module infers what other drafters are doing based on pack contents
and adjusts pick strategy accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ai_draft_bot.data.ingest_17l import CardMetadata


@dataclass
class OpponentModel:
    """Model of what opponent drafters are likely drafting.

    Tracks colors being drafted by upstream/downstream opponents.

    Attributes:
        upstream_colors: Colors being drafted by upstream opponents (5 values for WUBRG)
        downstream_colors: Colors being drafted by downstream opponents
        color_competition: How much competition for each color (0.0-1.0)
        pivot_opportunity: Opportunity to pivot into open colors (0.0-1.0)
    """

    upstream_colors: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
    )
    downstream_colors: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
    )
    color_competition: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
    )
    pivot_opportunity: float = 0.0


def infer_upstream_colors(
    pack_cards: Sequence[CardMetadata],
    expected_pack_size: int = 14,
    pick_number: int = 1,
) -> np.ndarray:
    """Infer which colors are being drafted by upstream opponents.

    Colors missing from pack (especially good cards) indicate upstream drafting.

    Args:
        pack_cards: Cards remaining in pack
        expected_pack_size: Expected initial pack size
        pick_number: Current pick number

    Returns:
        5-element array for WUBRG upstream draft intensity
        (0.0 = not being drafted, 1.0 = heavily drafted)
    """
    colors = ["W", "U", "B", "R", "G"]
    upstream_intensity = np.zeros(5, dtype=float)

    # How many cards have been removed from pack?
    cards_removed = expected_pack_size - len(pack_cards)

    if cards_removed == 0:
        return np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)  # First pick

    # For each color, check quality of remaining cards
    # If high-quality cards are missing, color is being drafted upstream
    for i, color in enumerate(colors):
        color_cards = [c for c in pack_cards if color in c.color]

        if not color_cards:
            # No cards in this color = heavily drafted
            upstream_intensity[i] = 0.9
            continue

        # Check quality of remaining color cards
        avg_quality = np.mean([c.gih_wr if c.gih_wr is not None else 0.5 for c in color_cards])

        # Low quality remaining = good cards taken = being drafted
        if avg_quality < 0.50:
            upstream_intensity[i] = 0.7
        elif avg_quality < 0.53:
            upstream_intensity[i] = 0.6
        else:
            # High quality remaining = open color
            upstream_intensity[i] = 0.3

    return upstream_intensity


def infer_downstream_colors(
    wheeled_cards: Sequence[CardMetadata],
    pick_number: int,
) -> np.ndarray:
    """Infer which colors are being drafted by downstream opponents.

    Colors that don't wheel back (in pack 2/3) indicate downstream drafting.

    Args:
        wheeled_cards: Cards that wheeled back to us
        pick_number: Current pick number

    Returns:
        5-element array for WUBRG downstream draft intensity
    """
    downstream_intensity = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)

    # This requires tracking cards across packs (complex)
    # For now, return neutral values
    # TODO: Implement full wheel tracking across pack 1/2/3

    return downstream_intensity


def compute_color_competition(
    upstream_colors: np.ndarray,
    downstream_colors: np.ndarray,
    our_colors: np.ndarray,
) -> np.ndarray:
    """Compute competition level for each color.

    High competition = both we and opponents want this color

    Args:
        upstream_colors: Upstream drafting intensity
        downstream_colors: Downstream drafting intensity
        our_colors: Our color commitment (0.0-1.0 for each color)

    Returns:
        5-element array for competition level per color
    """
    # Competition is high when:
    # 1. We want the color (high our_colors value)
    # 2. Opponents want it too (high upstream/downstream)

    total_opponent_intensity = (upstream_colors + downstream_colors) / 2.0

    # Element-wise multiplication: high when both we and opponents want it
    competition = our_colors * total_opponent_intensity

    return competition


def compute_pivot_opportunity(
    upstream_colors: np.ndarray,
    our_colors: np.ndarray,
) -> float:
    """Compute opportunity score for pivoting into open colors.

    High pivot opportunity when:
    - We're not heavily committed yet
    - Some colors are clearly open

    Args:
        upstream_colors: Upstream drafting intensity
        our_colors: Our current color commitment

    Returns:
        Pivot opportunity score (0.0-1.0)
    """
    # How committed are we? (sum of our color commitments)
    our_commitment = np.sum(our_colors)

    # How open is the most open color?
    most_open_color = np.min(upstream_colors)  # Low value = open

    # Early in draft + open colors = high pivot opportunity
    flexibility = 1.0 - (our_commitment / 2.0)  # Normalize to [0, 1]
    openness = 1.0 - most_open_color

    pivot_score = (flexibility + openness) / 2.0

    return max(0.0, min(1.0, pivot_score))


def build_opponent_model(
    pack_cards: Sequence[CardMetadata],
    our_picked_cards: Sequence[str],
    metadata: Mapping[str, CardMetadata],
    pick_number: int,
    pack_number: int,
) -> OpponentModel:
    """Build opponent model for current draft state.

    Args:
        pack_cards: Cards in current pack
        our_picked_cards: Cards we've picked so far
        metadata: Card metadata lookup
        pick_number: Current pick number
        pack_number: Current pack number

    Returns:
        OpponentModel with inferred opponent strategies
    """
    # Infer upstream drafting
    upstream = infer_upstream_colors(pack_cards, expected_pack_size=14, pick_number=pick_number)

    # Downstream inference requires wheel tracking (simplified for now)
    downstream = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)

    # Compute our color commitment
    colors = ["W", "U", "B", "R", "G"]
    our_colors = np.zeros(5, dtype=float)

    for i, color in enumerate(colors):
        color_count = sum(
            1
            for card_name in our_picked_cards
            if card_name in metadata and color in metadata[card_name].color
        )
        # Normalize by total cards picked (up to 10 for normalization)
        our_colors[i] = min(color_count / 10.0, 1.0)

    # Compute competition and pivot opportunity
    competition = compute_color_competition(upstream, downstream, our_colors)
    pivot_opp = compute_pivot_opportunity(upstream, our_colors)

    return OpponentModel(
        upstream_colors=upstream,
        downstream_colors=downstream,
        color_competition=competition,
        pivot_opportunity=pivot_opp,
    )


def opponent_model_to_vector(model: OpponentModel) -> np.ndarray:
    """Convert OpponentModel to numeric feature vector.

    Returns:
        16-element array: [upstream_colors (5), downstream_colors (5),
                          color_competition (5), pivot_opportunity (1)]
    """
    return np.concatenate(
        [
            model.upstream_colors,
            model.downstream_colors,
            model.color_competition,
            [model.pivot_opportunity],
        ]
    )
