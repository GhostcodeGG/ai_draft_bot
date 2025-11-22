"""Positional and signaling features for draft context.

This module provides features that capture draft-specific signals:
- Wheeling probability (will this card come back?)
- Color openness signals (what colors are flowing?)
- Pack quality analysis (is this pack strong or weak?)
- Table position dynamics (early vs late picks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ai_draft_bot.data.ingest_17l import CardMetadata, PickRecord


@dataclass
class PositionalFeatures:
    """Positional features for a single pick.

    Attributes:
        wheeling_probability: Likelihood card will wheel back (0.0-1.0)
        pack_quality: Overall pack strength relative to average (0.0-1.0)
        color_openness: How open each color appears to be (5 values for WUBRG)
        pick_stage: What stage of draft (early/mid/late) as 0.0-1.0
        pack_stage: What pack we're in (1/2/3) as 0.0-1.0
        rare_in_pack: Whether pack contains rare/mythic
        bomb_in_pack: Whether pack contains bomb (high WR rare)
        color_signal_strength: How strong the color signals are (0.0-1.0)
    """

    wheeling_probability: float = 0.0
    pack_quality: float = 0.5
    color_openness: np.ndarray = None  # type: ignore # 5 values for WUBRG
    pick_stage: float = 0.0
    pack_stage: float = 0.0
    rare_in_pack: bool = False
    bomb_in_pack: bool = False
    color_signal_strength: float = 0.0

    def __post_init__(self) -> None:
        if self.color_openness is None:
            self.color_openness = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)


def compute_wheeling_probability(
    card: CardMetadata,
    pick_number: int,
    pack_number: int,
) -> float:
    """Estimate probability that a card will wheel back to us.

    Uses ALSA (Average Last Seen At) if available, otherwise uses heuristics.

    Args:
        card: Card being evaluated
        pick_number: Current pick number (1-14 in typical draft)
        pack_number: Current pack number (1-3)

    Returns:
        Probability (0.0-1.0) that card will wheel
    """
    # If we have ALSA data, use it
    if card.alsa is not None:
        # ALSA is the average pick number when card is last seen
        # If ALSA > pick_number + pack_size/2, likely to wheel
        expected_return_pick = pick_number + (15 - pick_number)

        if card.alsa >= expected_return_pick:
            return 0.8  # High chance of wheeling
        elif card.alsa >= expected_return_pick - 2:
            return 0.5  # Medium chance
        else:
            return 0.2  # Low chance

    # Fallback heuristics if no ALSA
    # High pick = low wheel chance, late pick = high wheel chance
    pick_progress = pick_number / 14.0

    # Rarity matters - commons wheel more than rares
    rarity_factor = 1.0
    if card.rarity.lower() == "rare":
        rarity_factor = 0.3
    elif card.rarity.lower() == "mythic":
        rarity_factor = 0.1
    elif card.rarity.lower() == "uncommon":
        rarity_factor = 0.6

    # Win rate matters - high WR cards don't wheel
    wr_factor = 1.0
    if card.gih_wr is not None:
        if card.gih_wr > 0.60:
            wr_factor = 0.2  # Bombs don't wheel
        elif card.gih_wr > 0.55:
            wr_factor = 0.5
        elif card.gih_wr < 0.50:
            wr_factor = 1.2  # Bad cards wheel more

    wheel_prob = (1.0 - pick_progress) * rarity_factor * wr_factor
    return max(0.0, min(1.0, wheel_prob))


def compute_pack_quality(
    pack_cards: Sequence[CardMetadata],
    set_average_gih: float = 0.52,
) -> float:
    """Compute overall pack quality relative to average.

    Args:
        pack_cards: Cards in the current pack
        set_average_gih: Average GIH WR for the set (default ~52%)

    Returns:
        Pack quality score (0.0-1.0), where 0.5 is average
    """
    if not pack_cards:
        return 0.5

    # Compute average GIH WR of pack
    gih_values = [c.gih_wr for c in pack_cards if c.gih_wr is not None]

    if not gih_values:
        return 0.5  # No data, assume average

    avg_gih = np.mean(gih_values)

    # Normalize around set average
    # If pack is 5% above average, that's a strong pack
    quality = 0.5 + (avg_gih - set_average_gih) * 10

    return max(0.0, min(1.0, quality))


def compute_color_openness(
    pack_cards: Sequence[CardMetadata],
    cards_picked_by_others: Sequence[str],
    metadata: Mapping[str, CardMetadata],
) -> np.ndarray:
    """Estimate which colors are open based on pack contents.

    Open colors = colors with more/better cards remaining in pack
    Cut colors = colors missing from pack (being drafted upstream)

    Args:
        pack_cards: Cards currently in pack
        cards_picked_by_others: Cards picked by other drafters (inferred from missing cards)
        metadata: Card metadata lookup

    Returns:
        5-element array for WUBRG openness (0.0 = cut, 1.0 = wide open)
    """
    colors = ["W", "U", "B", "R", "G"]
    openness = np.zeros(5, dtype=float)

    # Count high-quality cards in each color remaining in pack
    for i, color in enumerate(colors):
        color_cards = [c for c in pack_cards if color in c.color]

        if color_cards:
            # Weight by win rate
            color_quality = np.mean(
                [c.gih_wr if c.gih_wr is not None else 0.5 for c in color_cards]
            )
            color_count = len(color_cards)

            # More good cards in this color = more open
            openness[i] = (color_quality * 2.0 + color_count * 0.1) / 2.0
        else:
            # No cards in this color = being cut
            openness[i] = 0.2

    # Normalize to [0, 1]
    openness = np.clip(openness, 0.0, 1.0)

    return openness


def compute_color_signal_strength(
    current_pack_colors: np.ndarray,
    previous_pack_colors: np.ndarray | None,
) -> float:
    """Compute how strong/consistent the color signals are.

    Strong signals = same colors open across multiple picks
    Weak signals = colors shifting/unclear

    Args:
        current_pack_colors: Color openness for current pack (5 values)
        previous_pack_colors: Color openness from previous pack (5 values)

    Returns:
        Signal strength (0.0 = weak/noisy, 1.0 = strong/clear)
    """
    if previous_pack_colors is None:
        return 0.5  # First pick, no signal yet

    # Compute correlation between current and previous color signals
    correlation_matrix = np.corrcoef(current_pack_colors, previous_pack_colors)
    correlation = correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0.0
    if np.isnan(correlation):
        correlation = 0.0

    # High correlation = strong/consistent signals
    # Low/negative correlation = weak/shifting signals
    signal_strength = (correlation + 1.0) / 2.0  # Map [-1, 1] to [0, 1]

    return max(0.0, min(1.0, signal_strength))


def extract_positional_features(
    pick: PickRecord,
    pack_cards: Sequence[CardMetadata],
    chosen_card: CardMetadata,
    metadata: Mapping[str, CardMetadata],
    previous_pack_colors: np.ndarray | None = None,
) -> PositionalFeatures:
    """Extract all positional features for a pick.

    Args:
        pick: Pick record with context
        pack_cards: Cards in current pack
        chosen_card: Card being picked
        metadata: Card metadata lookup
        previous_pack_colors: Color openness from previous pick (for signal strength)

    Returns:
        PositionalFeatures object
    """
    # Wheeling probability
    wheel_prob = compute_wheeling_probability(
        chosen_card,
        pick.pick_number,
        pick.pack_number,
    )

    # Pack quality
    pack_quality = compute_pack_quality(pack_cards)

    # Color openness (simplified - we don't have full draft info)
    color_openness = compute_color_openness(pack_cards, [], metadata)

    # Color signal strength
    signal_strength = compute_color_signal_strength(color_openness, previous_pack_colors)

    # Pick stage (0.0 = early, 1.0 = late in pack)
    pick_stage = pick.pick_number / 14.0

    # Pack stage (0.0 = pack 1, 1.0 = pack 3)
    pack_stage = (pick.pack_number - 1) / 2.0

    # Rare/bomb detection
    rare_in_pack = any(c.rarity.lower() in ["rare", "mythic"] for c in pack_cards)
    bomb_in_pack = any(
        c.rarity.lower() in ["rare", "mythic"] and (c.gih_wr or 0.0) > 0.58
        for c in pack_cards
    )

    return PositionalFeatures(
        wheeling_probability=wheel_prob,
        pack_quality=pack_quality,
        color_openness=color_openness,
        pick_stage=pick_stage,
        pack_stage=pack_stage,
        rare_in_pack=rare_in_pack,
        bomb_in_pack=bomb_in_pack,
        color_signal_strength=signal_strength,
    )


def positional_features_to_vector(features: PositionalFeatures) -> np.ndarray:
    """Convert PositionalFeatures to numeric feature vector.

    Returns:
        13-element array: [wheeling_prob, pack_quality, color_openness (5),
                          pick_stage, pack_stage, rare_in_pack, bomb_in_pack,
                          color_signal_strength]
    """
    return np.concatenate(
        [
            [features.wheeling_probability],
            [features.pack_quality],
            features.color_openness,
            [features.pick_stage],
            [features.pack_stage],
            [float(features.rare_in_pack)],
            [float(features.bomb_in_pack)],
            [features.color_signal_strength],
        ]
    )
