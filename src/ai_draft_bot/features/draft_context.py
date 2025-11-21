"""Feature extraction and labeling utilities for draft picks.

This module provides both baseline (simple) and advanced feature extraction.
Advanced features include win rates, draft context, deck composition, and pack signals.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence, Set

import numpy as np

from ai_draft_bot.data.ingest_17l import CardMetadata, PickRecord, group_picks_by_event
from ai_draft_bot.features.draft_state import compute_deck_stats, deck_stats_to_vector
from ai_draft_bot.features.synergies import compute_synergy_features
from ai_draft_bot.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PickFeatures:
    """Numeric vector for a single pick plus the chosen label."""

    features: np.ndarray
    label: str


def encode_color(color: str) -> np.ndarray:
    """One-hot encode the card color string into five slots.

    We collapse multi-color cards into the "multicolor" bucket to keep the space small.
    """

    colors = ["W", "U", "B", "R", "G"]
    vector = np.zeros(len(colors) + 1, dtype=float)
    if color in colors:
        vector[colors.index(color)] = 1.0
    else:
        vector[-1] = 1.0
    return vector


def rarity_value(rarity: str) -> float:
    mapping = {"common": 0.0, "uncommon": 0.33, "rare": 0.66, "mythic": 1.0}
    return mapping.get(rarity.lower(), 0.0)


def card_to_vector(card: CardMetadata, *, include_winrates: bool = True) -> np.ndarray:
    """Convert a card to a feature vector.

    Args:
        card: Card metadata
        include_winrates: If True, include win rate features (GIH WR, IWD, etc.)

    Returns:
        Feature vector with:
        - Basic: [mana_value, rarity_score, color_encoding (6)]
        - With win rates: + [gih_wr, oh_wr, gd_wr, iwd, alsa]
        Total: 8 features (basic) or 13 features (with win rates)
    """
    rarity_score = rarity_value(card.rarity)
    color_vector = encode_color(card.color)
    basic_features = np.concatenate(([card.mana_value, rarity_score], color_vector))

    if not include_winrates:
        return basic_features

    # Add win rate features (use 0.0 as default if not available)
    winrate_features = np.array(
        [
            card.gih_wr if card.gih_wr is not None else 0.0,
            card.oh_wr if card.oh_wr is not None else 0.0,
            card.gd_wr if card.gd_wr is not None else 0.0,
            card.iwd if card.iwd is not None else 0.0,
            card.alsa if card.alsa is not None else 0.0,
        ],
        dtype=float,
    )
    return np.concatenate([basic_features, winrate_features])


def build_pick_features(
    picks: Sequence[PickRecord],
    metadata: Mapping[str, CardMetadata],
) -> List[PickFeatures]:
    """Generate feature vectors for each pick in the dataset.

    Features include:
    - Mana value and rarity of the chosen card
    - Mean mana value and rarity across the pack
    - Color distribution across the pack
    """

    feature_rows: List[PickFeatures] = []
    for pick in picks:
        pack_vectors: List[np.ndarray] = []
        for name in pick.pack_contents:
            card = metadata.get(name)
            if card:
                pack_vectors.append(card_to_vector(card))
        if not pack_vectors:
            continue

        pack_matrix = np.vstack(pack_vectors)
        pack_mean = pack_matrix.mean(axis=0)

        chosen_card = metadata.get(pick.chosen_card)
        if not chosen_card:
            # Skip if we cannot label the pick
            continue

        feature_vector = np.concatenate([card_to_vector(chosen_card), pack_mean])
        feature_rows.append(PickFeatures(features=feature_vector, label=pick.chosen_card))
    return feature_rows


def build_advanced_pick_features(
    picks: Sequence[PickRecord],
    metadata: Mapping[str, CardMetadata],
) -> List[PickFeatures]:
    """Generate ADVANCED feature vectors for each pick with full context.

    This is the sophisticated feature extractor that uses:
    - Win rate statistics (GIH WR, IWD, ALSA, etc.)
    - Draft context (pick/pack number, cards picked so far)
    - Deck composition stats (mana curve, color commitment, creature count)
    - Pack-level signals (max/min/std of win rates, color availability)

    Feature breakdown:
    - Chosen card features (13): basic card stats + win rates
    - Pack aggregate features (13): mean of pack
    - Pack max features (13): best card in pack signals
    - Pack statistics (5): std dev of mana/wr, pack size, bomb count, rare count
    - Contextual (2): pick_number, pack_number
    - Deck state (23): from deck_stats_to_vector
    - Synergy features (6): color/curve/balance/archetype fit
    Total: ~75 features (vs baseline 16)
    """
    # Group picks by event to track draft progression
    grouped_picks = group_picks_by_event(picks)

    feature_rows: List[PickFeatures] = []

    for _event_id, event_picks in grouped_picks.items():
        # Track cards picked so far in this draft
        picked_so_far: List[str] = []

        for pick in event_picks:
            # Build pack vectors
            pack_vectors: List[np.ndarray] = []
            pack_cards: List[CardMetadata] = []

            for name in pick.pack_contents:
                card = metadata.get(name)
                if card:
                    pack_vectors.append(card_to_vector(card, include_winrates=True))
                    pack_cards.append(card)

            if not pack_vectors or not pack_cards:
                continue

            pack_matrix = np.vstack(pack_vectors)

            # Pack aggregate features
            pack_mean = pack_matrix.mean(axis=0)
            pack_max = pack_matrix.max(axis=0)  # Signals for best cards in pack
            pack_std = pack_matrix.std(axis=0)[:3]  # Std of mana, rarity, first color

            # Pack-level statistics
            pack_size = float(len(pack_cards))
            bomb_count = sum(1 for c in pack_cards if c.rarity.lower() in ["rare", "mythic"])
            rare_count = sum(1 for c in pack_cards if c.rarity.lower() == "rare")

            # Win rate statistics from pack
            gih_values = [c.gih_wr for c in pack_cards if c.gih_wr is not None]
            avg_pack_gih = np.mean(gih_values) if gih_values else 0.0
            max_pack_gih = np.max(gih_values) if gih_values else 0.0

            pack_stats = np.array(
                [pack_size, float(bomb_count), float(rare_count), avg_pack_gih, max_pack_gih],
                dtype=float,
            )

            # Contextual features
            contextual = np.array(
                [float(pick.pick_number), float(pick.pack_number)], dtype=float
            )

            # Deck state features (what we've picked so far)
            deck_stats = compute_deck_stats(picked_so_far, metadata)
            deck_vector = deck_stats_to_vector(deck_stats)

            # Chosen card features
            chosen_card = metadata.get(pick.chosen_card)
            if not chosen_card:
                continue

            chosen_vector = card_to_vector(chosen_card, include_winrates=True)

            # Synergy features (how well does chosen card fit our deck?)
            deck_card_objects = [metadata[c] for c in picked_so_far if c in metadata]
            synergy_vector = compute_synergy_features(chosen_card, deck_card_objects)

            # Concatenate all features
            feature_vector = np.concatenate(
                [
                    chosen_vector,  # 13
                    pack_mean,  # 13
                    pack_max,  # 13
                    pack_std,  # 3
                    pack_stats,  # 5
                    contextual,  # 2
                    deck_vector,  # 23
                    synergy_vector,  # 6
                ]
            )

            feature_rows.append(PickFeatures(features=feature_vector, label=pick.chosen_card))

            # Update picked cards for next iteration
            picked_so_far.append(pick.chosen_card)

    return feature_rows


def summarize_labels(picks: Iterable[PickFeatures]) -> Mapping[str, int]:
    """Count label frequencies for sanity checks."""

    counts: MutableMapping[str, int] = {}
    for pick in picks:
        counts[pick.label] = counts.get(pick.label, 0) + 1
    return counts
