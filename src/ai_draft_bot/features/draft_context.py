"""Feature extraction and labeling utilities for draft picks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

from ai_draft_bot.data.ingest_17l import CardMetadata, PickRecord


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


def card_to_vector(card: CardMetadata) -> np.ndarray:
    rarity_score = rarity_value(card.rarity)
    color_vector = encode_color(card.color)
    return np.concatenate(([card.mana_value, rarity_score], color_vector))


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


def summarize_labels(picks: Iterable[PickFeatures]) -> Mapping[str, int]:
    """Count label frequencies for sanity checks."""

    counts: MutableMapping[str, int] = {}
    for pick in picks:
        counts[pick.label] = counts.get(pick.label, 0) + 1
    return counts
