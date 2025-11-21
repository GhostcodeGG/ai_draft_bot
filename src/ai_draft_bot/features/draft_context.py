"""Feature extraction and labeling utilities for draft picks.

This module provides both baseline (simple) and advanced feature extraction.
Advanced features include win rates, draft context, deck composition, and pack signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

from ai_draft_bot.data.ingest_17l import CardMetadata, PickRecord, group_picks_by_event
from ai_draft_bot.features.archetypes import archetype_features_to_vector, load_archetype_config
from ai_draft_bot.features.card_text import (
    card_text_to_vector,
    extract_card_text_features,
)
from ai_draft_bot.features.draft_state import compute_deck_stats, deck_stats_to_vector
from ai_draft_bot.features.opponent_model import build_opponent_model, opponent_model_to_vector
from ai_draft_bot.features.positional import (
    extract_positional_features,
    positional_features_to_vector,
)
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
    logger.info(f"Building baseline features for {len(picks)} pick records")

    feature_rows: List[PickFeatures] = []
    skipped_no_pack = 0
    skipped_no_label = 0

    for pick in picks:
        pack_vectors: List[np.ndarray] = []
        for name in pick.pack_contents:
            card = metadata.get(name)
            if card:
                pack_vectors.append(card_to_vector(card))
        if not pack_vectors:
            skipped_no_pack += 1
            continue

        pack_matrix = np.vstack(pack_vectors)
        pack_mean = pack_matrix.mean(axis=0)

        chosen_card = metadata.get(pick.chosen_card)
        if not chosen_card:
            # Skip if we cannot label the pick
            skipped_no_label += 1
            continue

        feature_vector = np.concatenate([card_to_vector(chosen_card), pack_mean])
        feature_rows.append(PickFeatures(features=feature_vector, label=pick.chosen_card))

    logger.info(f"Generated {len(feature_rows)} feature vectors (16 dimensions)")
    if skipped_no_pack > 0:
        logger.warning(f"Skipped {skipped_no_pack} picks with no valid pack cards in metadata")
    if skipped_no_label > 0:
        logger.warning(
            f"Skipped {skipped_no_label} picks with chosen card not in metadata"
        )

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
    logger.info(f"Building ADVANCED features for {len(picks)} pick records")

    # Group picks by event to track draft progression
    grouped_picks = group_picks_by_event(picks)
    logger.info(f"Processing {len(grouped_picks)} unique draft events")

    feature_rows: List[PickFeatures] = []
    skipped_no_pack = 0
    skipped_no_label = 0

    for event_idx, (_event_id, event_picks) in enumerate(grouped_picks.items(), 1):
        if event_idx % 100 == 0:
            logger.debug(f"Processing event {event_idx}/{len(grouped_picks)}")
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
                skipped_no_pack += 1
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
                skipped_no_label += 1
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

    # Log summary
    if feature_rows:
        feature_dim = len(feature_rows[0].features)
        logger.info(f"Generated {len(feature_rows)} feature vectors ({feature_dim} dimensions)")
    else:
        logger.warning("No feature vectors generated!")

    if skipped_no_pack > 0:
        logger.warning(f"Skipped {skipped_no_pack} picks with no valid pack cards in metadata")
    if skipped_no_label > 0:
        logger.warning(
            f"Skipped {skipped_no_label} picks with chosen card not in metadata"
        )

    return feature_rows


def build_ultra_advanced_pick_features(
    picks: Sequence[PickRecord],
    metadata: Mapping[str, CardMetadata],
    archetype_config_path: str | None = None,
) -> List[PickFeatures]:
    """Generate ULTRA-ADVANCED feature vectors with ALL enhancements.

    This is the next-generation feature extractor that includes:
    - All advanced features (78 dims)
    - Card text analysis (11 dims)
    - Positional features (13 dims)
    - Opponent modeling (16 dims)
    - Enhanced archetype modeling (4 dims)
    - Win rate interaction features (8 dims)

    Total: ~130 features (vs 78 advanced, 16 baseline)

    Feature breakdown:
    - Chosen card features (13): basic stats + win rates
    - Pack aggregate features (13): mean of pack
    - Pack max features (13): best card in pack
    - Pack statistics (5): std dev, pack size, bombs
    - Contextual (2): pick/pack number
    - Deck state (23): mana curve, colors, composition
    - Original synergy (6): color/curve/balance fit
    - Card text features (11): keywords, removal, card advantage
    - Positional features (13): wheeling, signals, openness
    - Opponent model (16): upstream/downstream colors, competition
    - Advanced archetypes (4): archetype fit scores
    - Win rate interactions (8): WR × synergy, WR × position

    Args:
        picks: Pick records
        metadata: Card metadata with win rates
        archetype_config_path: Optional path to archetype config JSON

    Returns:
        List of PickFeatures with ~130-dimensional vectors
    """
    logger.info(f"Building ULTRA-ADVANCED features for {len(picks)} pick records")
    logger.info(
        "This includes: text analysis, positional, opponent modeling, "
        "and enhanced archetypes"
    )

    # Load archetype config
    archetypes = None
    if archetype_config_path:
        from pathlib import Path
        archetypes = load_archetype_config(Path(archetype_config_path))
        logger.info(f"Loaded {len(archetypes)} archetype definitions from config")

    # Group picks by event
    grouped_picks = group_picks_by_event(picks)
    logger.info(f"Processing {len(grouped_picks)} unique draft events")

    feature_rows: List[PickFeatures] = []
    skipped_no_pack = 0
    skipped_no_label = 0

    for event_idx, (_event_id, event_picks) in enumerate(grouped_picks.items(), 1):
        if event_idx % 100 == 0:
            logger.debug(f"Processing event {event_idx}/{len(grouped_picks)}")

        # Track draft state
        picked_so_far: List[str] = []
        previous_pack_colors = None

        for pick in event_picks:
            # Build pack vectors and card objects
            pack_vectors: List[np.ndarray] = []
            pack_cards: List[CardMetadata] = []

            for name in pick.pack_contents:
                card = metadata.get(name)
                if card:
                    pack_vectors.append(card_to_vector(card, include_winrates=True))
                    pack_cards.append(card)

            if not pack_vectors or not pack_cards:
                skipped_no_pack += 1
                continue

            pack_matrix = np.vstack(pack_vectors)

            # Pack aggregate features (from advanced)
            pack_mean = pack_matrix.mean(axis=0)
            pack_max = pack_matrix.max(axis=0)
            pack_std = pack_matrix.std(axis=0)[:3]

            # Pack-level statistics
            pack_size = float(len(pack_cards))
            bomb_count = sum(1 for c in pack_cards if c.rarity.lower() in ["rare", "mythic"])
            rare_count = sum(1 for c in pack_cards if c.rarity.lower() == "rare")

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

            # Deck state features
            deck_stats = compute_deck_stats(picked_so_far, metadata)
            deck_vector = deck_stats_to_vector(deck_stats)

            # Chosen card
            chosen_card = metadata.get(pick.chosen_card)
            if not chosen_card:
                skipped_no_label += 1
                continue

            chosen_vector = card_to_vector(chosen_card, include_winrates=True)

            # Original synergy features
            deck_card_objects = [metadata[c] for c in picked_so_far if c in metadata]
            synergy_vector = compute_synergy_features(chosen_card, deck_card_objects)

            # NEW: Card text features
            card_text_features = extract_card_text_features(chosen_card, card_text=None)
            card_text_vector = card_text_to_vector(card_text_features)

            # NEW: Positional features
            positional_features = extract_positional_features(
                pick, pack_cards, chosen_card, metadata, previous_pack_colors
            )
            positional_vector = positional_features_to_vector(positional_features)
            previous_pack_colors = positional_features.color_openness  # Track for next pick

            # NEW: Opponent modeling
            opponent_model = build_opponent_model(
                pack_cards, picked_so_far, metadata, pick.pick_number, pick.pack_number
            )
            opponent_vector = opponent_model_to_vector(opponent_model)

            # NEW: Advanced archetype features
            archetype_vector = archetype_features_to_vector(
                chosen_card, deck_card_objects, archetypes
            )

            # NEW: Win rate interaction features
            wr_interactions = compute_winrate_interactions(
                chosen_card, synergy_vector, positional_features, card_text_features
            )

            # Concatenate ALL features
            feature_vector = np.concatenate(
                [
                    chosen_vector,        # 13
                    pack_mean,            # 13
                    pack_max,             # 13
                    pack_std,             # 3
                    pack_stats,           # 5
                    contextual,           # 2
                    deck_vector,          # 23
                    synergy_vector,       # 6
                    card_text_vector,     # 11
                    positional_vector,    # 13
                    opponent_vector,      # 16
                    archetype_vector,     # 4
                    wr_interactions,      # 8
                ]
            )

            feature_rows.append(PickFeatures(features=feature_vector, label=pick.chosen_card))

            # Update picked cards
            picked_so_far.append(pick.chosen_card)

    # Log summary
    if feature_rows:
        feature_dim = len(feature_rows[0].features)
        logger.info(
            f"Generated {len(feature_rows)} ULTRA-ADVANCED feature vectors "
            f"({feature_dim} dimensions)"
        )
        logger.info("Feature composition:")
        logger.info("  - Card/pack features: 49 dims")
        logger.info("  - Deck state: 23 dims")
        logger.info("  - Original synergy: 6 dims")
        logger.info("  - Card text analysis: 11 dims")
        logger.info("  - Positional features: 13 dims")
        logger.info("  - Opponent modeling: 16 dims")
        logger.info("  - Advanced archetypes: 4 dims")
        logger.info("  - Win rate interactions: 8 dims")
    else:
        logger.warning("No feature vectors generated!")

    if skipped_no_pack > 0:
        logger.warning(f"Skipped {skipped_no_pack} picks with no valid pack cards")
    if skipped_no_label > 0:
        logger.warning(f"Skipped {skipped_no_label} picks with missing chosen card")

    return feature_rows


def compute_winrate_interactions(
    card: CardMetadata,
    synergy_vector: np.ndarray,
    positional_features: object,
    card_text_features: object,
) -> np.ndarray:
    """Compute interaction features between win rate and other signals.

    These capture non-linear relationships like:
    - High WR cards are more valuable when they fit deck colors
    - Early picks with high WR are bombs
    - Removal with high WR is premium

    Returns:
        8-element array of interaction features
    """
    gih_wr = card.gih_wr if card.gih_wr is not None else 0.5
    iwd = card.iwd if card.iwd is not None else 0.0

    # Extract relevant features
    color_synergy = synergy_vector[1] if len(synergy_vector) > 1 else 0.5
    curve_synergy = synergy_vector[2] if len(synergy_vector) > 2 else 0.5

    # Type checking for positional_features
    wheel_prob = getattr(positional_features, 'wheeling_probability', 0.5)
    pack_quality = getattr(positional_features, 'pack_quality', 0.5)
    pick_stage = getattr(positional_features, 'pick_stage', 0.5)

    # Type checking for card_text_features
    is_removal = float(getattr(card_text_features, 'is_removal', False))
    gives_ca = float(getattr(card_text_features, 'gives_card_advantage', False))

    # Interaction features
    interactions = np.array(
        [
            gih_wr * color_synergy,     # WR × color fit
            gih_wr * curve_synergy,     # WR × curve fit
            gih_wr * (1.0 - pick_stage),  # WR × early pick (bombs picked early)
            iwd * color_synergy,        # IWD × color fit
            gih_wr * is_removal,        # Premium removal
            gih_wr * gives_ca,          # Premium card advantage
            gih_wr * pack_quality,      # Good cards in good packs
            (1.0 - wheel_prob) * gih_wr,  # Non-wheeling bombs
        ],
        dtype=float,
    )

    return interactions


def summarize_labels(picks: Iterable[PickFeatures]) -> Mapping[str, int]:
    """Count label frequencies for sanity checks."""

    counts: MutableMapping[str, int] = {}
    for pick in picks:
        counts[pick.label] = counts.get(pick.label, 0) + 1
    return counts
