"""Card synergy detection and archetype scoring.

This module provides utilities to detect synergies between cards and score
how well a card fits into the developing draft archetype.
"""

from __future__ import annotations

from typing import List, Set

import numpy as np

from ai_draft_bot.data.ingest_17l import CardMetadata

# Color pair archetypes (common in many sets)
COLOR_PAIR_ARCHETYPES = {
    "WU": "fliers",  # White-Blue: Fliers
    "UB": "control",  # Blue-Black: Control
    "BR": "sacrifice",  # Black-Red: Sacrifice
    "RG": "aggro",  # Red-Green: Aggro
    "GW": "tokens",  # Green-White: Tokens
    "WB": "lifegain",  # White-Black: Lifegain
    "UR": "spells",  # Blue-Red: Spells matter
    "BG": "graveyard",  # Black-Green: Graveyard
    "RW": "equipment",  # Red-White: Equipment/Aggro
    "GU": "ramp",  # Green-Blue: Ramp
}


def extract_keywords(type_line: str) -> Set[str]:
    """Extract important keywords from card type line.

    Returns:
        Set of relevant card types/supertypes
    """
    keywords = set()
    type_lower = type_line.lower()

    # Card types
    if "creature" in type_lower:
        keywords.add("creature")
    if "instant" in type_lower:
        keywords.add("instant")
    if "sorcery" in type_lower:
        keywords.add("sorcery")
    if "enchantment" in type_lower:
        keywords.add("enchantment")
    if "artifact" in type_lower:
        keywords.add("artifact")
    if "planeswalker" in type_lower:
        keywords.add("planeswalker")
    if "land" in type_lower:
        keywords.add("land")

    # Supertypes
    if "legendary" in type_lower:
        keywords.add("legendary")

    return keywords


def compute_color_synergy(card: CardMetadata, deck_cards: List[CardMetadata]) -> float:
    """Compute how well a card's color fits the current deck.

    Returns:
        Score from 0.0 to 1.0 indicating color synergy
        - 1.0: Perfect fit (matches main colors)
        - 0.5: Splash-able
        - 0.0: Off-color
    """
    if not deck_cards:
        return 1.0  # First pick is always "on-color"

    # Count colors in deck
    color_counts = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0}
    for deck_card in deck_cards:
        for c in deck_card.color:
            if c in color_counts:
                color_counts[c] += 1

    # Find primary colors (top 2)
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    primary_colors = {color for color, count in sorted_colors[:2] if count > 0}

    # Check card's colors
    card_colors = set(card.color)

    if not card_colors:  # Colorless
        return 0.7

    # Calculate overlap
    overlap = card_colors & primary_colors
    if len(overlap) == len(card_colors):  # All colors match
        return 1.0
    elif len(overlap) > 0:  # Partial match
        return 0.7
    else:  # Off-color
        return 0.3


def compute_mana_curve_synergy(card: CardMetadata, deck_cards: List[CardMetadata]) -> float:
    """Compute how well a card fits the mana curve.

    Returns:
        Score from 0.0 to 1.0
        - Higher if filling gaps in the curve
        - Lower if adding to already-saturated CMC slots
    """
    if not deck_cards:
        return 1.0

    # Build current curve
    curve_counts = [0] * 8  # 0-7+
    for deck_card in deck_cards:
        mv = min(int(deck_card.mana_value), 7)
        curve_counts[mv] += 1

    total_cards = len(deck_cards)
    card_mv = min(int(card.mana_value), 7)

    # Ideal curve percentages (rough heuristic)
    ideal_curve = {
        0: 0.0,  # Minimal 0-drops
        1: 0.05,  # Few 1-drops
        2: 0.20,  # Many 2-drops
        3: 0.25,  # Peak at 3
        4: 0.20,  # Solid 4-drops
        5: 0.15,  # Some 5-drops
        6: 0.10,  # Few 6-drops
        7: 0.05,  # Rare 7+
    }

    current_pct = curve_counts[card_mv] / total_cards if total_cards > 0 else 0.0
    ideal_pct = ideal_curve.get(card_mv, 0.1)

    # Penalize if over-represented, reward if under-represented
    if current_pct < ideal_pct:
        return 1.0  # Filling a gap
    elif current_pct < ideal_pct * 1.5:
        return 0.7  # Slightly over
    else:
        return 0.4  # Too many at this CMC


def compute_creature_spell_balance(
    card: CardMetadata, deck_cards: List[CardMetadata]
) -> float:
    """Score based on creature/spell balance.

    Typical limited decks want ~60% creatures, 40% spells.

    Returns:
        Score from 0.0 to 1.0
    """
    if not deck_cards:
        return 1.0

    creature_count = sum(1 for c in deck_cards if "Creature" in c.type_line)
    total = len(deck_cards)
    creature_pct = creature_count / total

    is_creature = "Creature" in card.type_line

    # Target: 60% creatures
    if is_creature:
        if creature_pct < 0.6:
            return 1.0  # Need more creatures
        elif creature_pct < 0.7:
            return 0.8
        else:
            return 0.5  # Too many creatures
    else:  # Spell
        if creature_pct > 0.6:
            return 1.0  # Need more spells
        elif creature_pct > 0.5:
            return 0.8
        else:
            return 0.5  # Too many spells


def compute_archetype_synergy(card: CardMetadata, deck_cards: List[CardMetadata]) -> float:
    """Compute archetype-specific synergy.

    This is a simplified heuristic. In a production system, you would:
    - Use card text analysis (NLP)
    - Maintain set-specific synergy tables
    - Learn synergies from data

    Returns:
        Score from 0.0 to 1.0
    """
    if not deck_cards:
        return 1.0

    # Identify deck's color pair
    color_counts = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0}
    for deck_card in deck_cards:
        for c in deck_card.color:
            if c in color_counts:
                color_counts[c] += 1

    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    primary_colors = "".join([color for color, count in sorted_colors[:2] if count > 0])

    # Heuristic synergies based on archetype
    archetype = COLOR_PAIR_ARCHETYPES.get(primary_colors, "unknown")

    card_keywords = extract_keywords(card.type_line)

    # Simple keyword-based synergy detection
    synergy_score = 0.5  # Neutral baseline

    if archetype == "fliers" and "creature" in card_keywords:
        # WU fliers: bonus for low-cost creatures and evasion
        if card.mana_value <= 3:
            synergy_score = 0.8
    elif archetype == "control" and "instant" in card_keywords:
        # UB control: bonus for instants and high-cost cards
        synergy_score = 0.9
    elif archetype == "aggro":
        # RG aggro: bonus for low-cost creatures
        if "creature" in card_keywords and card.mana_value <= 3:
            synergy_score = 0.9
    elif archetype == "spells" and ("instant" in card_keywords or "sorcery" in card_keywords):
        # UR spells: bonus for instants/sorceries
        synergy_score = 0.9

    return synergy_score


def compute_overall_synergy_score(
    card: CardMetadata, deck_cards: List[CardMetadata]
) -> float:
    """Compute overall synergy score combining multiple factors.

    Args:
        card: Card being evaluated
        deck_cards: Cards already in deck

    Returns:
        Weighted average synergy score from 0.0 to 1.0
    """
    color_syn = compute_color_synergy(card, deck_cards)
    curve_syn = compute_mana_curve_synergy(card, deck_cards)
    balance_syn = compute_creature_spell_balance(card, deck_cards)
    archetype_syn = compute_archetype_synergy(card, deck_cards)

    # Weighted average (color is most important)
    weights = {
        "color": 0.4,
        "curve": 0.2,
        "balance": 0.2,
        "archetype": 0.2,
    }

    overall = (
        color_syn * weights["color"]
        + curve_syn * weights["curve"]
        + balance_syn * weights["balance"]
        + archetype_syn * weights["archetype"]
    )

    return overall


def compute_synergy_features(
    card: CardMetadata, deck_cards: List[CardMetadata]
) -> np.ndarray:
    """Compute synergy feature vector for a card.

    Returns:
        6-element array: [overall_synergy, color_synergy, curve_synergy,
                         balance_synergy, archetype_synergy, deck_size_normalized]
    """
    overall = compute_overall_synergy_score(card, deck_cards)
    color_syn = compute_color_synergy(card, deck_cards)
    curve_syn = compute_mana_curve_synergy(card, deck_cards)
    balance_syn = compute_creature_spell_balance(card, deck_cards)
    archetype_syn = compute_archetype_synergy(card, deck_cards)

    # Normalized deck size (0.0 at start, 1.0 at ~40 cards)
    deck_size_norm = min(len(deck_cards) / 40.0, 1.0)

    return np.array(
        [overall, color_syn, curve_syn, balance_syn, archetype_syn, deck_size_norm],
        dtype=float,
    )
