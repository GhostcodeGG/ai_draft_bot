"""Set-specific archetype modeling and dynamic archetype discovery.

This module provides sophisticated archetype detection that goes beyond
generic color pair heuristics to capture set-specific strategies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Sequence

import numpy as np

from ai_draft_bot.data.ingest_17l import CardMetadata


@dataclass
class ArchetypeDefinition:
    """Definition of a draft archetype.

    Attributes:
        name: Archetype name (e.g., "WU Fliers", "BR Sacrifice")
        primary_colors: Main colors (e.g., ["W", "U"])
        secondary_colors: Splash colors (optional)
        keywords: Keywords that signal this archetype
        synergy_types: Mechanic categories (e.g., ["flying", "tempo"])
        card_preferences: Card types this archetype wants
        target_curve: Preferred mana curve (lower = aggro, higher = control)
    """

    name: str
    primary_colors: List[str]
    secondary_colors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    synergy_types: List[str] = field(default_factory=list)
    card_preferences: Mapping[str, float] = field(default_factory=dict)
    target_curve: float = 3.0  # Average CMC target


# Default archetype definitions (generic cross-set)
DEFAULT_ARCHETYPES = [
    ArchetypeDefinition(
        name="WU Fliers",
        primary_colors=["W", "U"],
        keywords=["flying", "flash", "bounce"],
        synergy_types=["evasion", "tempo"],
        card_preferences={"creature": 0.65, "instant": 0.20, "enchantment": 0.10},
        target_curve=2.8,
    ),
    ArchetypeDefinition(
        name="UB Control",
        primary_colors=["U", "B"],
        keywords=["draw", "removal", "counter"],
        synergy_types=["control", "card_advantage"],
        card_preferences={"instant": 0.25, "sorcery": 0.15, "creature": 0.50},
        target_curve=3.5,
    ),
    ArchetypeDefinition(
        name="BR Sacrifice",
        primary_colors=["B", "R"],
        keywords=["sacrifice", "dies", "damage"],
        synergy_types=["sacrifice", "aristocrats"],
        card_preferences={"creature": 0.70, "sorcery": 0.15},
        target_curve=3.0,
    ),
    ArchetypeDefinition(
        name="RG Aggro",
        primary_colors=["R", "G"],
        keywords=["haste", "trample", "power"],
        synergy_types=["aggro", "combat"],
        card_preferences={"creature": 0.75, "instant": 0.15},
        target_curve=2.5,
    ),
    ArchetypeDefinition(
        name="GW Tokens",
        primary_colors=["G", "W"],
        keywords=["token", "anthem", "pump"],
        synergy_types=["tokens", "go_wide"],
        card_preferences={"creature": 0.65, "enchantment": 0.15, "sorcery": 0.10},
        target_curve=3.2,
    ),
    ArchetypeDefinition(
        name="WB Lifegain",
        primary_colors=["W", "B"],
        keywords=["lifegain", "lifelink", "drain"],
        synergy_types=["lifegain", "attrition"],
        card_preferences={"creature": 0.65, "enchantment": 0.15},
        target_curve=3.0,
    ),
    ArchetypeDefinition(
        name="UR Spells",
        primary_colors=["U", "R"],
        keywords=["instant", "sorcery", "prowess", "magecraft"],
        synergy_types=["spells_matter", "tempo"],
        card_preferences={"instant": 0.30, "sorcery": 0.20, "creature": 0.45},
        target_curve=2.8,
    ),
    ArchetypeDefinition(
        name="BG Graveyard",
        primary_colors=["B", "G"],
        keywords=["graveyard", "mill", "recursion", "delve"],
        synergy_types=["graveyard", "value"],
        card_preferences={"creature": 0.65, "sorcery": 0.15, "instant": 0.10},
        target_curve=3.3,
    ),
    ArchetypeDefinition(
        name="RW Aggro",
        primary_colors=["R", "W"],
        keywords=["first strike", "haste", "equipment"],
        synergy_types=["aggro", "combat"],
        card_preferences={"creature": 0.70, "equipment": 0.10, "instant": 0.15},
        target_curve=2.6,
    ),
    ArchetypeDefinition(
        name="GU Ramp",
        primary_colors=["G", "U"],
        keywords=["ramp", "draw", "big creatures"],
        synergy_types=["ramp", "card_advantage"],
        card_preferences={"creature": 0.60, "sorcery": 0.15, "land": 0.05},
        target_curve=3.8,
    ),
]


def load_archetype_config(config_path: Path | str) -> List[ArchetypeDefinition]:
    """Load set-specific archetype definitions from JSON config.

    Args:
        config_path: Path to archetype config JSON file

    Returns:
        List of ArchetypeDefinition objects

    Example config format:
    {
      "archetypes": [
        {
          "name": "WU Fliers",
          "primary_colors": ["W", "U"],
          "keywords": ["flying", "flash"],
          "target_curve": 2.8
        }
      ]
    }
    """
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_ARCHETYPES

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    archetypes = []
    for arch_data in data.get("archetypes", []):
        archetypes.append(
            ArchetypeDefinition(
                name=arch_data["name"],
                primary_colors=arch_data.get("primary_colors", []),
                secondary_colors=arch_data.get("secondary_colors", []),
                keywords=arch_data.get("keywords", []),
                synergy_types=arch_data.get("synergy_types", []),
                card_preferences=arch_data.get("card_preferences", {}),
                target_curve=arch_data.get("target_curve", 3.0),
            )
        )

    return archetypes if archetypes else DEFAULT_ARCHETYPES


def score_deck_archetype_fit(
    deck_cards: Sequence[CardMetadata],
    archetype: ArchetypeDefinition,
) -> float:
    """Score how well a deck fits a given archetype.

    Args:
        deck_cards: Cards in the deck
        archetype: Archetype definition to score against

    Returns:
        Fit score (0.0-1.0), higher = better fit
    """
    if not deck_cards:
        return 0.0

    score = 0.0
    factors = 0

    # 1. Color match
    color_counts = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0}
    for card in deck_cards:
        for c in card.color:
            if c in color_counts:
                color_counts[c] += 1

    primary_card_count = sum(color_counts[c] for c in archetype.primary_colors)
    total_colored_cards = sum(color_counts.values())

    if total_colored_cards > 0:
        color_match = primary_card_count / total_colored_cards
        score += color_match
        factors += 1

    # 2. Mana curve match
    avg_mv = np.mean([c.mana_value for c in deck_cards])
    curve_diff = abs(avg_mv - archetype.target_curve)
    curve_score = max(0.0, 1.0 - curve_diff / 2.0)  # Penalty for deviation
    score += curve_score
    factors += 1

    # 3. Card type preferences
    if archetype.card_preferences:
        type_score = 0.0
        for card in deck_cards:
            for pref_type, pref_weight in archetype.card_preferences.items():
                if pref_type.lower() in card.type_line.lower():
                    type_score += pref_weight

        type_score = min(type_score / len(deck_cards), 1.0)
        score += type_score
        factors += 1

    # Average all factors
    return score / factors if factors > 0 else 0.0


def identify_deck_archetypes(
    deck_cards: Sequence[CardMetadata],
    archetypes: Sequence[ArchetypeDefinition] | None = None,
    top_k: int = 3,
) -> List[tuple[ArchetypeDefinition, float]]:
    """Identify the top-K archetypes that match the current deck.

    Args:
        deck_cards: Cards in deck so far
        archetypes: List of archetype definitions (uses default if None)
        top_k: Number of top archetypes to return

    Returns:
        List of (archetype, score) tuples, sorted by score descending
    """
    if archetypes is None:
        archetypes = DEFAULT_ARCHETYPES

    scores = []
    for archetype in archetypes:
        fit_score = score_deck_archetype_fit(deck_cards, archetype)
        scores.append((archetype, fit_score))

    # Sort by score descending and return top-K
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def compute_archetype_synergy(
    card: CardMetadata,
    deck_cards: Sequence[CardMetadata],
    archetypes: Sequence[ArchetypeDefinition] | None = None,
) -> float:
    """Compute how well a card fits the deck's emerging archetype(s).

    Args:
        card: Card being evaluated
        deck_cards: Cards in deck so far
        archetypes: Archetype definitions

    Returns:
        Archetype synergy score (0.0-1.0)
    """
    if not deck_cards:
        return 0.5  # Neutral for first pick

    # Identify deck's top archetypes
    top_archetypes = identify_deck_archetypes(deck_cards, archetypes, top_k=2)

    if not top_archetypes:
        return 0.5

    # Score card against top archetype
    primary_archetype, primary_score = top_archetypes[0]

    # Check if card matches primary colors
    color_match = any(c in primary_archetype.primary_colors for c in card.color)

    # Check if card matches archetype keywords
    card_name_lower = card.name.lower()
    card_type_lower = card.type_line.lower()

    keyword_match = any(
        kw in card_name_lower or kw in card_type_lower for kw in primary_archetype.keywords
    )

    # Check mana curve fit
    curve_diff = abs(card.mana_value - primary_archetype.target_curve)
    curve_fit = max(0.0, 1.0 - curve_diff / 3.0)

    # Weighted combination
    synergy = (
        (0.4 * float(color_match))
        + (0.3 * float(keyword_match))
        + (0.3 * curve_fit)
    )

    return max(0.0, min(1.0, synergy))


def archetype_features_to_vector(
    card: CardMetadata,
    deck_cards: Sequence[CardMetadata],
    archetypes: Sequence[ArchetypeDefinition] | None = None,
) -> np.ndarray:
    """Extract archetype-based features as a vector.

    Returns:
        4-element array: [archetype_synergy, primary_archetype_score,
                         secondary_archetype_score, archetype_consistency]
    """
    if not deck_cards:
        return np.array([0.5, 0.0, 0.0, 0.0], dtype=float)

    archetype_synergy = compute_archetype_synergy(card, deck_cards, archetypes)

    # Get top 2 archetype scores for deck
    top_archetypes = identify_deck_archetypes(deck_cards, archetypes, top_k=2)

    primary_score = top_archetypes[0][1] if top_archetypes else 0.0
    secondary_score = top_archetypes[1][1] if len(top_archetypes) > 1 else 0.0

    # Archetype consistency = how much better is primary vs secondary
    # High consistency = focused deck, low = unfocused
    consistency = primary_score - secondary_score if secondary_score > 0 else primary_score

    return np.array(
        [archetype_synergy, primary_score, secondary_score, consistency],
        dtype=float,
    )
