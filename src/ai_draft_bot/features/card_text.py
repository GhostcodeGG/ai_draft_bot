"""Card text analysis for keyword extraction and mechanic detection.

This module provides sophisticated card text parsing to extract:
- Keyword abilities (Flying, First Strike, Removal, etc.)
- Tribal synergies (Elves, Goblins, Humans, etc.)
- Mechanic synergies (Energy, Sacrifice, Graveyard, etc.)
- Card text categories (Removal, Card Draw, Ramp, etc.)

Performance: Uses pre-computed cache when available (10x faster than API calls)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Set

import numpy as np

from ai_draft_bot.data.ingest_17l import CardMetadata
from ai_draft_bot.utils.cache import cached_card_features

logger = logging.getLogger(__name__)

# Pre-computed features cache
_FEATURES_CACHE_PATH = Path("cache/scryfall/card_text_features.json")
_FEATURES_CACHE: dict[str, dict] | None = None
_CACHE_LOADED = False

# Make Scryfall optional to avoid breaking tests
try:
    from ai_draft_bot.data.scryfall_client import get_keywords, get_oracle_text
    SCRYFALL_AVAILABLE = True
except ImportError:
    SCRYFALL_AVAILABLE = False
    # Provide fallback functions
    def get_keywords(card_name: str) -> Set[str]:
        return set()
    def get_oracle_text(card_name: str) -> str:
        return ""

# Common MTG keyword abilities
KEYWORD_ABILITIES = {
    "flying",
    "first strike",
    "double strike",
    "deathtouch",
    "defender",
    "hexproof",
    "indestructible",
    "lifelink",
    "menace",
    "reach",
    "trample",
    "vigilance",
    "haste",
    "flash",
    "prowess",
    "ward",
    "protection",
}

# Evasion keywords (help deal damage)
EVASION_KEYWORDS = {
    "flying",
    "menace",
    "unblockable",
    "can't be blocked",
    "shadow",
    "horsemanship",
}

# Combat keywords
COMBAT_KEYWORDS = {
    "first strike",
    "double strike",
    "deathtouch",
    "trample",
    "vigilance",
}

# Removal indicators
REMOVAL_KEYWORDS = {
    "destroy",
    "exile",
    "deal damage",
    "damage to any target",
    "damage to target creature",
    "bounce",
    "return to hand",
    "sacrifice",
    "-X/-X",
}

# Card advantage indicators
CARD_ADVANTAGE_KEYWORDS = {
    "draw",
    "search your library",
    "mill",
    "surveil",
    "scry",
    "look at the top",
}

# Ramp indicators
RAMP_KEYWORDS = {
    "add",
    "mana",
    "untap",
    "search your library for a land",
    "put a land onto the battlefield",
}

# Common tribal types
TRIBAL_TYPES = {
    "elf",
    "elves",
    "goblin",
    "goblins",
    "human",
    "humans",
    "zombie",
    "zombies",
    "vampire",
    "vampires",
    "dragon",
    "dragons",
    "merfolk",
    "angel",
    "angels",
    "demon",
    "demons",
    "beast",
    "beasts",
    "soldier",
    "soldiers",
    "wizard",
    "wizards",
}

# Graveyard synergies
GRAVEYARD_KEYWORDS = {
    "graveyard",
    "flashback",
    "delve",
    "escape",
    "embalm",
    "eternalize",
    "unearth",
    "return from your graveyard",
}

# Sacrifice synergies
SACRIFICE_KEYWORDS = {
    "sacrifice",
    "dies",
    "when this creature dies",
    "whenever a creature dies",
}

# +1/+1 counter synergies
COUNTER_KEYWORDS = {
    "+1/+1 counter",
    "proliferate",
    "adapt",
    "evolve",
    "modular",
}


@dataclass
class CardTextFeatures:
    """Structured representation of card text features.

    Attributes:
        has_keywords: Presence of any keyword abilities
        keyword_count: Number of different keywords
        has_evasion: Has evasion ability (Flying, Menace, etc.)
        has_combat_keyword: Has combat-relevant keyword
        is_removal: Card provides removal
        gives_card_advantage: Card draws or filters cards
        is_ramp: Card provides mana acceleration
        tribal_synergies: Set of tribal types mentioned
        has_graveyard_synergy: Interacts with graveyard
        has_sacrifice_synergy: Sacrifice theme
        has_counter_synergy: +1/+1 counter synergy
        power_level_score: Heuristic power level (0.0-1.0)
    """

    has_keywords: bool = False
    keyword_count: int = 0
    has_evasion: bool = False
    has_combat_keyword: bool = False
    is_removal: bool = False
    gives_card_advantage: bool = False
    is_ramp: bool = False
    tribal_synergies: Set[str] = None  # type: ignore
    has_graveyard_synergy: bool = False
    has_sacrifice_synergy: bool = False
    has_counter_synergy: bool = False
    power_level_score: float = 0.5

    def __post_init__(self) -> None:
        if self.tribal_synergies is None:
            self.tribal_synergies = set()


def _load_features_cache() -> dict[str, dict]:
    """Load pre-computed features cache from disk.

    Returns:
        Dictionary mapping card names to feature data, or empty dict if cache not found
    """
    global _FEATURES_CACHE, _CACHE_LOADED

    if _CACHE_LOADED:
        return _FEATURES_CACHE or {}

    _CACHE_LOADED = True

    if not _FEATURES_CACHE_PATH.exists():
        logger.debug(f"Features cache not found at {_FEATURES_CACHE_PATH}")
        logger.debug("Run 'python scripts/cache_scryfall.py build --all-standard' to create cache")
        _FEATURES_CACHE = {}
        return {}

    try:
        with open(_FEATURES_CACHE_PATH, "r", encoding="utf-8") as f:
            _FEATURES_CACHE = json.load(f)
        logger.info(f"✓ Loaded {len(_FEATURES_CACHE)} pre-computed card features from cache")
        return _FEATURES_CACHE
    except Exception as e:
        logger.warning(f"Failed to load features cache: {e}")
        _FEATURES_CACHE = {}
        return {}


def _features_from_cache(card_name: str) -> CardTextFeatures | None:
    """Load features from pre-computed cache if available.

    Args:
        card_name: Name of the card

    Returns:
        CardTextFeatures if found in cache, None otherwise
    """
    cache = _load_features_cache()

    if not cache or card_name not in cache:
        return None

    try:
        data = cache[card_name]["features"]
        return CardTextFeatures(
            has_keywords=data["has_keywords"],
            keyword_count=data["keyword_count"],
            has_evasion=data["has_evasion"],
            has_combat_keyword=data["has_combat_keyword"],
            is_removal=data["is_removal"],
            gives_card_advantage=data["gives_card_advantage"],
            is_ramp=data["is_ramp"],
            tribal_synergies=set(data["tribal_synergies"]),
            has_graveyard_synergy=data["has_graveyard_synergy"],
            has_sacrifice_synergy=data["has_sacrifice_synergy"],
            has_counter_synergy=data["has_counter_synergy"],
            power_level_score=data["power_level_score"],
        )
    except Exception as e:
        logger.warning(f"Error loading cached features for {card_name}: {e}")
        return None


@cached_card_features
def extract_card_text_features(
    card: CardMetadata,
    card_text: str | None = None,
    use_scryfall: bool = True,
) -> CardTextFeatures:
    """Extract structured features from card text.

    Args:
        card: Card metadata
        card_text: Optional oracle text (if not provided, will fetch from Scryfall)
        use_scryfall: Whether to fetch real card text from Scryfall API

    Returns:
        CardTextFeatures with extracted information

    Performance:
        Checks pre-computed cache first (instant), falls back to Scryfall API (75ms)
    """
    # OPTIMIZATION: Check pre-computed cache first (10x faster than API)
    cached_features = _features_from_cache(card.name)
    if cached_features is not None:
        return cached_features

    # Try to get real card text from Scryfall
    if card_text is None and use_scryfall:
        try:
            card_text = get_oracle_text(card.name) or ""
        except Exception:
            card_text = ""

    # Fallback to empty string if still None
    if card_text is None:
        card_text = ""

    text_lower = card_text.lower()
    name_lower = card.name.lower()
    type_lower = card.type_line.lower()

    # Extract keywords (try Scryfall official keywords first)
    found_keywords = set()

    if use_scryfall:
        try:
            scryfall_keywords = get_keywords(card.name)
            for kw in scryfall_keywords:
                found_keywords.add(kw.lower())
        except Exception:
            pass  # Fall back to text parsing

    # Also parse from text as fallback
    for keyword in KEYWORD_ABILITIES:
        if keyword in text_lower:
            found_keywords.add(keyword)

    # Check for evasion
    has_evasion = any(kw in text_lower for kw in EVASION_KEYWORDS)

    # Check for combat keywords
    has_combat = any(kw in text_lower for kw in COMBAT_KEYWORDS)

    # Check for removal (also use name heuristics)
    is_removal = any(kw in text_lower for kw in REMOVAL_KEYWORDS) or any(
        word in name_lower
        for word in ["murder", "bolt", "destroy", "exile", "removal", "kill", "doom"]
    )

    # Check for card advantage
    gives_ca = any(kw in text_lower for kw in CARD_ADVANTAGE_KEYWORDS)

    # Check for ramp
    is_ramp = any(kw in text_lower for kw in RAMP_KEYWORDS) or "land" in type_lower

    # Extract tribal synergies from type line and text
    tribal_synergies = set()
    for tribal in TRIBAL_TYPES:
        if tribal in type_lower or tribal in text_lower:
            tribal_synergies.add(tribal)

    # Graveyard synergy
    has_graveyard = any(kw in text_lower for kw in GRAVEYARD_KEYWORDS)

    # Sacrifice synergy
    has_sacrifice = any(kw in text_lower for kw in SACRIFICE_KEYWORDS)

    # Counter synergy
    has_counters = any(kw in text_lower for kw in COUNTER_KEYWORDS)

    # Calculate power level score (heuristic)
    power_score = 0.5  # Baseline

    # Rarity boost
    if card.rarity.lower() == "rare":
        power_score += 0.15
    elif card.rarity.lower() == "mythic":
        power_score += 0.25

    # Keyword boost
    power_score += min(len(found_keywords) * 0.05, 0.15)

    # Removal is premium
    if is_removal:
        power_score += 0.10

    # Card advantage is premium
    if gives_ca:
        power_score += 0.10

    # Evasion is valuable
    if has_evasion:
        power_score += 0.05

    # Clamp to [0, 1]
    power_score = max(0.0, min(1.0, power_score))

    return CardTextFeatures(
        has_keywords=len(found_keywords) > 0,
        keyword_count=len(found_keywords),
        has_evasion=has_evasion,
        has_combat_keyword=has_combat,
        is_removal=is_removal,
        gives_card_advantage=gives_ca,
        is_ramp=is_ramp,
        tribal_synergies=tribal_synergies,
        has_graveyard_synergy=has_graveyard,
        has_sacrifice_synergy=has_sacrifice,
        has_counter_synergy=has_counters,
        power_level_score=power_score,
    )


def card_text_to_vector(features: CardTextFeatures) -> np.ndarray:
    """Convert CardTextFeatures to a numeric feature vector.

    Returns:
        11-element array: [has_keywords, keyword_count, has_evasion, has_combat,
                          is_removal, gives_ca, is_ramp, has_graveyard, has_sacrifice,
                          has_counters, power_level_score]
    """
    return np.array(
        [
            float(features.has_keywords),
            float(features.keyword_count),
            float(features.has_evasion),
            float(features.has_combat_keyword),
            float(features.is_removal),
            float(features.gives_card_advantage),
            float(features.is_ramp),
            float(features.has_graveyard_synergy),
            float(features.has_sacrifice_synergy),
            float(features.has_counter_synergy),
            features.power_level_score,
        ],
        dtype=float,
    )


def compute_text_synergy_score(
    card_features: CardTextFeatures,
    deck_features: list[CardTextFeatures],
) -> float:
    """Compute how well a card's mechanics synergize with the deck.

    Args:
        card_features: Features of the card being evaluated
        deck_features: Features of cards already in deck

    Returns:
        Synergy score from 0.0 to 1.0
    """
    if not deck_features:
        return 0.5  # Neutral for first pick

    synergy_score = 0.5  # Baseline

    # Tribal synergy
    if card_features.tribal_synergies:
        deck_tribals = set()
        for df in deck_features:
            deck_tribals.update(df.tribal_synergies)

        overlap = card_features.tribal_synergies & deck_tribals
        if overlap:
            synergy_score += 0.2  # Strong tribal synergy

    # Graveyard synergy
    deck_graveyard_count = sum(1 for df in deck_features if df.has_graveyard_synergy)
    if card_features.has_graveyard_synergy and deck_graveyard_count > 0:
        synergy_score += min(0.15, deck_graveyard_count * 0.05)

    # Sacrifice synergy
    deck_sacrifice_count = sum(1 for df in deck_features if df.has_sacrifice_synergy)
    if card_features.has_sacrifice_synergy and deck_sacrifice_count > 0:
        synergy_score += min(0.15, deck_sacrifice_count * 0.05)

    # Counter synergy
    deck_counter_count = sum(1 for df in deck_features if df.has_counter_synergy)
    if card_features.has_counter_synergy and deck_counter_count > 0:
        synergy_score += min(0.15, deck_counter_count * 0.05)

    # Removal is always valuable (but slightly more so in control decks)
    if card_features.is_removal:
        deck_removal_count = sum(1 for df in deck_features if df.is_removal)
        if deck_removal_count < 3:  # Need more removal
            synergy_score += 0.1

    # Card advantage is always valuable
    if card_features.gives_card_advantage:
        synergy_score += 0.05

    return max(0.0, min(1.0, synergy_score))
