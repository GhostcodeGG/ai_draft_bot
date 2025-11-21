"""Track draft state and compute advanced contextual features.

This module provides utilities to track the evolving state of a draft and extract
rich features that capture deck composition, mana curve, color commitment, and
strategic signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Sequence

import numpy as np

from ai_draft_bot.data.ingest_17l import CardMetadata, PickRecord


@dataclass
class DraftState:
    """Tracks the current state of a single drafter's pool during a draft.

    Attributes:
        picked_cards: List of card names picked so far (in chronological order)
        current_pick_number: Current pick number in the draft (1-indexed)
        current_pack_number: Current pack number (1-indexed, typically 1-3)
    """

    picked_cards: List[str] = field(default_factory=list)
    current_pick_number: int = 0
    current_pack_number: int = 1

    def add_pick(self, card_name: str, pack_number: int, pick_number: int) -> None:
        """Record a new pick and update draft state."""
        self.picked_cards.append(card_name)
        self.current_pick_number = pick_number
        self.current_pack_number = pack_number

    def get_deck_size(self) -> int:
        """Return the number of cards picked so far."""
        return len(self.picked_cards)


@dataclass
class DeckStats:
    """Statistical summary of the current deck composition.

    Attributes:
        total_cards: Number of cards in deck so far
        color_counts: Dictionary mapping each color (W/U/B/R/G) to count
        color_commitment: Primary colors based on card count
        avg_mana_value: Average converted mana cost
        mana_curve: Distribution of cards by mana value (buckets: 0-1, 2, 3, 4, 5, 6+)
        creature_count: Number of creatures
        spell_count: Number of non-creature spells
        removal_count: Heuristic count of removal spells
        rarity_distribution: Count by rarity (common/uncommon/rare/mythic)
        avg_gih_wr: Average GIH WR of cards in deck (if available)
    """

    total_cards: int = 0
    color_counts: Mapping[str, int] = field(default_factory=dict)
    color_commitment: List[str] = field(default_factory=list)
    avg_mana_value: float = 0.0
    mana_curve: List[int] = field(default_factory=lambda: [0] * 7)
    creature_count: int = 0
    spell_count: int = 0
    removal_count: int = 0
    rarity_distribution: Mapping[str, int] = field(default_factory=dict)
    avg_gih_wr: float = 0.0


def compute_deck_stats(
    picked_cards: Sequence[str], metadata: Mapping[str, CardMetadata]
) -> DeckStats:
    """Compute comprehensive statistics about the current deck composition.

    Args:
        picked_cards: List of card names picked so far
        metadata: Card metadata lookup

    Returns:
        DeckStats with aggregated information about deck composition
    """
    if not picked_cards:
        return DeckStats()

    # Initialize counters
    color_counts: dict[str, int] = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0}
    mana_curve = [0] * 7  # Buckets: 0-1, 2, 3, 4, 5, 6+
    rarity_counts: dict[str, int] = {"common": 0, "uncommon": 0, "rare": 0, "mythic": 0}
    creature_count = 0
    spell_count = 0
    removal_count = 0
    total_mv = 0.0
    total_gih_wr = 0.0
    gih_wr_count = 0

    for card_name in picked_cards:
        card = metadata.get(card_name)
        if not card:
            continue

        # Color tracking
        if card.color in color_counts:
            color_counts[card.color] += 1
        elif len(card.color) > 1:  # Multicolor
            for c in card.color:
                if c in color_counts:
                    color_counts[c] += 1

        # Mana curve
        mv = int(card.mana_value)
        if mv <= 1:
            mana_curve[0] += 1
        elif mv >= 6:
            mana_curve[6] += 1
        else:
            mana_curve[mv] += 1
        total_mv += card.mana_value

        # Creature vs spell
        if "Creature" in card.type_line:
            creature_count += 1
        else:
            spell_count += 1

        # Heuristic removal detection
        type_lower = card.type_line.lower()
        if any(
            keyword in type_lower
            for keyword in ["instant", "sorcery", "enchantment", "artifact"]
        ):
            # Simple heuristic - refine with actual card text analysis later
            if any(
                word in card.name.lower()
                for word in ["murder", "bolt", "destroy", "exile", "removal", "kill"]
            ):
                removal_count += 1

        # Rarity
        rarity_key = card.rarity.lower()
        if rarity_key in rarity_counts:
            rarity_counts[rarity_key] += 1

        # Win rate
        if card.gih_wr is not None:
            total_gih_wr += card.gih_wr
            gih_wr_count += 1

    # Determine color commitment (colors with most cards)
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    color_commitment = [color for color, count in sorted_colors if count > 0][:2]

    avg_mv = total_mv / len(picked_cards) if picked_cards else 0.0
    avg_gih = total_gih_wr / gih_wr_count if gih_wr_count > 0 else 0.0

    return DeckStats(
        total_cards=len(picked_cards),
        color_counts=color_counts,
        color_commitment=color_commitment,
        avg_mana_value=avg_mv,
        mana_curve=mana_curve,
        creature_count=creature_count,
        spell_count=spell_count,
        removal_count=removal_count,
        rarity_distribution=rarity_counts,
        avg_gih_wr=avg_gih,
    )


def deck_stats_to_vector(stats: DeckStats) -> np.ndarray:
    """Convert DeckStats to a fixed-size feature vector.

    Returns:
        numpy array with following features:
        [total_cards, color_counts (5), avg_mv, mana_curve (7),
         creature_count, spell_count, removal_count, rarity_counts (4),
         avg_gih_wr]
        Total: 23 features
    """
    features = [
        float(stats.total_cards),
        # Color counts (5 features)
        float(stats.color_counts.get("W", 0)),
        float(stats.color_counts.get("U", 0)),
        float(stats.color_counts.get("B", 0)),
        float(stats.color_counts.get("R", 0)),
        float(stats.color_counts.get("G", 0)),
        # Average mana value
        float(stats.avg_mana_value),
        # Mana curve (7 buckets)
        *[float(x) for x in stats.mana_curve],
        # Counts
        float(stats.creature_count),
        float(stats.spell_count),
        float(stats.removal_count),
        # Rarity distribution (4 features)
        float(stats.rarity_distribution.get("common", 0)),
        float(stats.rarity_distribution.get("uncommon", 0)),
        float(stats.rarity_distribution.get("rare", 0)),
        float(stats.rarity_distribution.get("mythic", 0)),
        # Win rate
        float(stats.avg_gih_wr),
    ]
    return np.array(features, dtype=float)


def build_draft_states(picks: Sequence[PickRecord]) -> Mapping[str, DraftState]:
    """Build DraftState for each event ID by replaying picks chronologically.

    Args:
        picks: List of all pick records

    Returns:
        Dictionary mapping event_id to DraftState
    """
    from ai_draft_bot.data.ingest_17l import group_picks_by_event

    states: dict[str, DraftState] = {}
    grouped = group_picks_by_event(picks)

    for event_id, event_picks in grouped.items():
        state = DraftState()
        for pick in event_picks:
            state.add_pick(pick.chosen_card, pick.pack_number, pick.pick_number)
        states[event_id] = state

    return states
