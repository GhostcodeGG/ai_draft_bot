"""Pre-compute Scryfall card text features for all Standard sets.

This script fetches all cards from specified MTG sets and pre-computes their
text features (keywords, removal, card advantage, etc.) to avoid API calls
during training and inference.

Usage:
    python scripts/cache_scryfall.py --sets "BRO,ONE,MOM,WOE,LCI,MKM,OTJ,BLB,DSK,FDN"
    python scripts/cache_scryfall.py --all-standard
    python scripts/cache_scryfall.py --set-codes BRO ONE MOM
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

from ai_draft_bot.data.scryfall_client import get_cache_info, get_full_card_info
from ai_draft_bot.features.card_text import (
    CardTextFeatures,
    extract_card_text_features,
)
from ai_draft_bot.data.ingest_17l import CardMetadata
from ai_draft_bot.utils.logging_config import setup_logging

app = typer.Typer(help="Pre-compute Scryfall card features for fast training/inference")
logger = logging.getLogger(__name__)

# Standard-legal sets (update quarterly when new sets release)
STANDARD_SETS = [
    "BRO",  # The Brothers' War
    "ONE",  # Phyrexia: All Will Be One
    "MOM",  # March of the Machine
    "WOE",  # Wilds of Eldraine
    "LCI",  # The Lost Caverns of Ixalan
    "MKM",  # Murders at Karlov Manor
    "OTJ",  # Outlaws of Thunder Junction
    "BLB",  # Bloomburrow
    "DSK",  # Duskmourn: House of Horror
    "FDN",  # Foundations
]

CACHE_DIR = Path("cache/scryfall")
FEATURES_CACHE_FILE = CACHE_DIR / "card_text_features.json"


def fetch_set_cards(set_code: str) -> list[dict[str, Any]]:
    """Fetch all cards from a specific set via Scryfall API.

    Args:
        set_code: Three-letter set code (e.g., "BRO", "ONE")

    Returns:
        List of card info dictionaries
    """
    import scrython as scryfallsdk

    logger.info(f"Fetching cards from set: {set_code}")
    cards_data = []

    try:
        # Search for all cards in the set (excluding tokens, art cards, etc.)
        query = f'set:{set_code} (type:creature OR type:instant OR type:sorcery OR type:enchantment OR type:artifact OR type:planeswalker OR type:land)'
        cards = scryfallsdk.cards.Search(query)

        for card in cards:
            # Use get_full_card_info to leverage existing caching
            card_info = get_full_card_info(card.name)
            if card_info:
                cards_data.append(card_info)

        logger.info(f"✓ Fetched {len(cards_data)} cards from {set_code}")
        return cards_data

    except Exception as e:
        logger.error(f"Error fetching set {set_code}: {e}")
        return []


def compute_text_features(card_info: dict[str, Any]) -> dict[str, Any]:
    """Compute text features for a card.

    Args:
        card_info: Card information from Scryfall

    Returns:
        Dictionary with computed features
    """
    # Create minimal CardMetadata for feature extraction
    metadata = CardMetadata(
        name=card_info["name"],
        mana_value=card_info.get("cmc", 0),
        color=",".join(card_info.get("colors", [])),
        rarity=card_info.get("rarity", "common"),
        type_line=card_info.get("type_line", ""),
    )

    # Extract features (will use provided oracle_text instead of API call)
    features = extract_card_text_features(
        card=metadata,
        card_text=card_info.get("oracle_text", ""),
        use_scryfall=False,  # Don't call API, we already have the text
    )

    return {
        "name": card_info["name"],
        "oracle_text": card_info.get("oracle_text", ""),
        "keywords": card_info.get("keywords", []),
        "type_line": card_info.get("type_line", ""),
        "features": {
            "has_keywords": features.has_keywords,
            "keyword_count": features.keyword_count,
            "has_evasion": features.has_evasion,
            "has_combat_keyword": features.has_combat_keyword,
            "is_removal": features.is_removal,
            "gives_card_advantage": features.gives_card_advantage,
            "is_ramp": features.is_ramp,
            "tribal_synergies": list(features.tribal_synergies),
            "has_graveyard_synergy": features.has_graveyard_synergy,
            "has_sacrifice_synergy": features.has_sacrifice_synergy,
            "has_counter_synergy": features.has_counter_synergy,
            "power_level_score": features.power_level_score,
        },
    }


@app.command()
def build(
    set_codes: list[str] = typer.Option(
        None,
        "--set-codes",
        "-s",
        help="Space-separated set codes (e.g., BRO ONE MOM)",
    ),
    all_standard: bool = typer.Option(
        False,
        "--all-standard",
        help="Cache all Standard-legal sets",
    ),
    output_path: Path = typer.Option(
        FEATURES_CACHE_FILE,
        "--output-path",
        "-o",
        help="Output JSON file path",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level",
    ),
) -> None:
    """Build card text features cache from Scryfall data.

    Examples:
        python scripts/cache_scryfall.py build --all-standard
        python scripts/cache_scryfall.py build --set-codes BRO ONE MOM
    """
    setup_logging(level=log_level)

    # Determine which sets to cache
    if all_standard:
        sets_to_cache = STANDARD_SETS
        logger.info(f"Caching all {len(sets_to_cache)} Standard sets")
    elif set_codes:
        sets_to_cache = [s.upper() for s in set_codes]
        logger.info(f"Caching {len(sets_to_cache)} sets: {', '.join(sets_to_cache)}")
    else:
        typer.echo("Error: Must specify --all-standard or --set-codes")
        raise typer.Exit(1)

    # Ensure cache directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache if it exists
    existing_cache = {}
    if output_path.exists():
        logger.info(f"Loading existing cache from {output_path}")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_cache = json.load(f)
            logger.info(f"✓ Loaded {len(existing_cache)} existing entries")
        except Exception as e:
            logger.warning(f"Could not load existing cache: {e}")

    # Fetch and process cards from each set
    all_features = existing_cache.copy()
    total_new = 0
    total_updated = 0

    for set_code in sets_to_cache:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing set: {set_code}")
        logger.info(f"{'='*60}")

        cards = fetch_set_cards(set_code)

        if not cards:
            logger.warning(f"No cards found for set {set_code}, skipping")
            continue

        # Compute features for each card
        logger.info(f"Computing text features for {len(cards)} cards...")
        for card_info in tqdm(cards, desc=f"{set_code}"):
            try:
                card_name = card_info["name"]
                features_data = compute_text_features(card_info)

                if card_name in all_features:
                    total_updated += 1
                else:
                    total_new += 1

                all_features[card_name] = features_data

            except Exception as e:
                logger.error(f"Error processing card {card_info.get('name', 'unknown')}: {e}")
                continue

    # Save cache to disk
    logger.info(f"\n{'='*60}")
    logger.info(f"Saving cache to {output_path}")
    logger.info(f"{'='*60}")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_features, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Cache saved successfully!")
        logger.info(f"  - Total entries: {len(all_features)}")
        logger.info(f"  - New entries: {total_new}")
        logger.info(f"  - Updated entries: {total_updated}")
        logger.info(f"  - File size: {output_path.stat().st_size / 1024:.1f} KB")

    except Exception as e:
        logger.error(f"Failed to save cache: {e}")
        raise typer.Exit(1)

    # Show cache statistics
    cache_info = get_cache_info()
    logger.info(f"\nScryfall API Cache Stats:")
    logger.info(f"  - Disk entries: {cache_info['disk_entries']}")
    logger.info(f"  - Memory cache size: {cache_info['size']}/{cache_info['maxsize']}")
    logger.info(f"  - API errors: {cache_info['api_errors']}")


@app.command()
def stats(
    cache_path: Path = typer.Option(
        FEATURES_CACHE_FILE,
        "--cache-path",
        "-c",
        help="Path to features cache file",
    ),
) -> None:
    """Display statistics about the cached features.

    Examples:
        python scripts/cache_scryfall.py stats
    """
    if not cache_path.exists():
        typer.echo(f"Cache file not found: {cache_path}")
        typer.echo("Run 'build' command first to create the cache.")
        raise typer.Exit(1)

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # Compute statistics
    total_cards = len(cache)
    with_keywords = sum(1 for c in cache.values() if c["features"]["has_keywords"])
    removal_cards = sum(1 for c in cache.values() if c["features"]["is_removal"])
    ca_cards = sum(1 for c in cache.values() if c["features"]["gives_card_advantage"])
    evasion_cards = sum(1 for c in cache.values() if c["features"]["has_evasion"])

    # Display statistics
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Card Text Features Cache Statistics")
    typer.echo(f"{'='*60}")
    typer.echo(f"Cache file: {cache_path}")
    typer.echo(f"File size: {cache_path.stat().st_size / 1024:.1f} KB")
    typer.echo(f"\nTotal cards: {total_cards}")
    typer.echo(f"\nFeature Breakdown:")
    typer.echo(f"  - Has keywords: {with_keywords} ({with_keywords/total_cards:.1%})")
    typer.echo(f"  - Removal: {removal_cards} ({removal_cards/total_cards:.1%})")
    typer.echo(f"  - Card advantage: {ca_cards} ({ca_cards/total_cards:.1%})")
    typer.echo(f"  - Evasion: {evasion_cards} ({evasion_cards/total_cards:.1%})")

    # Show sample entries
    typer.echo(f"\nSample entries:")
    for i, (name, data) in enumerate(list(cache.items())[:3]):
        typer.echo(f"\n{i+1}. {name}")
        typer.echo(f"   Keywords: {', '.join(data['keywords']) if data['keywords'] else 'None'}")
        typer.echo(f"   Removal: {data['features']['is_removal']}")
        typer.echo(f"   Power score: {data['features']['power_level_score']:.2f}")


@app.command()
def update(
    set_code: str = typer.Argument(..., help="Set code to add/update (e.g., AED for Aetherdrift)"),
    cache_path: Path = typer.Option(
        FEATURES_CACHE_FILE,
        "--cache-path",
        "-c",
        help="Path to features cache file",
    ),
) -> None:
    """Update cache with a new set (e.g., when Aetherdrift releases).

    Examples:
        python scripts/cache_scryfall.py update AED
    """
    setup_logging()

    if not cache_path.exists():
        logger.warning(f"Cache file not found at {cache_path}, creating new cache")
        existing_cache = {}
    else:
        with open(cache_path, "r", encoding="utf-8") as f:
            existing_cache = json.load(f)

    logger.info(f"Updating cache with set: {set_code}")

    # Fetch new set
    cards = fetch_set_cards(set_code.upper())

    if not cards:
        logger.error(f"No cards found for set {set_code}")
        raise typer.Exit(1)

    # Compute features
    new_count = 0
    updated_count = 0

    for card_info in tqdm(cards, desc=f"Processing {set_code}"):
        try:
            card_name = card_info["name"]
            features_data = compute_text_features(card_info)

            if card_name in existing_cache:
                updated_count += 1
            else:
                new_count += 1

            existing_cache[card_name] = features_data

        except Exception as e:
            logger.error(f"Error processing {card_info.get('name', 'unknown')}: {e}")

    # Save updated cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(existing_cache, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Cache updated!")
    logger.info(f"  - New cards: {new_count}")
    logger.info(f"  - Updated cards: {updated_count}")
    logger.info(f"  - Total entries: {len(existing_cache)}")


if __name__ == "__main__":
    app()
