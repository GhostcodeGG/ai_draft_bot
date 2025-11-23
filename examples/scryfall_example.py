"""Example usage of the Scryfall API client.

This script demonstrates how to use the scryfall_client module to:
- Fetch card oracle text
- Get keyword abilities
- Use bulk fetching
- Monitor cache performance
- Prefetch entire sets

Run this after installing dependencies:
    pip install -e .
    python examples/scryfall_example.py
"""

from ai_draft_bot.data.scryfall_client import (
    bulk_fetch_cards,
    get_cache_info,
    get_full_card_info,
    get_keywords,
    get_oracle_text,
)


def example_basic_usage() -> None:
    """Demonstrate basic card lookups."""
    print("=" * 70)
    print("BASIC USAGE EXAMPLES")
    print("=" * 70)

    # Example 1: Get oracle text
    print("\n1. Fetching oracle text for 'Lightning Bolt':")
    text = get_oracle_text("Lightning Bolt")
    if text:
        print(f"   Oracle text: {text}")
    else:
        print("   Card not found!")

    # Example 2: Get keywords
    print("\n2. Fetching keywords for 'Serra Angel':")
    keywords = get_keywords("Serra Angel")
    print(f"   Keywords: {keywords}")

    # Example 3: Get full card info
    print("\n3. Getting full info for 'Counterspell':")
    info = get_full_card_info("Counterspell")
    if info:
        print(f"   Name: {info['name']}")
        print(f"   Mana Cost: {info['mana_cost']}")
        print(f"   Type: {info['type_line']}")
        print(f"   Oracle Text: {info['oracle_text']}")
        print(f"   Keywords: {info['keywords']}")


def example_bulk_fetching() -> None:
    """Demonstrate bulk card fetching."""
    print("\n" + "=" * 70)
    print("BULK FETCHING EXAMPLE")
    print("=" * 70)

    cards_to_fetch = [
        "Lightning Bolt",
        "Giant Growth",
        "Counterspell",
        "Ancestral Recall",
        "Black Lotus",
    ]

    print(f"\nFetching {len(cards_to_fetch)} cards...")
    results = bulk_fetch_cards(cards_to_fetch)

    for name, card in results.items():
        if card:
            oracle_text = getattr(card, "oracle_text", "No text")
            print(f"  ✓ {name}: {oracle_text[:50]}...")
        else:
            print(f"  ✗ {name}: Not found")


def example_cache_monitoring() -> None:
    """Demonstrate cache statistics."""
    print("\n" + "=" * 70)
    print("CACHE MONITORING")
    print("=" * 70)

    # Fetch some cards
    print("\nFetching 3 cards for the first time...")
    get_oracle_text("Lightning Bolt")
    get_oracle_text("Counterspell")
    get_oracle_text("Giant Growth")

    # Show cache stats
    stats = get_cache_info()
    print("\nCache statistics after first fetch:")
    print(f"  Cache size: {stats['size']}/{stats['maxsize']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  API errors: {stats['api_errors']}")

    # Fetch the same cards again
    print("\nFetching same cards again (should use cache)...")
    get_oracle_text("Lightning Bolt")
    get_oracle_text("Counterspell")
    get_oracle_text("Giant Growth")

    # Show updated stats
    stats = get_cache_info()
    print("\nCache statistics after second fetch:")
    print(f"  Cache size: {stats['size']}/{stats['maxsize']}")
    print(f"  Cache hits: {stats['hits']} (increased!)")
    print(f"  Cache misses: {stats['misses']}")

    # Calculate hit rate
    total = stats["hits"] + stats["misses"]
    hit_rate = (stats["hits"] / total * 100) if total > 0 else 0
    print(f"  Hit rate: {hit_rate:.1f}%")


def example_set_prefetching() -> None:
    """Demonstrate set prefetching (commented out to avoid long runtime)."""
    print("\n" + "=" * 70)
    print("SET PREFETCHING (OPTIONAL)")
    print("=" * 70)

    print("\nYou can prefetch an entire set to warm the cache:")
    print("  count = prefetch_set_cards('MID')  # Midnight Hunt")
    print("  print(f'Prefetched {count} cards')")
    print("\nThis is useful when training on a specific Limited format.")
    print("Uncomment the code below to try it:")
    print()

    # Uncomment to actually prefetch a set (will take ~30-60 seconds):
    # print("\nPrefetching Midnight Hunt (MID)...")
    # count = prefetch_set_cards("MID")
    # print(f"Successfully prefetched {count} cards!")


def example_integration_with_features() -> None:
    """Demonstrate how card_text.py uses the Scryfall client."""
    print("\n" + "=" * 70)
    print("INTEGRATION WITH FEATURE EXTRACTION")
    print("=" * 70)

    print("\nThe card_text.py module uses this client automatically:")
    print("""
    from ai_draft_bot.features.card_text import extract_card_text_features
    from ai_draft_bot.data.ingest_17l import CardMetadata

    # Create sample card metadata
    card = CardMetadata(
        name="Serra Angel",
        mana_value=5.0,
        rarity="Uncommon",
        color_identity="W",
        type_line="Creature — Angel",
        colors=["W"],
        gih_wr=0.58,  # Game-in-hand win rate
        iwd=0.12      # Improvement when drawn
    )

    # Extract features (will fetch from Scryfall automatically)
    features = extract_card_text_features(card, use_scryfall=True)

    print(f"Keywords found: {features.keyword_count}")
    print(f"Has evasion: {features.has_evasion}")
    print(f"Has combat keyword: {features.has_combat_keyword}")
    """)


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print("SCRYFALL API CLIENT EXAMPLES")
    print("=" * 70)

    try:
        example_basic_usage()
        example_bulk_fetching()
        example_cache_monitoring()
        example_set_prefetching()
        example_integration_with_features()

        print("\n" + "=" * 70)
        print("EXAMPLES COMPLETE!")
        print("=" * 70)
        print("\nThe Scryfall client is ready to use in your draft bot.")
        print("It will automatically cache cards and respect API rate limits.")
        print("\nFor more info, see: src/ai_draft_bot/data/scryfall_client.py")

    except ImportError as e:
        print("\n" + "=" * 70)
        print("SETUP REQUIRED")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nPlease install the package first:")
        print("  pip install -e .")
        print("\nOr install just the dependencies:")
        print("  pip install scryfallsdk")

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Make sure you have internet connection for Scryfall API access.")


if __name__ == "__main__":
    main()
