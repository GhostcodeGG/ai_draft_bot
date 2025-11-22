"""Scryfall API client for fetching Magic: The Gathering card data.

This module provides a cached interface to the Scryfall API for retrieving:
- Oracle text (official card text)
- Keywords (official keyword abilities)
- Card metadata

Features:
- LRU caching (2000 cards in memory)
- Rate limiting (respects Scryfall's 50-100ms guideline)
- Bulk fetching (up to 75 cards per request)
- Graceful error handling
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

import scryfallsdk
from scryfallsdk import Card

logger = logging.getLogger(__name__)

# Rate limiting: Scryfall requests ~50-100ms between requests
_last_request_time: float = 0.0
_MIN_REQUEST_INTERVAL = 0.075  # 75ms between requests (conservative)

# Cache statistics
_cache_hits = 0
_cache_misses = 0
_api_errors = 0


def _rate_limit() -> None:
    """Enforce rate limiting between API requests."""
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time

    if time_since_last < _MIN_REQUEST_INTERVAL:
        sleep_time = _MIN_REQUEST_INTERVAL - time_since_last
        time.sleep(sleep_time)

    _last_request_time = time.time()


@lru_cache(maxsize=2000)
def _get_card_cached(card_name: str) -> Card | None:
    """Fetch a card from Scryfall with caching and rate limiting.

    Args:
        card_name: The name of the card to fetch

    Returns:
        Card object or None if not found

    Note:
        This is an internal function. Use get_oracle_text() or get_keywords()
        for specific data extraction.
    """
    global _cache_hits, _cache_misses, _api_errors

    try:
        _rate_limit()
        card = scryfallsdk.cards.Named(fuzzy=card_name)
        _cache_misses += 1
        logger.debug(f"Fetched card from Scryfall: {card_name}")
        return card

    except scryfallsdk.ScryfallError as e:
        _api_errors += 1
        logger.warning(f"Scryfall API error for '{card_name}': {e}")
        return None

    except Exception as e:
        _api_errors += 1
        logger.error(f"Unexpected error fetching '{card_name}': {e}")
        return None


def get_oracle_text(card_name: str) -> str | None:
    """Get the official oracle text for a card.

    Args:
        card_name: The name of the card

    Returns:
        Oracle text string, or None if card not found or has no text

    Examples:
        >>> text = get_oracle_text("Lightning Bolt")
        >>> if text:
        ...     print(text)
        Lightning Bolt deals 3 damage to any target.
    """
    card = _get_card_cached(card_name)

    if card is None:
        return None

    # Some cards have no oracle text (e.g., basic lands, vanilla creatures)
    return getattr(card, "oracle_text", None)


def get_keywords(card_name: str) -> list[str]:
    """Get the official keyword abilities for a card.

    Args:
        card_name: The name of the card

    Returns:
        List of keyword strings (e.g., ["Flying", "First Strike"])
        Empty list if card not found or has no keywords

    Examples:
        >>> keywords = get_keywords("Serra Angel")
        >>> print(keywords)
        ['Flying', 'Vigilance']
    """
    card = _get_card_cached(card_name)

    if card is None:
        return []

    # Scryfall provides keywords as a list
    keywords = getattr(card, "keywords", [])
    return keywords if keywords else []


def get_card_types(card_name: str) -> list[str]:
    """Get the card's type line components.

    Args:
        card_name: The name of the card

    Returns:
        List of types (e.g., ["Creature", "Angel"])
        Empty list if card not found

    Examples:
        >>> types = get_card_types("Serra Angel")
        >>> print(types)
        ['Creature', 'Angel']
    """
    card = _get_card_cached(card_name)

    if card is None:
        return []

    type_line = getattr(card, "type_line", "")
    # Parse type line: "Creature — Angel" -> ["Creature", "Angel"]
    # Remove em-dash and split
    type_line = type_line.replace("—", "-")
    parts = [t.strip() for t in type_line.replace("-", " ").split()]
    return parts


def bulk_fetch_cards(card_names: list[str]) -> dict[str, Card | None]:
    """Fetch multiple cards efficiently (up to 75 per request).

    Note: Scryfall's Collection API supports bulk fetching, but the scryfallsdk
    library doesn't expose it directly. This implementation fetches cards individually
    but uses caching to minimize redundant requests.

    Args:
        card_names: List of card names to fetch

    Returns:
        Dictionary mapping card names to Card objects (or None if not found)

    Examples:
        >>> cards = bulk_fetch_cards(["Lightning Bolt", "Counterspell", "Giant Growth"])
        >>> for name, card in cards.items():
        ...     if card:
        ...         print(f"{name}: {card.oracle_text}")
    """
    results = {}

    for name in card_names:
        results[name] = _get_card_cached(name)

    return results


def prefetch_set_cards(set_code: str) -> int:
    """Prefetch all cards from a specific set to warm the cache.

    This is useful when you know you'll be working with a specific set
    (e.g., training on a single Limited format).

    Args:
        set_code: Three-letter set code (e.g., "MID", "NEO", "ONE")

    Returns:
        Number of cards successfully prefetched

    Examples:
        >>> count = prefetch_set_cards("MID")
        >>> print(f"Prefetched {count} cards from Midnight Hunt")
    """
    try:
        logger.info(f"Prefetching cards from set: {set_code}")
        cards = scryfallsdk.cards.Search(f"set:{set_code}")

        count = 0
        for card in cards:
            # Just access the card to populate the cache
            _get_card_cached(card.name)
            count += 1

            # Rate limit between fetches
            if count % 10 == 0:
                logger.debug(f"Prefetched {count} cards...")

        logger.info(f"Successfully prefetched {count} cards from {set_code}")
        return count

    except Exception as e:
        logger.error(f"Error prefetching set {set_code}: {e}")
        return 0


def get_cache_info() -> dict[str, Any]:
    """Get cache statistics for monitoring and debugging.

    Returns:
        Dictionary with cache statistics:
        - hits: Number of cache hits (reused data)
        - misses: Number of cache misses (new API calls)
        - errors: Number of API errors
        - size: Current cache size
        - maxsize: Maximum cache size

    Examples:
        >>> info = get_cache_info()
        >>> print(f"Cache hit rate: {info['hits'] / (info['hits'] + info['misses']):.2%}")
    """
    cache_info = _get_card_cached.cache_info()

    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "size": cache_info.currsize,
        "maxsize": cache_info.maxsize,
        "api_errors": _api_errors,
    }


def clear_cache() -> None:
    """Clear the card cache.

    Useful for testing or if you want to force fresh data from Scryfall.
    """
    global _cache_hits, _cache_misses, _api_errors

    _get_card_cached.cache_clear()
    _cache_hits = 0
    _cache_misses = 0
    _api_errors = 0
    logger.info("Scryfall cache cleared")


def get_full_card_info(card_name: str) -> dict[str, Any] | None:
    """Get comprehensive card information in a single call.

    Args:
        card_name: The name of the card

    Returns:
        Dictionary with card information, or None if card not found

    Examples:
        >>> info = get_full_card_info("Lightning Bolt")
        >>> if info:
        ...     print(f"{info['name']}: {info['oracle_text']}")
        ...     print(f"Keywords: {info['keywords']}")
    """
    card = _get_card_cached(card_name)

    if card is None:
        return None

    return {
        "name": card.name,
        "oracle_text": getattr(card, "oracle_text", None),
        "keywords": getattr(card, "keywords", []),
        "type_line": getattr(card, "type_line", ""),
        "mana_cost": getattr(card, "mana_cost", ""),
        "cmc": getattr(card, "cmc", 0),
        "colors": getattr(card, "colors", []),
        "rarity": getattr(card, "rarity", "common"),
        "power": getattr(card, "power", None),
        "toughness": getattr(card, "toughness", None),
    }
