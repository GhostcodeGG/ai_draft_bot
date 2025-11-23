"""Compute card statistics (win rates) from 17Lands game data.

This script calculates:
- GIH WR (Games in Hand Win Rate): Win% when card is in hand during game
- OH WR (Opening Hand Win Rate): Win% when card is in opening hand
- GD WR (Game Draw Win Rate): Win% when card is drawn during game
- IWD (Improvement When Drawn): Percentage point improvement
- Games Played: Number of games where card was in deck

Usage:
    python scripts/compute_card_stats.py \\
        --game-data "17L dataset/game_data_public.EOE.PremierDraft.csv.gz" \\
        --output data/eoe_card_stats.csv
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer(help="Compute card statistics from game data")


@app.command()
def compute(
    game_data: Path = typer.Option(..., help="Path to game data CSV/CSV.GZ"),
    output: Path = typer.Option(..., help="Output CSV with card statistics"),
    sample: int = typer.Option(None, help="Sample N games for testing"),
) -> None:
    """Compute win rate statistics from game data."""

    typer.echo(f"Loading game data from {game_data}...")

    # Load game data
    if str(game_data).endswith('.gz'):
        with gzip.open(game_data, 'rt') as f:
            if sample:
                df = pd.read_csv(f, nrows=sample)
                typer.echo(f"Sampled {sample} games for testing")
            else:
                df = pd.read_csv(f)
    else:
        if sample:
            df = pd.read_csv(game_data, nrows=sample)
        else:
            df = pd.read_csv(game_data)

    typer.echo(f"Loaded {len(df)} games")

    # Get card columns
    deck_cols = [c for c in df.columns if c.startswith('deck_')]
    drawn_cols = [c for c in df.columns if c.startswith('drawn_')]
    opening_cols = [c for c in df.columns if c.startswith('opening_hand_')]

    typer.echo(f"Found {len(deck_cols)} unique cards")

    # Compute statistics for each card
    card_stats = []

    typer.echo("\nComputing win rates...")
    for deck_col in tqdm(deck_cols, desc="Processing cards"):
        card_name = deck_col.replace('deck_', '')
        drawn_col = f'drawn_{card_name}'
        opening_col = f'opening_hand_{card_name}'

        # Filter to games where card was in deck
        in_deck = df[deck_col] > 0

        if in_deck.sum() == 0:
            continue  # Skip cards never played

        games_in_deck = df[in_deck]

        # GIH WR: Games where card was in hand (opening or drawn)
        in_hand = (games_in_deck[opening_col] > 0) | (games_in_deck[drawn_col] > 0)
        gih_games = games_in_deck[in_hand]
        gih_wr = gih_games['won'].mean() if len(gih_games) > 0 else None

        # OH WR: Games where card was in opening hand
        in_opening = games_in_deck[opening_col] > 0
        oh_games = games_in_deck[in_opening]
        oh_wr = oh_games['won'].mean() if len(oh_games) > 0 else None

        # GD WR: Games where card was drawn (not in opening hand)
        drawn_only = (games_in_deck[drawn_col] > 0) & (games_in_deck[opening_col] == 0)
        gd_games = games_in_deck[drawn_only]
        gd_wr = gd_games['won'].mean() if len(gd_games) > 0 else None

        # IWD: Improvement when drawn (GIH WR - deck win rate without card)
        not_in_hand = ~in_hand
        deck_without_games = games_in_deck[not_in_hand]
        deck_wr = deck_without_games['won'].mean() if len(deck_without_games) > 0 else None
        iwd = (gih_wr - deck_wr) if (gih_wr is not None and deck_wr is not None) else None

        card_stats.append({
            'name': card_name,
            'games_in_deck': in_deck.sum(),
            'gih_games': len(gih_games),
            'gih_wr': gih_wr,
            'oh_wr': oh_wr,
            'gd_wr': gd_wr,
            'iwd': iwd,
            'deck_wr': deck_wr,
        })

    # Create DataFrame
    stats_df = pd.DataFrame(card_stats)

    # Add basic card info (we'll need to get this from Scryfall or infer)
    stats_df['color'] = 'unknown'  # Will be filled by Scryfall
    stats_df['rarity'] = 'common'  # Will be filled by Scryfall
    stats_df['type_line'] = 'Unknown'  # Will be filled by Scryfall
    stats_df['mana_value'] = 3.0  # Will be filled by Scryfall

    # Reorder columns to match expected format
    stats_df = stats_df[['name', 'color', 'rarity', 'type_line', 'mana_value',
                         'gih_wr', 'oh_wr', 'gd_wr', 'iwd', 'games_in_deck']]

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save
    stats_df.to_csv(output, index=False)

    typer.echo(f"\n✓ Computed statistics for {len(stats_df)} cards")
    typer.echo(f"  Saved to: {output}")
    typer.echo(f"\nSummary:")
    typer.echo(f"  Average GIH WR: {stats_df['gih_wr'].mean():.3f}")
    typer.echo(f"  Cards with 100+ games: {(stats_df['games_in_deck'] >= 100).sum()}")
    typer.echo(f"  Cards with 1000+ games: {(stats_df['games_in_deck'] >= 1000).sum()}")

    typer.echo(f"\n⚠ Note: color, rarity, type_line, mana_value are placeholders")
    typer.echo(f"  Run Scryfall cache to fill in real card details")


if __name__ == "__main__":
    app()
