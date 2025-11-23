"""Convert 17Lands public CSV draft data to JSONL format.

The public 17Lands datasets use a wide CSV format where each pack card
has its own column (pack_card_*). This script converts to the JSONL format
expected by our ingestion code.

Usage:
    python scripts/convert_17l_csv_to_jsonl.py \\
        --input "17L dataset/draft_data_public.EOE.PremierDraft.csv.gz" \\
        --output data/eoe_drafts.jsonl
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer(help="Convert 17Lands CSV to JSONL format")


@app.command()
def convert(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input CSV or CSV.GZ file"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output JSONL file"),
    limit: int = typer.Option(None, "--limit", help="Limit number of picks (for testing)"),
    chunk_size: int = typer.Option(50000, "--chunk-size", help="Process this many rows at a time"),
) -> None:
    """Convert 17Lands CSV draft data to JSONL format.

    The CSV format has:
    - One row per pick
    - pack_card_* columns (1 or 0 indicating if card is in pack)
    - pool_* columns (count of cards in pool)

    The JSONL format has:
    - One JSON object per pick
    - pack_contents: list of card names
    - pool: list of card names (with duplicates)
    """
    typer.echo(f"Reading {input_path} in chunks of {chunk_size:,} rows...")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass: get column names from first chunk
    typer.echo("Scanning for column names...")
    if str(input_path).endswith('.gz'):
        with gzip.open(input_path, 'rt') as f:
            first_chunk = pd.read_csv(f, nrows=1)
    else:
        first_chunk = pd.read_csv(input_path, nrows=1)

    pack_cols = [c for c in first_chunk.columns if c.startswith('pack_card_')]
    pool_cols = [c for c in first_chunk.columns if c.startswith('pool_')]

    typer.echo(f"Found {len(pack_cols)} possible pack cards")
    typer.echo(f"Found {len(pool_cols)} possible pool cards")

    # Process in chunks to avoid memory issues
    typer.echo(f"\nConverting to JSONL format...")
    total_picks = 0
    chunk_num = 0

    # Open output file
    with open(output_path, 'w', encoding='utf-8') as out:
        # Read CSV in chunks
        if str(input_path).endswith('.gz'):
            with gzip.open(input_path, 'rt') as f:
                chunks = pd.read_csv(f, chunksize=chunk_size)
                for df in chunks:
                    if limit and total_picks >= limit:
                        break

                    chunk_num += 1
                    typer.echo(f"Processing chunk {chunk_num}: {len(df):,} picks...")

                    for idx, row in df.iterrows():
                        if limit and total_picks >= limit:
                            break

                        # Extract pack contents (cards where pack_card_* == 1)
                        pack_contents = []
                        for col in pack_cols:
                            if row[col] == 1:
                                card_name = col.replace('pack_card_', '')
                                pack_contents.append(card_name)

                        # Extract pool (cards where pool_* > 0)
                        pool = []
                        for col in pool_cols:
                            count = row[col]
                            if count > 0:
                                card_name = col.replace('pool_', '')
                                pool.extend([card_name] * int(count))

                        # Create pick record
                        pick_record = {
                            "event_id": str(row['draft_id']),
                            "pack_number": int(row['pack_number']),
                            "pick_number": int(row['pick_number']),
                            "chosen_card": str(row['pick']),
                            "pack_contents": pack_contents,
                            "pool": pool,
                            "expansion": str(row['expansion']),
                            "rank": str(row.get('rank', '')),
                        }

                        out.write(json.dumps(pick_record) + '\n')
                        total_picks += 1
        else:
            chunks = pd.read_csv(input_path, chunksize=chunk_size)
            for df in chunks:
                if limit and total_picks >= limit:
                    break

                chunk_num += 1
                typer.echo(f"Processing chunk {chunk_num}: {len(df):,} picks...")

                for idx, row in df.iterrows():
                    if limit and total_picks >= limit:
                        break

                    # Extract pack contents
                    pack_contents = []
                    for col in pack_cols:
                        if row[col] == 1:
                            card_name = col.replace('pack_card_', '')
                            pack_contents.append(card_name)

                    # Extract pool
                    pool = []
                    for col in pool_cols:
                        count = row[col]
                        if count > 0:
                            card_name = col.replace('pool_', '')
                            pool.extend([card_name] * int(count))

                    pick_record = {
                        "event_id": str(row['draft_id']),
                        "pack_number": int(row['pack_number']),
                        "pick_number": int(row['pick_number']),
                        "chosen_card": str(row['pick']),
                        "pack_contents": pack_contents,
                        "pool": pool,
                        "expansion": str(row['expansion']),
                        "rank": str(row.get('rank', '')),
                    }

                    out.write(json.dumps(pick_record) + '\n')
                    total_picks += 1

    typer.echo(f"\nConverted {total_picks:,} picks to {output_path}")
    typer.echo(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


@app.command()
def extract_metadata(
    draft_csv: Path = typer.Option(..., help="Draft CSV file"),
    output_csv: Path = typer.Option(..., help="Output card metadata CSV"),
) -> None:
    """Extract basic card metadata from draft CSV.

    Note: This creates a minimal metadata file without win rates.
    For full metadata with win rates, download from 17Lands directly.
    """
    typer.echo(f"Reading {draft_csv}...")

    if str(draft_csv).endswith('.gz'):
        with gzip.open(draft_csv, 'rt') as f:
            df = pd.read_csv(f, nrows=1000)  # Sample for card list
    else:
        df = pd.read_csv(draft_csv, nrows=1000)

    # Get all unique card names from pack_card columns
    pack_cols = [c for c in df.columns if c.startswith('pack_card_')]
    cards = [c.replace('pack_card_', '') for c in pack_cols]

    typer.echo(f"Found {len(cards)} unique cards")

    # Create minimal metadata
    metadata = pd.DataFrame({
        'name': cards,
        'color': 'unknown',  # Would need Scryfall to fill
        'rarity': 'common',
        'type_line': 'Unknown',
        'mana_value': 3.0,
    })

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(output_csv, index=False)
    typer.echo(f"✓ Created metadata for {len(cards)} cards at {output_csv}")
    typer.echo(f"\n⚠ This metadata is minimal - missing win rates and card details!")
    typer.echo(f"  For better results, download card ratings CSV from 17Lands")


if __name__ == "__main__":
    app()
