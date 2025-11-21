"""Command line entrypoints for ai-draft-bot."""

from __future__ import annotations

from pathlib import Path
from typing import List

import typer

from ai_draft_bot.data.ingest_17l import CardMetadata, PickRecord, parse_card_metadata
from ai_draft_bot.features.draft_context import build_pick_features
from ai_draft_bot.models.drafter import DraftModel

app = typer.Typer(help="Simulate draft picks and run quick evaluations")


def _load_metadata(path: Path) -> dict[str, CardMetadata]:
    typer.echo(f"Loading card metadata from {path}")
    return dict(parse_card_metadata(path))


@app.command()
def simulate(
    model_path: Path = typer.Option(..., help="Path to a trained model artifact produced by train.py"),
    metadata_path: Path = typer.Option(..., help="Path to a card metadata CSV export"),
    pack: List[str] = typer.Option(..., help="Comma-separated card names in the pack"),
) -> None:
    """Score each card in a pack and display the top recommendation."""

    model = DraftModel.load(model_path)
    metadata = _load_metadata(metadata_path)

    pack_cards = [name.strip() for name in pack]
    picks = [
        PickRecord(
            event_id="sim",
            pack_number=1,
            pick_number=1,
            chosen_card=candidate,
            pack_contents=pack_cards,
        )
        for candidate in pack_cards
    ]
    features = build_pick_features(picks, metadata)

    if not features:
        typer.echo("Could not build features for any cards. Check card names and metadata.")
        raise typer.Exit(code=1)

    scored = [
        (row.label, model.predict_proba(row.features).get(row.label, 0.0))
        for row in features
    ]
    scored.sort(key=lambda item: item[1], reverse=True)

    typer.echo("Top recommendations:")
    for label, score in scored:
        typer.echo(f"  {label}: {score:.3f}")


if __name__ == "__main__":
    app()
