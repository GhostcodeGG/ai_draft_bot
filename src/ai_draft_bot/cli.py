"""Command line entrypoints for ai-draft-bot."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List

import typer

from ai_draft_bot.data.ingest_17l import CardMetadata, PickRecord, parse_card_metadata
from ai_draft_bot.features.draft_context import (
    build_advanced_pick_features,
    build_pick_features,
    build_ultra_advanced_pick_features,
)
from ai_draft_bot.models.advanced_drafter import AdvancedDraftModel
from ai_draft_bot.models.drafter import DraftModel
from ai_draft_bot.models.ensemble_drafter import EnsembleDraftModel
from ai_draft_bot.models.neural_drafter import NeuralDraftModel

app = typer.Typer(help="Simulate draft picks and run quick evaluations")


def _load_metadata(path: Path) -> dict[str, CardMetadata]:
    typer.echo(f"Loading card metadata from {path}")
    return dict(parse_card_metadata(path))


class FeatureSet(str, Enum):
    BASELINE = "baseline"
    ADVANCED = "advanced"
    ULTRA = "ultra"


class ModelKind(str, Enum):
    BASELINE = "baseline"
    ADVANCED = "advanced"
    NEURAL = "neural"
    ENSEMBLE = "ensemble"


def _load_model(kind: ModelKind, model_path: Path):
    if kind == ModelKind.BASELINE:
        return DraftModel.load(model_path)
    if kind == ModelKind.ADVANCED:
        return AdvancedDraftModel.load(model_path)
    if kind == ModelKind.NEURAL:
        return NeuralDraftModel.load(model_path)
    if kind == ModelKind.ENSEMBLE:
        return EnsembleDraftModel.load(model_path)
    raise ValueError(f"Unsupported model kind: {kind}")


@app.command()
def simulate(
    model_path: Path = typer.Option(
        ..., help="Path to a trained model artifact"
    ),
    metadata_path: Path = typer.Option(..., help="Path to a card metadata CSV export"),
    pack: List[str] = typer.Option(..., help="Comma-separated card names in the pack"),
    feature_set: FeatureSet = typer.Option(
        FeatureSet.BASELINE,
        help="Feature set to use when building pack features (baseline/advanced/ultra)",
    ),
    model_kind: ModelKind = typer.Option(
        ModelKind.BASELINE,
        help="Model architecture to load (baseline/advanced/neural/ensemble)",
    ),
    archetype_config: Path | None = typer.Option(
        None, help="Archetype config JSON (used for ultra features)"
    ),
) -> None:
    """Score each card in a pack and display the top recommendation."""

    model = _load_model(model_kind, model_path)
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

    if feature_set == FeatureSet.BASELINE:
        features = build_pick_features(picks, metadata)
    elif feature_set == FeatureSet.ADVANCED:
        features = build_advanced_pick_features(picks, metadata)
    else:
        archetype_config_str = str(archetype_config) if archetype_config else None
        features = build_ultra_advanced_pick_features(picks, metadata, archetype_config_str)

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
