"""Command line entrypoints for ai-draft-bot."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
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


@app.command()
def explain(
    model_path: Path = typer.Option(..., help="Path to a trained model artifact"),
    metadata_path: Path = typer.Option(..., help="Path to a card metadata CSV export"),
    pack: List[str] = typer.Option(..., help="Card names in the pack (use multiple --pack flags)"),
    model_kind: ModelKind = typer.Option(ModelKind.ADVANCED, help="Model type to load"),
    feature_set: FeatureSet = typer.Option(FeatureSet.ADVANCED, help="Feature set to use"),
    top_k: int = typer.Option(5, help="Number of top features to show"),
) -> None:
    """Explain why the model recommends specific picks using SHAP values.

    This command shows which features most influenced the model's decision,
    helping you understand what the model values in draft picks.

    Example:
        python -m ai_draft_bot.cli explain \\
            --model-path artifacts/advanced_model.joblib \\
            --metadata-path data/cards.csv \\
            --pack "Lightning Bolt" --pack "Counterspell" --pack "Llanowar Elves"
    """
    try:
        from ai_draft_bot.explainability.shap_explainer import DraftExplainer
    except ImportError:
        typer.echo("Error: SHAP is not installed. Install with: pip install shap")
        raise typer.Exit(code=1)

    metadata = _load_metadata(metadata_path)
    model = _load_model(model_kind, model_path)

    typer.echo(f"Explaining picks from pack: {', '.join(pack)}")

    # Build features for each card in the pack
    features = _build_features(pack, metadata, feature_set, [])

    if not features:
        typer.echo("Could not build features for any cards.")
        raise typer.Exit(code=1)

    # Get prediction
    scored = [
        (row.label, model.predict_proba(row.features).get(row.label, 0.0))
        for row in features
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    recommended_card, confidence = scored[0]

    typer.echo(f"\n✨ Recommended pick: {recommended_card} (confidence: {confidence:.1%})")

    # Get features for the recommended card
    recommended_features = next(row for row in features if row.label == recommended_card)

    # Create and fit explainer
    typer.echo("\nFitting SHAP explainer (this may take a moment)...")
    explainer = DraftExplainer(model, feature_names=None)

    # Use all pack features as background data
    background_data = np.vstack([row.features for row in features])
    try:
        explainer.fit(background_data, max_samples=len(features))
    except Exception as e:
        typer.echo(f"Warning: Could not fit SHAP explainer: {e}")
        typer.echo("This may be because SHAP only supports tree-based models.")
        raise typer.Exit(code=1)

    # Explain the pick
    try:
        explanation = explainer.explain_pick(recommended_features.features, top_k=top_k)

        typer.echo(f"\nTop {top_k} features influencing this pick:")
        for i, (feature_name, contribution) in enumerate(explanation.top_features.items(), 1):
            direction = "↑" if contribution > 0 else "↓"
            typer.echo(f"  {i}. {feature_name}: {direction} {abs(contribution):.4f}")

        if explanation.alternative_picks:
            typer.echo("\nAlternative picks:")
            for alt_card, prob in explanation.alternative_picks[:3]:
                typer.echo(f"  {alt_card}: {prob:.1%}")

    except Exception as e:
        typer.echo(f"Error generating explanation: {e}")
        raise typer.Exit(code=1)

    typer.echo("\n✓ Explanation complete!")


if __name__ == "__main__":
    app()
