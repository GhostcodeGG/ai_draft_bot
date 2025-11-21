"""Train drafter models (baseline or advanced)."""

from __future__ import annotations

from pathlib import Path

import typer

from ai_draft_bot.data.ingest_17l import parse_card_metadata, parse_draft_logs
from ai_draft_bot.features.draft_context import build_advanced_pick_features, build_pick_features
from ai_draft_bot.models.advanced_drafter import (
    AdvancedTrainConfig,
    ModelType,
    train_advanced_model,
)
from ai_draft_bot.models.drafter import TrainConfig, train_model

app = typer.Typer(help="Train models from 17L exports")


@app.command()
def run(
    drafts_path: Path = typer.Option(..., help="Path to a 17L JSONL draft export"),
    metadata_path: Path = typer.Option(..., help="Path to a 17L card metadata CSV"),
    output_path: Path = typer.Option("artifacts/model.joblib", help="Where to store the model"),
    test_size: float = typer.Option(0.2, help="Validation split fraction"),
    max_iter: int = typer.Option(500, help="Max iterations for logistic regression"),
    c: float = typer.Option(1.0, help="Inverse regularization strength"),
    random_state: int = typer.Option(13, help="Deterministic split for reproducibility"),
) -> None:
    """Train the baseline drafter."""

    typer.echo("Loading datasets…")
    picks = parse_draft_logs(drafts_path)
    metadata = parse_card_metadata(metadata_path)

    if not picks:
        typer.echo("No picks loaded from draft log. Aborting.")
        raise typer.Exit(code=1)

    typer.echo(f"Loaded {len(picks)} picks; building features…")
    rows = build_pick_features(picks, metadata)
    if not rows:
        typer.echo("No features built; ensure card names align between logs and metadata.")
        raise typer.Exit(code=1)

    config = TrainConfig(
        test_size=test_size,
        max_iter=max_iter,
        C=c,
        random_state=random_state,
    )
    result = train_model(rows, config=config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = result.model.artifacts.model.classes_  # ensure model is fitted before saving
    result.model.save(output_path)

    typer.echo(
        "Training complete: accuracy={:.3f} (train={}, val={})".format(
            result.metrics.accuracy,
            result.metrics.train_samples,
            result.metrics.validation_samples,
        )
    )
    typer.echo(f"Model saved to {output_path}")


@app.command()
def advanced(
    drafts_path: Path = typer.Option(..., help="Path to a 17L JSONL draft export"),
    metadata_path: Path = typer.Option(..., help="Path to a 17L card metadata CSV"),
    output_path: Path = typer.Option(
        "artifacts/advanced_model.joblib", help="Where to store the model"
    ),
    model_type: ModelType = typer.Option(
        ModelType.XGBOOST, help="Model architecture (xgboost, lightgbm)"
    ),
    test_size: float = typer.Option(0.2, help="Validation split fraction"),
    n_estimators: int = typer.Option(500, help="Number of boosting rounds"),
    max_depth: int = typer.Option(8, help="Maximum tree depth"),
    learning_rate: float = typer.Option(0.1, help="Learning rate"),
    early_stopping: int = typer.Option(50, help="Early stopping rounds"),
    use_gpu: bool = typer.Option(False, help="Use GPU acceleration if available"),
    random_state: int = typer.Option(13, help="Random seed for reproducibility"),
) -> None:
    """Train an ADVANCED drafter with XGBoost/LightGBM and 75+ features.

    This uses:
    - Win rate statistics (GIH WR, IWD, ALSA)
    - Draft context (pick/pack number, cards picked so far)
    - Deck composition (mana curve, color commitment, creature count)
    - Synergy features (color/curve/archetype fit)
    - Pack signals (bombs, rares, win rate distribution)
    """
    typer.echo(f"🚀 Training ADVANCED model with {model_type.value}...")
    typer.echo("Loading datasets…")
    picks = parse_draft_logs(drafts_path)
    metadata = parse_card_metadata(metadata_path)

    if not picks:
        typer.echo("❌ No picks loaded from draft log. Aborting.")
        raise typer.Exit(code=1)

    typer.echo(f"✓ Loaded {len(picks)} picks")
    typer.echo("🔧 Building ADVANCED features (75+ dimensions)...")
    rows = build_advanced_pick_features(picks, metadata)

    if not rows:
        typer.echo("❌ No features built; ensure card names align between logs and metadata.")
        raise typer.Exit(code=1)

    typer.echo(f"✓ Built {len(rows)} feature vectors")
    typer.echo(f"📊 Feature dimensionality: {rows[0].features.shape[0]} features")

    config = AdvancedTrainConfig(
        model_type=model_type,
        test_size=test_size,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping,
        use_gpu=use_gpu,
        random_state=random_state,
    )

    typer.echo("🏋️ Training model...")
    result = train_advanced_model(rows, config=config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.model.save(output_path)

    typer.echo("=" * 60)
    typer.echo("🎉 TRAINING COMPLETE!")
    typer.echo(
        "📈 Validation Accuracy: {:.3f} ({:.1f}%)".format(
            result.metrics.accuracy, result.metrics.accuracy * 100
        )
    )
    typer.echo(f"📦 Training samples: {result.metrics.train_samples}")
    typer.echo(f"📦 Validation samples: {result.metrics.validation_samples}")

    # Show top feature importance
    if result.metrics.feature_importance:
        typer.echo("\n🔝 Top 10 Most Important Features:")
        sorted_feats = sorted(
            result.metrics.feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for feat_idx, importance in sorted_feats:
            typer.echo(f"   Feature {feat_idx}: {importance:.2f}")

    typer.echo(f"\n💾 Model saved to {output_path}")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
