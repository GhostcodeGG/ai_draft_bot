"""Train drafter models (baseline, advanced, ultra, or optimized)."""

from __future__ import annotations

from pathlib import Path

import typer

from ai_draft_bot.data.ingest_17l import (
    parse_card_metadata,
    parse_draft_logs,
    validate_card_coverage,
)
from ai_draft_bot.features.draft_context import (
    build_advanced_pick_features,
    build_pick_features,
    build_ultra_advanced_pick_features,
)
from ai_draft_bot.models.advanced_drafter import (
    AdvancedTrainConfig,
    ModelType,
    train_advanced_model,
)
from ai_draft_bot.models.drafter import TrainConfig, train_model
from ai_draft_bot.optimization.optuna_tuner import OptunaConfig, optimize_and_train
from ai_draft_bot.utils.logging_config import setup_logging

app = typer.Typer(help="Train models from 17L draft exports.")


def _warn_missing_metadata(picks, metadata) -> None:
    missing_cards = validate_card_coverage(picks, metadata)
    if missing_cards and len(missing_cards) > len(metadata) * 0.5:
        typer.echo(
            f"[WARN] {len(missing_cards)} cards missing from metadata (>50% of cards).\n"
            "       Draft logs and metadata may be from different sets."
        )


@app.command()
def run(
    drafts_path: Path = typer.Option(..., help="Path to a 17L JSONL draft export"),
    metadata_path: Path = typer.Option(..., help="Path to a 17L card metadata CSV"),
    output_path: Path = typer.Option("artifacts/model.joblib", help="Where to store the model"),
    test_size: float = typer.Option(0.2, help="Validation split fraction"),
    max_iter: int = typer.Option(500, help="Max iterations for logistic regression"),
    c: float = typer.Option(1.0, help="Inverse regularization strength"),
    random_state: int = typer.Option(13, help="Deterministic split for reproducibility"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
) -> None:
    """Train the baseline drafter (logistic regression, 16 features)."""
    setup_logging(level=log_level)

    typer.echo("Loading datasets.")
    picks = parse_draft_logs(drafts_path)
    metadata = parse_card_metadata(metadata_path)

    if not picks:
        typer.echo(
            "[ERROR] No picks loaded from draft log.\n"
            f"        Check that {drafts_path} is a valid JSONL file."
        )
        raise typer.Exit(code=1)

    _warn_missing_metadata(picks, metadata)

    typer.echo(f"Loaded {len(picks)} picks; building baseline features (16 dims).")
    rows = build_pick_features(picks, metadata)
    if not rows:
        typer.echo(
            "[ERROR] No features built.\n"
            "        - Card names may not match the metadata CSV\n"
            "        - Packs may be empty or missing metadata rows"
        )
        raise typer.Exit(code=1)

    config = TrainConfig(
        test_size=test_size,
        max_iter=max_iter,
        C=c,
        random_state=random_state,
    )
    result = train_model(rows, config=config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
) -> None:
    """Train an ADVANCED drafter with 78 features."""
    setup_logging(level=log_level)

    typer.echo(f"Training ADVANCED model with {model_type.value}...")
    typer.echo("Loading datasets.")
    picks = parse_draft_logs(drafts_path)
    metadata = parse_card_metadata(metadata_path)

    if not picks:
        typer.echo("[ERROR] No picks loaded from draft log.")
        raise typer.Exit(code=1)

    _warn_missing_metadata(picks, metadata)

    typer.echo(f"Loaded {len(picks)} picks")
    typer.echo("Building ADVANCED features (78 dims)...")
    rows = build_advanced_pick_features(picks, metadata)

    if not rows:
        typer.echo(
            "[ERROR] No features built.\n"
            "        - Card names may not match the metadata CSV\n"
            "        - Packs may be empty or missing metadata rows\n"
            "        - Win rate columns may be missing for advanced features"
        )
        raise typer.Exit(code=1)

    typer.echo(f"Built {len(rows)} feature vectors")
    typer.echo(f"Feature dimensionality: {rows[0].features.shape[0]} features")

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

    typer.echo("Training model...")
    result = train_advanced_model(rows, config=config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.model.save(output_path)

    typer.echo("=" * 60)
    typer.echo("ADVANCED TRAINING COMPLETE")
    typer.echo(
        "Validation Accuracy: {:.3f} ({:.1f}%)".format(
            result.metrics.accuracy, result.metrics.accuracy * 100
        )
    )
    typer.echo(f"Training samples: {result.metrics.train_samples}")
    typer.echo(f"Validation samples: {result.metrics.validation_samples}")

    if result.metrics.feature_importance:
        typer.echo("\nTop 10 Most Important Features:")
        sorted_feats = sorted(
            result.metrics.feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for feat_idx, importance in sorted_feats:
            typer.echo(f"  Feature {feat_idx}: {importance:.2f}")

    typer.echo(f"\nModel saved to {output_path}")


@app.command()
def ultra(
    drafts_path: Path = typer.Option(..., help="Path to a 17L JSONL draft export"),
    metadata_path: Path = typer.Option(..., help="Path to a 17L card metadata CSV"),
    output_path: Path = typer.Option(
        "artifacts/ultra_model.joblib", help="Where to store the model"
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
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
    archetype_config: Path | None = typer.Option(
        None, help="Archetype config JSON (used for ultra features)"
    ),
) -> None:
    """Train an ULTRA drafter with 130+ features."""
    setup_logging(level=log_level)

    typer.echo("Training ULTRA model...")
    typer.echo("Loading datasets.")
    picks = parse_draft_logs(drafts_path)
    metadata = parse_card_metadata(metadata_path)

    if not picks:
        typer.echo("[ERROR] No picks loaded from draft log.")
        raise typer.Exit(code=1)

    _warn_missing_metadata(picks, metadata)

    typer.echo(f"Loaded {len(picks)} picks")
    typer.echo("Building ULTRA features (130+ dims)...")
    archetype_config_str = str(archetype_config) if archetype_config else None
    rows = build_ultra_advanced_pick_features(picks, metadata, archetype_config_str)

    if not rows:
        typer.echo(
            "[ERROR] No features built.\n"
            "        - Card names may not match the metadata CSV\n"
            "        - Packs may be empty or missing metadata rows\n"
            "        - Win rate columns may be missing for ultra features"
        )
        raise typer.Exit(code=1)

    typer.echo(f"Built {len(rows)} feature vectors")
    typer.echo(f"Feature dimensionality: {rows[0].features.shape[0]} features")

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

    typer.echo("Training ULTRA model...")
    result = train_advanced_model(rows, config=config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.model.save(output_path)

    typer.echo("=" * 70)
    typer.echo("ULTRA TRAINING COMPLETE")
    typer.echo("=" * 70)
    typer.echo(
        "Validation Accuracy: {:.3f} ({:.1f}%)".format(
            result.metrics.accuracy, result.metrics.accuracy * 100
        )
    )
    typer.echo(f"Training samples: {result.metrics.train_samples}")
    typer.echo(f"Validation samples: {result.metrics.validation_samples}")

    if result.metrics.feature_importance:
        typer.echo("\nTop 15 Most Important Features:")
        sorted_feats = sorted(
            result.metrics.feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:15]
        for feat_idx, importance in sorted_feats:
            typer.echo(f"  Feature {feat_idx}: {importance:.2f}")

    typer.echo(f"\nModel saved to {output_path}")
    typer.echo("\nExpected performance (typical):")
    typer.echo("  Baseline (16 features): ~40% accuracy")
    typer.echo("  Advanced (78 features): ~60-70% accuracy")
    typer.echo("  Ultra (130+ features): ~70-85% accuracy (target)")
    typer.echo("=" * 70)


@app.command()
def optimize(
    drafts_path: Path = typer.Option(..., help="Path to a 17L JSONL draft export"),
    metadata_path: Path = typer.Option(..., help="Path to a 17L card metadata CSV"),
    output_path: Path = typer.Option(
        "artifacts/optimized_model.joblib", help="Where to store the optimized model"
    ),
    model_type: ModelType = typer.Option(
        ModelType.XGBOOST, help="Model architecture to optimize"
    ),
    n_trials: int = typer.Option(100, help="Number of Optuna optimization trials"),
    test_size: float = typer.Option(0.2, help="Validation split fraction"),
    use_gpu: bool = typer.Option(False, help="Use GPU acceleration if available"),
    random_state: int = typer.Option(13, help="Random seed for reproducibility"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    use_ultra_features: bool = typer.Option(
        True, help="Use ultra features (130+) instead of advanced (78)"
    ),
    archetype_config: Path | None = typer.Option(
        None, help="Archetype config for ultra features"
    ),
) -> None:
    """Auto-tune hyperparameters with Optuna and train the final model."""
    setup_logging(level=log_level)

    typer.echo("=" * 70)
    typer.echo("AUTOMATIC HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    typer.echo("=" * 70)
    typer.echo(f"Model type: {model_type.value}")
    typer.echo(f"Optimization trials: {n_trials}")
    typer.echo(f"Using {'ULTRA' if use_ultra_features else 'ADVANCED'} features")
    typer.echo()

    typer.echo("Loading datasets.")
    picks = parse_draft_logs(drafts_path)
    metadata = parse_card_metadata(metadata_path)

    if not picks:
        typer.echo("[ERROR] No picks loaded from draft log.")
        raise typer.Exit(code=1)

    validate_card_coverage(picks, metadata)
    typer.echo(f"Loaded {len(picks)} picks")

    if use_ultra_features:
        typer.echo("Building ULTRA features (130+ dims)...")
        archetype_config_str = str(archetype_config) if archetype_config else None
        rows = build_ultra_advanced_pick_features(picks, metadata, archetype_config_str)
    else:
        typer.echo("Building ADVANCED features (78 dims)...")
        rows = build_advanced_pick_features(picks, metadata)

    if not rows:
        typer.echo("[ERROR] No features built.")
        raise typer.Exit(code=1)

    typer.echo(f"Built {len(rows)} feature vectors")
    typer.echo(f"Feature dimensionality: {rows[0].features.shape[0]} features")
    typer.echo()

    optuna_config = OptunaConfig(
        n_trials=n_trials,
        model_type=model_type,
        test_size=test_size,
        random_state=random_state,
        use_gpu=use_gpu,
    )

    typer.echo("Starting Bayesian hyperparameter search...")
    typer.echo("  This will try different combinations to find optimal parameters.")
    typer.echo()

    result = optimize_and_train(rows, optuna_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.model.save(output_path)

    typer.echo()
    typer.echo("=" * 70)
    typer.echo("OPTIMIZATION COMPLETE")
    typer.echo("=" * 70)
    typer.echo(
        f"Final Validation Accuracy: {result.metrics.accuracy:.3f} "
        f"({result.metrics.accuracy * 100:.1f}%)"
    )
    typer.echo(f"Training samples: {result.metrics.train_samples}")
    typer.echo(f"Validation samples: {result.metrics.validation_samples}")
    typer.echo(f"\nOptimized model saved to {output_path}")
    typer.echo("\nExpected +2-4% accuracy vs default settings.")
    typer.echo("=" * 70)


if __name__ == "__main__":
    app()
