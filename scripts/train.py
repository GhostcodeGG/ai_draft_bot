"""Train the baseline drafter model."""

from __future__ import annotations

from pathlib import Path

import typer

from ai_draft_bot.data.ingest_17l import parse_card_metadata, parse_draft_logs
from ai_draft_bot.features.draft_context import build_pick_features
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
    result.model.artifacts.model.classes_  # ensure model is fitted before saving
    result.model.save(output_path)

    typer.echo(
        "Training complete: accuracy={:.3f} (train={}, val={})".format(
            result.metrics.accuracy,
            result.metrics.train_samples,
            result.metrics.validation_samples,
        )
    )
    typer.echo(f"Model saved to {output_path}")


if __name__ == "__main__":
    app()
