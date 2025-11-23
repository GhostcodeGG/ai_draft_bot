"""Performance benchmarking for AI Draft Bot models.

This script measures and compares:
- Inference speed (picks/second)
- Memory usage (MB)
- Accuracy metrics (top-1, top-3, NDCG@3)
- Training time

Usage:
    python scripts/benchmark.py \\
        --model-path artifacts/model.joblib \\
        --test-data test_picks.jsonl \\
        --metadata cards.csv
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import typer
from tabulate import tabulate

from ai_draft_bot.data.ingest_17l import parse_card_metadata, parse_draft_logs
from ai_draft_bot.features.draft_context import build_ultra_advanced_pick_features
from ai_draft_bot.models.advanced_drafter import AdvancedDraftModel
from ai_draft_bot.utils.metrics import ndcg_at_ks, topk_accuracies

app = typer.Typer(help="Benchmark model performance")


@dataclass
class BenchmarkResults:
    """Results from a single model benchmark."""

    model_name: str
    accuracy_top1: float
    accuracy_top3: float
    ndcg_at_3: float
    inference_time_ms: float
    throughput_picks_per_sec: float
    memory_mb: float
    feature_dim: int
    num_classes: int


def measure_memory() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**2


def benchmark_model(
    model: Any,
    test_features: list,
    model_name: str = "Model",
) -> BenchmarkResults:
    """Benchmark a single model's performance.

    Args:
        model: Trained model to benchmark
        test_features: List of PickFeatures for testing
        model_name: Name of the model for display

    Returns:
        BenchmarkResults with all metrics
    """
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Benchmarking: {model_name}")
    typer.echo(f"{'=' * 60}")

    # Extract features and labels
    X_test = np.vstack([row.features for row in test_features])
    y_test = [row.label for row in test_features]

    typer.echo(f"Test set: {len(X_test)} picks, {X_test.shape[1]} features")

    # Measure memory before inference
    mem_before = measure_memory()

    # Measure inference time
    typer.echo("Measuring inference speed...")
    start_time = time.time()

    predictions = []
    probabilities = []
    for features in X_test:
        pred = model.predict(features)
        proba = model.predict_proba(features)
        predictions.append(pred)
        probabilities.append(proba)

    elapsed = time.time() - start_time
    mem_after = measure_memory()

    # Calculate metrics
    throughput = len(X_test) / elapsed
    avg_latency_ms = (elapsed / len(X_test)) * 1000

    typer.echo(f"✓ Inference complete in {elapsed:.2f}s")
    typer.echo(f"  Throughput: {throughput:.0f} picks/sec")
    typer.echo(f"  Avg latency: {avg_latency_ms:.2f} ms/pick")
    typer.echo(f"  Memory delta: {mem_after - mem_before:.1f} MB")

    # Calculate accuracy metrics
    typer.echo("\nCalculating accuracy metrics...")

    # Convert probabilities to score matrix
    all_cards = model.get_label_encoder().classes_
    proba_matrix = np.zeros((len(probabilities), len(all_cards)))

    for i, proba_dict in enumerate(probabilities):
        for card, prob in proba_dict.items():
            if card in all_cards:
                card_idx = np.where(all_cards == card)[0][0]
                proba_matrix[i, card_idx] = prob

    # Calculate top-k accuracies
    topk_acc = topk_accuracies(proba_matrix, y_test, ks=[1, 3, 5])
    ndcg_scores = ndcg_at_ks(proba_matrix, y_test, ks=[3, 5, 10])

    typer.echo(f"✓ Metrics calculated")
    typer.echo(f"  Top-1 accuracy: {topk_acc[1]:.1%}")
    typer.echo(f"  Top-3 accuracy: {topk_acc[3]:.1%}")
    typer.echo(f"  NDCG@3: {ndcg_scores[3]:.3f}")

    return BenchmarkResults(
        model_name=model_name,
        accuracy_top1=topk_acc[1],
        accuracy_top3=topk_acc[3],
        ndcg_at_3=ndcg_scores[3],
        inference_time_ms=avg_latency_ms,
        throughput_picks_per_sec=throughput,
        memory_mb=mem_after - mem_before,
        feature_dim=X_test.shape[1],
        num_classes=len(all_cards),
    )


@app.command()
def run(
    model_path: Path = typer.Option(..., help="Path to trained model"),
    test_data: Path = typer.Option(..., help="Path to test JSONL data"),
    metadata_path: Path = typer.Option(..., help="Path to card metadata CSV"),
    model_name: str = typer.Option("Model", help="Name for display"),
    max_picks: int = typer.Option(1000, help="Maximum picks to use for benchmarking"),
) -> None:
    """Run performance benchmark on a trained model."""
    typer.echo("AI Draft Bot - Performance Benchmark")
    typer.echo("=" * 60)

    # Load test data
    typer.echo(f"\nLoading test data from {test_data}...")
    picks = list(parse_draft_logs(test_data))
    metadata = dict(parse_card_metadata(metadata_path))

    typer.echo(f"✓ Loaded {len(picks)} picks, {len(metadata)} cards")

    # Limit picks for faster benchmarking
    if len(picks) > max_picks:
        typer.echo(f"  Sampling {max_picks} picks for benchmarking...")
        import random

        random.seed(42)
        picks = random.sample(picks, max_picks)

    # Build features
    typer.echo("\nBuilding features...")
    features = build_ultra_advanced_pick_features(picks, metadata)
    typer.echo(f"✓ Built {len(features)} feature vectors")

    if not features:
        typer.echo("Error: No features built. Check data quality.")
        raise typer.Exit(code=1)

    # Load model
    typer.echo(f"\nLoading model from {model_path}...")
    model = AdvancedDraftModel.load(model_path)
    typer.echo(f"✓ Model loaded")

    # Run benchmark
    results = benchmark_model(model, features, model_name)

    # Display results table
    typer.echo("\n" + "=" * 60)
    typer.echo("BENCHMARK RESULTS")
    typer.echo("=" * 60)

    table_data = [
        ["Model", results.model_name],
        ["", ""],
        ["Accuracy Metrics", ""],
        ["  Top-1 Accuracy", f"{results.accuracy_top1:.2%}"],
        ["  Top-3 Accuracy", f"{results.accuracy_top3:.2%}"],
        ["  NDCG@3", f"{results.ndcg_at_3:.4f}"],
        ["", ""],
        ["Performance", ""],
        ["  Throughput", f"{results.throughput_picks_per_sec:.0f} picks/sec"],
        ["  Avg Latency", f"{results.inference_time_ms:.2f} ms/pick"],
        ["  Memory Usage", f"{results.memory_mb:.1f} MB"],
        ["", ""],
        ["Model Info", ""],
        ["  Feature Dimensions", str(results.feature_dim)],
        ["  Number of Classes", str(results.num_classes)],
    ]

    typer.echo(tabulate(table_data, tablefmt="simple"))
    typer.echo("\n✓ Benchmark complete!")


if __name__ == "__main__":
    app()
