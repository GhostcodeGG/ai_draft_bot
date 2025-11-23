"""CLI script to start the FastAPI server.

Usage:
    python scripts/serve.py \\
        --model-path artifacts/advanced_model.joblib \\
        --metadata-path data/cards.csv \\
        --port 8000
"""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

app = typer.Typer(help="Start the AI Draft Bot API server")


@app.command()
def serve(
    model_path: Path = typer.Option(..., help="Path to trained model (.joblib)"),
    metadata_path: Path = typer.Option(..., help="Path to card metadata CSV"),
    lstm_model_path: Path = typer.Option(None, help="Optional: Path to LSTM model (.pt)"),
    lstm_encoder_path: Path = typer.Option(None, help="Optional: Path to LSTM encoder (.joblib)"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    workers: int = typer.Option(1, help="Number of worker processes"),
) -> None:
    """Start the FastAPI server with the specified model.

    The server will load the model and metadata on startup and be ready
    to serve predictions via the REST API. Optionally load LSTM model
    for /predict/sequence endpoint.

    Example:
        # XGBoost only
        python scripts/serve.py \\
            --model-path artifacts/advanced_model.joblib \\
            --metadata-path data/cards.csv \\
            --port 8000

        # With LSTM support
        python scripts/serve.py \\
            --model-path artifacts/advanced_model.joblib \\
            --metadata-path data/cards.csv \\
            --lstm-model-path artifacts/lstm_model.pt \\
            --lstm-encoder-path artifacts/lstm_encoder.joblib \\
            --port 8000
    """
    typer.echo("=" * 60)
    typer.echo("AI Draft Bot API Server")
    typer.echo("=" * 60)
    typer.echo(f"Model: {model_path}")
    typer.echo(f"Metadata: {metadata_path}")
    if lstm_model_path:
        typer.echo(f"LSTM Model: {lstm_model_path}")
        typer.echo(f"LSTM Encoder: {lstm_encoder_path}")
    typer.echo(f"Listening on: http://{host}:{port}")
    typer.echo("=" * 60)

    # Validate paths exist
    if not model_path.exists():
        typer.echo(f"Error: Model not found at {model_path}", err=True)
        raise typer.Exit(code=1)

    if not metadata_path.exists():
        typer.echo(f"Error: Metadata not found at {metadata_path}", err=True)
        raise typer.Exit(code=1)

    # Validate LSTM paths if provided
    if lstm_model_path and not lstm_model_path.exists():
        typer.echo(f"Error: LSTM model not found at {lstm_model_path}", err=True)
        raise typer.Exit(code=1)

    if lstm_encoder_path and not lstm_encoder_path.exists():
        typer.echo(f"Error: LSTM encoder not found at {lstm_encoder_path}", err=True)
        raise typer.Exit(code=1)

    # Load model at startup
    from ai_draft_bot.api.server import load_model, load_lstm_model

    load_model(model_path, metadata_path)
    typer.echo("\n✓ XGBoost model loaded successfully")

    # Optionally load LSTM model
    if lstm_model_path and lstm_encoder_path:
        try:
            load_lstm_model(lstm_model_path, lstm_encoder_path)
            typer.echo("✓ LSTM model loaded successfully")
            typer.echo("  → /predict/sequence endpoint available")
        except Exception as e:
            typer.echo(f"⚠ Warning: LSTM model failed to load: {e}", err=True)
            typer.echo("  → /predict/sequence endpoint will not be available")

    typer.echo(f"\nStarting server with {workers} worker(s)...\n")

    # Start the server
    uvicorn.run(
        "ai_draft_bot.api.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Can't use workers with reload
        log_level="info",
    )


if __name__ == "__main__":
    app()
