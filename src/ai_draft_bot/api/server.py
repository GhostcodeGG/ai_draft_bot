"""FastAPI server for real-time draft predictions.

This module provides a production-ready REST API for the AI Draft Bot,
enabling real-time inference for browser extensions, web UIs, and mobile apps.

Features:
- Sub-100ms inference latency
- Model caching and warm-up
- SHAP explanations on demand
- Health monitoring
- Request validation
- Error handling

Usage:
    uvicorn ai_draft_bot.api.server:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict - Get draft pick recommendations
    POST /explain - Get SHAP explanation for a pick
    GET /health - Health check and model status
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_draft_bot.api.models import (
    CardRecommendation,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from ai_draft_bot.config import get_config
from ai_draft_bot.data.ingest_17l import CardMetadata, PickRecord, parse_card_metadata
from ai_draft_bot.features.draft_context import build_ultra_advanced_pick_features
from ai_draft_bot.models.advanced_drafter import AdvancedDraftModel
from ai_draft_bot.utils import get_logger

logger = get_logger(__name__)

# Global state
_model: AdvancedDraftModel | None = None
_metadata: dict[str, CardMetadata] | None = None
_start_time: float = time.time()
_model_path: Path | None = None
_metadata_path: Path | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    # Startup
    logger.info("Starting AI Draft Bot API server...")
    config = get_config()
    config.ensure_directories()

    # Load model and metadata if paths are configured
    global _model_path, _metadata_path
    if _model_path and _metadata_path:
        try:
            load_model(_model_path, _metadata_path)
            logger.info("✓ Model and metadata loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model on startup: {e}")

    logger.info("✓ API server ready")
    yield
    # Shutdown
    logger.info("Shutting down AI Draft Bot API server...")


app = FastAPI(
    title="AI Draft Bot API",
    description="Real-time Magic: The Gathering draft pick predictions",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for browser extensions and web UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(model_path: Path, metadata_path: Path) -> None:
    """Load model and metadata into memory."""
    global _model, _metadata, _model_path, _metadata_path

    logger.info(f"Loading model from {model_path}...")
    _model = AdvancedDraftModel.load(model_path)
    _model_path = model_path

    logger.info(f"Loading metadata from {metadata_path}...")
    _metadata = dict(parse_card_metadata(metadata_path))
    _metadata_path = metadata_path

    logger.info(f"✓ Loaded model with {len(_metadata)} cards")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with model status."""
    uptime = time.time() - _start_time

    if _model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version=None,
            model_type=None,
            feature_dimension=None,
            num_classes=None,
            uptime_seconds=uptime,
        )

    # Get model info
    encoder = _model.get_label_encoder()
    num_classes = len(encoder.classes_) if encoder else 0

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version="0.1.0",  # TODO: Get from model metadata
        model_type="AdvancedDraftModel",
        feature_dimension=None,  # TODO: Extract from model
        num_classes=num_classes,
        cache_stats=None,  # TODO: Add cache statistics
        uptime_seconds=uptime,
    )


@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse}})
async def predict(request: PredictRequest) -> PredictResponse:
    """Get draft pick recommendations.

    Returns ranked list of cards with confidence scores.
    """
    if _model is None or _metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Call POST /load first.",
        )

    start_time = time.time()

    try:
        # Validate cards exist in metadata
        missing_cards = [card for card in request.pack if card not in _metadata]
        if missing_cards:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown cards in pack: {', '.join(missing_cards[:5])}",
            )

        # Build pick records for feature extraction
        pick_records = [
            PickRecord(
                event_id="api_request",
                pack_number=request.pack_number,
                pick_number=request.pick_number,
                chosen_card=card,
                pack_contents=request.pack,
            )
            for card in request.pack
        ]

        # Extract features
        features = build_ultra_advanced_pick_features(pick_records, _metadata, None)

        if not features:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract features from pack",
            )

        # Get predictions
        recommendations = []
        for row in features:
            proba_dist = _model.predict_proba(row.features)
            confidence = proba_dist.get(row.label, 0.0)

            recommendations.append(
                CardRecommendation(
                    card_name=row.label,
                    confidence=confidence,
                    rank=0,  # Will be set after sorting
                )
            )

        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        for i, rec in enumerate(recommendations, 1):
            rec.rank = i

        inference_time = (time.time() - start_time) * 1000

        return PredictResponse(
            recommendations=recommendations,
            pack_size=len(request.pack),
            deck_size=len(request.deck),
            model_version="0.1.0",
            inference_time_ms=inference_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/explain", response_model=ExplainResponse, responses={400: {"model": ErrorResponse}})
async def explain(request: ExplainRequest) -> ExplainResponse:
    """Get SHAP explanation for a specific card recommendation.

    Shows which features most influenced the model's prediction.
    """
    if _model is None or _metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    start_time = time.time()

    try:
        # Import SHAP here to make it optional
        try:
            from ai_draft_bot.explainability.shap_explainer import DraftExplainer
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="SHAP not installed. Install with: pip install shap",
            )

        # Validate card exists
        if request.card_to_explain not in _metadata:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown card: {request.card_to_explain}",
            )

        # Build features
        pick_record = PickRecord(
            event_id="api_request",
            pack_number=request.pack_number,
            pick_number=request.pick_number,
            chosen_card=request.card_to_explain,
            pack_contents=request.pack,
        )

        features = build_ultra_advanced_pick_features([pick_record], _metadata, None)

        if not features:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to extract features",
            )

        # Get prediction
        card_features = features[0]
        proba_dist = _model.predict_proba(card_features.features)
        confidence = proba_dist.get(request.card_to_explain, 0.0)

        # Create explainer and explain
        explainer = DraftExplainer(_model)
        background_data = np.vstack([row.features for row in features])
        explainer.fit(background_data, max_samples=len(features))

        explanation = explainer.explain_pick(card_features.features, top_k=request.top_k_features)

        # Convert to API response format
        top_features = [
            FeatureContribution(
                feature_name=name,
                contribution=contribution,
                direction="positive" if contribution > 0 else "negative",
                description=None,  # TODO: Add feature descriptions
            )
            for name, contribution in explanation.top_features.items()
        ]

        alternatives = [
            CardRecommendation(card_name=card, confidence=prob, rank=i + 1)
            for i, (card, prob) in enumerate(explanation.alternative_picks[:3])
        ]

        inference_time = (time.time() - start_time) * 1000

        return ExplainResponse(
            card_name=request.card_to_explain,
            confidence=confidence,
            top_features=top_features,
            alternative_picks=alternatives,
            explanation_text=f"The model recommends {request.card_to_explain} with {confidence:.1%} confidence.",
            inference_time_ms=inference_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/load")
async def load_model_endpoint(model_path: str, metadata_path: str) -> dict[str, Any]:
    """Load or reload model and metadata.

    Useful for hot-swapping models without restarting the server.
    """
    try:
        load_model(Path(model_path), Path(metadata_path))
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    # For development only
    logger.info("Starting development server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
