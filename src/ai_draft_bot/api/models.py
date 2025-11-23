"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request for draft pick prediction."""

    pack: list[str] = Field(..., description="List of card names in the current pack", min_length=1)
    deck: list[str] = Field(default_factory=list, description="Cards already picked in this draft")
    pack_number: int = Field(1, ge=1, le=3, description="Current pack number (1-3)")
    pick_number: int = Field(1, ge=1, le=15, description="Current pick number (1-15)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pack": ["Lightning Bolt", "Counterspell", "Llanowar Elves"],
                    "deck": ["Swords to Plowshares", "Path to Exile"],
                    "pack_number": 1,
                    "pick_number": 3,
                }
            ]
        }
    }


class CardRecommendation(BaseModel):
    """Single card recommendation with metadata."""

    card_name: str = Field(..., description="Name of the recommended card")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence (0-1)")
    rank: int = Field(..., ge=1, description="Rank in pack (1 = top recommendation)")
    top_features: dict[str, float] | None = Field(
        None, description="Top contributing features (if explain=True)"
    )


class PredictResponse(BaseModel):
    """Response with card recommendations."""

    recommendations: list[CardRecommendation] = Field(
        ..., description="Ranked list of card recommendations"
    )
    pack_size: int = Field(..., description="Number of cards in pack")
    deck_size: int = Field(..., description="Number of cards in deck")
    model_version: str = Field(..., description="Model version used for prediction")
    inference_time_ms: float = Field(..., description="Time taken for inference in milliseconds")


class ExplainRequest(BaseModel):
    """Request for prediction explanation."""

    pack: list[str] = Field(..., description="List of card names in the current pack")
    deck: list[str] = Field(default_factory=list, description="Cards already picked")
    card_to_explain: str = Field(..., description="Card to explain")
    pack_number: int = Field(1, ge=1, le=3)
    pick_number: int = Field(1, ge=1, le=15)
    top_k_features: int = Field(5, ge=1, le=20, description="Number of top features to return")


class FeatureContribution(BaseModel):
    """Feature contribution to prediction."""

    feature_name: str
    contribution: float
    direction: str = Field(..., description="'positive' or 'negative'")
    description: str | None = Field(None, description="Human-readable explanation")


class ExplainResponse(BaseModel):
    """Response with prediction explanation."""

    card_name: str
    confidence: float
    top_features: list[FeatureContribution]
    alternative_picks: list[CardRecommendation]
    explanation_text: str = Field(..., description="Natural language explanation")
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="'healthy' or 'unhealthy'")
    model_loaded: bool
    model_version: str | None
    model_type: str | None
    feature_dimension: int | None
    num_classes: int | None
    cache_stats: dict[str, Any] | None = None
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")
    code: str | None = Field(None, description="Error code")
