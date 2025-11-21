"""SHAP-based model explainability for draft predictions.

This module provides human-readable explanations for why the model
made specific pick recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import shap

logger = logging.getLogger("ai_draft_bot.explainability.shap_explainer")


@dataclass
class PickExplanation:
    """Explanation for a single pick recommendation.

    Attributes:
        recommended_card: Card the model recommended
        confidence: Prediction confidence (0.0-1.0)
        top_features: Dict of feature_name -> contribution
        alternative_picks: List of (card_name, probability) tuples
    """

    recommended_card: str
    confidence: float
    top_features: Mapping[str, float]
    alternative_picks: list[tuple[str, float]]


class DraftExplainer:
    """Explainer for draft model predictions using SHAP."""

    def __init__(self, model, feature_names: list[str] | None = None):
        """Initialize explainer.

        Args:
            model: Trained draft model
            feature_names: Optional list of feature names for readability
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def fit(self, background_data: np.ndarray, max_samples: int = 100) -> None:
        """Fit the SHAP explainer on background data.

        Args:
            background_data: Sample of training data for SHAP baseline
            max_samples: Maximum background samples to use
        """
        logger.info("Fitting SHAP explainer...")

        # Sample background data if too large
        if len(background_data) > max_samples:
            indices = np.random.choice(
                len(background_data), max_samples, replace=False
            )
            background_data = background_data[indices]

        # Create SHAP explainer (TreeExplainer for XGBoost/LightGBM)
        try:
            self.explainer = shap.TreeExplainer(self.model.artifacts.model)
            logger.info("✓ SHAP explainer fitted successfully")
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            raise

    def explain_pick(
        self, features: np.ndarray, top_k: int = 5
    ) -> PickExplanation:
        """Explain a single pick prediction.

        Args:
            features: Feature vector for the pick
            top_k: Number of top contributing features to return

        Returns:
            PickExplanation with human-readable explanation
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        # Get prediction
        predicted_card = self.model.predict(features)
        proba_dist = self.model.predict_proba(features)

        confidence = proba_dist.get(predicted_card, 0.0)

        # Get SHAP values
        shap_values = self.explainer.shap_values(features.reshape(1, -1))

        # Get top contributing features
        if isinstance(shap_values, list):
            # Multi-class: get SHAP values for predicted class
            pred_idx = list(proba_dist.keys()).index(predicted_card)
            feature_contributions = shap_values[pred_idx][0]
        else:
            feature_contributions = shap_values[0]

        # Sort by absolute contribution
        top_indices = np.argsort(np.abs(feature_contributions))[-top_k:][::-1]

        top_features = {}
        for idx in top_indices:
            if self.feature_names and idx < len(self.feature_names):
                feat_name = self.feature_names[idx]
            else:
                feat_name = f"Feature_{idx}"

            contribution = float(feature_contributions[idx])
            top_features[feat_name] = contribution

        # Get alternative picks
        alternatives = sorted(proba_dist.items(), key=lambda x: x[1], reverse=True)[
            1:4
        ]

        return PickExplanation(
            recommended_card=predicted_card,
            confidence=confidence,
            top_features=top_features,
            alternative_picks=alternatives,
        )

    def explain_pick_human_readable(self, features: np.ndarray) -> str:
        """Get human-readable explanation text.

        Args:
            features: Feature vector

        Returns:
            Formatted explanation string
        """
        explanation = self.explain_pick(features)

        lines = [
            f"Recommended Pick: {explanation.recommended_card}",
            f"Confidence: {explanation.confidence:.1%}",
            "",
            "Why this card?",
        ]

        for feat_name, contribution in explanation.top_features.items():
            direction = "increases" if contribution > 0 else "decreases"
            lines.append(f"  • {feat_name}: {direction} pick value ({contribution:+.3f})")

        if explanation.alternative_picks:
            lines.append("")
            lines.append("Alternative considerations:")
            for card, prob in explanation.alternative_picks:
                lines.append(f"  • {card} ({prob:.1%})")

        return "\n".join(lines)
