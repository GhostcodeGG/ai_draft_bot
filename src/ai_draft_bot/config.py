"""Centralized configuration management for AI Draft Bot.

This module provides a unified configuration system for all aspects of the application,
including caching, feature extraction, model hyperparameters, and deployment settings.
Configuration values can be loaded from environment variables for production flexibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class CacheConfig:
    """Configuration for caching behavior."""

    # Scryfall API cache
    scryfall_cache_enabled: bool = True
    scryfall_cache_dir: Path = field(default_factory=lambda: Path("cache/scryfall"))
    scryfall_lru_size: int = 2000
    scryfall_rate_limit_ms: int = 75  # Milliseconds between requests

    # General feature cache
    feature_cache_enabled: bool = True
    feature_cache_dir: Path = field(default_factory=lambda: Path("cache/features"))

    @classmethod
    def from_env(cls) -> CacheConfig:
        """Load cache configuration from environment variables."""
        return cls(
            scryfall_cache_enabled=os.getenv("SCRYFALL_CACHE_ENABLED", "true").lower() == "true",
            scryfall_cache_dir=Path(os.getenv("SCRYFALL_CACHE_DIR", "cache/scryfall")),
            scryfall_lru_size=int(os.getenv("SCRYFALL_LRU_SIZE", "2000")),
            scryfall_rate_limit_ms=int(os.getenv("SCRYFALL_RATE_LIMIT_MS", "75")),
            feature_cache_enabled=os.getenv("FEATURE_CACHE_ENABLED", "true").lower() == "true",
            feature_cache_dir=Path(os.getenv("FEATURE_CACHE_DIR", "cache/features")),
        )


@dataclass
class FeatureConfig:
    """Configuration for feature extraction behavior."""

    # Feature thresholds
    removal_threshold: float = 0.5  # Threshold for identifying removal spells
    bomb_rarity_threshold: str = "rare"  # Minimum rarity for "bomb" detection
    wheel_probability_threshold: float = 0.3  # Threshold for wheeling probability

    # Text analysis
    use_scryfall: bool = True  # Whether to use Scryfall for card text analysis
    max_workers: int = 4  # Max workers for parallel feature extraction

    # Archetype detection
    min_archetype_cards: int = 3  # Minimum cards needed to detect archetype

    @classmethod
    def from_env(cls) -> FeatureConfig:
        """Load feature configuration from environment variables."""
        return cls(
            removal_threshold=float(os.getenv("REMOVAL_THRESHOLD", "0.5")),
            bomb_rarity_threshold=os.getenv("BOMB_RARITY_THRESHOLD", "rare"),
            wheel_probability_threshold=float(os.getenv("WHEEL_PROB_THRESHOLD", "0.3")),
            use_scryfall=os.getenv("USE_SCRYFALL", "true").lower() == "true",
            max_workers=int(os.getenv("MAX_FEATURE_WORKERS", "4")),
            min_archetype_cards=int(os.getenv("MIN_ARCHETYPE_CARDS", "3")),
        )


@dataclass
class ModelConfig:
    """Default hyperparameters for various model architectures."""

    # XGBoost defaults
    xgboost_max_depth: int = 8
    xgboost_learning_rate: float = 0.1
    xgboost_n_estimators: int = 500

    # LightGBM defaults
    lightgbm_max_depth: int = 8
    lightgbm_learning_rate: float = 0.1
    lightgbm_n_estimators: int = 500

    # Neural network defaults
    neural_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    neural_batch_size: int = 128
    neural_learning_rate: float = 0.001
    neural_epochs: int = 50

    # Ensemble defaults
    ensemble_method: str = "weighted"  # 'voting', 'weighted', or 'stacking'

    @classmethod
    def from_env(cls) -> ModelConfig:
        """Load model configuration from environment variables."""
        return cls(
            xgboost_max_depth=int(os.getenv("XGBOOST_MAX_DEPTH", "8")),
            xgboost_learning_rate=float(os.getenv("XGBOOST_LR", "0.1")),
            xgboost_n_estimators=int(os.getenv("XGBOOST_N_ESTIMATORS", "500")),
            lightgbm_max_depth=int(os.getenv("LIGHTGBM_MAX_DEPTH", "8")),
            lightgbm_learning_rate=float(os.getenv("LIGHTGBM_LR", "0.1")),
            lightgbm_n_estimators=int(os.getenv("LIGHTGBM_N_ESTIMATORS", "500")),
            neural_batch_size=int(os.getenv("NEURAL_BATCH_SIZE", "128")),
            neural_learning_rate=float(os.getenv("NEURAL_LR", "0.001")),
            neural_epochs=int(os.getenv("NEURAL_EPOCHS", "50")),
            ensemble_method=os.getenv("ENSEMBLE_METHOD", "weighted"),
        )


@dataclass
class Config:
    """Main configuration object combining all sub-configurations."""

    cache: CacheConfig = field(default_factory=CacheConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)

    # Global settings
    log_level: str = "INFO"
    random_seed: int = 13  # Lucky number 13 for reproducibility

    # Paths
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # Singleton instance
    _instance: ClassVar[Config | None] = None

    @classmethod
    def get_instance(cls) -> Config:
        """Get or create the singleton configuration instance."""
        if cls._instance is None:
            cls._instance = cls.from_env()
        return cls._instance

    @classmethod
    def from_env(cls) -> Config:
        """Load full configuration from environment variables."""
        return cls(
            cache=CacheConfig.from_env(),
            features=FeatureConfig.from_env(),
            models=ModelConfig.from_env(),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            random_seed=int(os.getenv("RANDOM_SEED", "13")),
            artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "artifacts")),
            data_dir=Path(os.getenv("DATA_DIR", "data")),
        )

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.cache.scryfall_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Don't create data_dir as it should contain user-provided data


# Global configuration accessor
def get_config() -> Config:
    """Get the global configuration instance.

    Returns:
        The singleton Config object with all settings.

    Example:
        >>> config = get_config()
        >>> print(config.cache.scryfall_cache_dir)
        cache/scryfall
    """
    return Config.get_instance()
