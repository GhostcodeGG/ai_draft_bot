"""Optuna-based hyperparameter optimization for draft models.

This module provides automatic hyperparameter tuning using Bayesian optimization
to find optimal model configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import optuna
from optuna.trial import Trial

from ai_draft_bot.features.draft_context import PickFeatures
from ai_draft_bot.models.advanced_drafter import (
    AdvancedTrainConfig,
    ModelType,
    train_advanced_model,
)

logger = logging.getLogger("ai_draft_bot.optimization.optuna_tuner")


@dataclass
class OptunaConfig:
    """Configuration for Optuna hyperparameter tuning.

    Attributes:
        n_trials: Number of optimization trials
        timeout: Maximum optimization time in seconds
        model_type: Which model to optimize (xgboost, lightgbm)
        test_size: Validation split
        random_state: Random seed
        use_gpu: Whether to use GPU
    """

    n_trials: int = 100
    timeout: int | None = None  # No timeout by default
    model_type: ModelType = ModelType.XGBOOST
    test_size: float = 0.2
    random_state: int = 13
    use_gpu: bool = False


def create_objective(
    rows: Sequence[PickFeatures],
    config: OptunaConfig,
):
    """Create Optuna objective function.

    Args:
        rows: Training data
        config: Optuna configuration

    Returns:
        Objective function for Optuna
    """

    def objective(trial: Trial) -> float:
        """Optuna objective: maximize validation accuracy.

        Args:
            trial: Optuna trial object

        Returns:
            Validation accuracy to maximize
        """
        # Suggest hyperparameters
        if config.model_type == ModelType.XGBOOST:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 4, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            }
        elif config.model_type == ModelType.LIGHTGBM:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 4, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            }
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        # Train model with these hyperparameters
        train_config = AdvancedTrainConfig(
            model_type=config.model_type,
            test_size=config.test_size,
            random_state=config.random_state,
            early_stopping_rounds=50,
            use_gpu=config.use_gpu,
            **params,
        )

        try:
            result = train_advanced_model(rows, config=train_config)
            accuracy = result.metrics.accuracy

            # Log trial results
            logger.info(
                f"Trial {trial.number}: accuracy={accuracy:.4f}, params={trial.params}"
            )

            return accuracy

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return very low score for failed trials
            return 0.0

    return objective


def optimize_hyperparameters(
    rows: Sequence[PickFeatures],
    config: OptunaConfig | None = None,
) -> tuple[dict, float]:
    """Optimize hyperparameters using Optuna.

    Args:
        rows: Training data
        config: Optuna configuration

    Returns:
        Tuple of (best_params, best_accuracy)
    """
    if config is None:
        config = OptunaConfig()

    logger.info("=" * 60)
    logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Model type: {config.model_type.value}")
    logger.info(f"Number of trials: {config.n_trials}")
    logger.info(f"Dataset size: {len(rows)} picks")
    logger.info("=" * 60)

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    # Create objective
    objective = create_objective(rows, config)

    # Optimize
    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        show_progress_bar=True,
    )

    # Results
    best_trial = study.best_trial
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best accuracy: {best_trial.value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"Total trials: {len(study.trials)}")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    logger.info(f"Completed trials: {len(completed)}")
    logger.info("=" * 60)

    return best_trial.params, best_trial.value


def optimize_and_train(
    rows: Sequence[PickFeatures],
    optuna_config: OptunaConfig | None = None,
):
    """Optimize hyperparameters and train final model.

    Args:
        rows: Training data
        optuna_config: Optuna configuration

    Returns:
        TrainResult with optimized model
    """
    if optuna_config is None:
        optuna_config = OptunaConfig()

    # Optimize
    best_params, best_accuracy = optimize_hyperparameters(rows, optuna_config)

    logger.info("Training final model with optimized hyperparameters...")

    # Train final model
    train_config = AdvancedTrainConfig(
        model_type=optuna_config.model_type,
        test_size=optuna_config.test_size,
        random_state=optuna_config.random_state,
        early_stopping_rounds=50,
        use_gpu=optuna_config.use_gpu,
        **best_params,
    )

    result = train_advanced_model(rows, config=train_config)

    logger.info(f"Final model accuracy: {result.metrics.accuracy:.4f}")
    return result
