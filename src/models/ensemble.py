"""
Ensemble Model Training for Sharp Sports Predictor.

Provides:
- VotingRegressor ensemble
- StackingRegressor with meta-learner
- Automatic hyperparameter optimization
- Cross-validation support
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.config import get_config
from ..utils.logging_config import get_logger, StructuredLogger, log_timing

logger = get_logger(__name__)
structured_logger = StructuredLogger(__name__)


@dataclass
class TrainingResult:
    """Results from model training."""

    model: Any
    train_mae: float
    test_mae: float
    cv_scores: Optional[List[float]] = None
    feature_names: Optional[List[str]] = None
    improvement: float = 0.0


class EnsembleTrainer:
    """
    Trains ensemble models for CFB prediction.

    Supports:
    - VotingRegressor (weighted average)
    - StackingRegressor (meta-learner)
    - Automatic weight optimization
    """

    def __init__(self):
        """Initialize the ensemble trainer."""
        self.config = get_config()
        self._models: Dict[str, Any] = {}
        self._best_weights: Optional[List[float]] = None

    def create_base_models(self) -> Dict[str, Any]:
        """Create the base models for ensemble."""
        cfg = self.config.model

        # HistGradientBoosting - handles NaN natively
        hgb = HistGradientBoostingRegressor(
            max_iter=cfg.hgb_max_iter,
            max_depth=cfg.hgb_max_depth,
            learning_rate=cfg.hgb_learning_rate,
            l2_regularization=cfg.hgb_l2_regularization,
            random_state=cfg.random_state,
        )

        # RandomForest with imputer
        rf = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('rf', RandomForestRegressor(
                n_estimators=cfg.rf_n_estimators,
                max_depth=cfg.rf_max_depth,
                min_samples_split=cfg.rf_min_samples_split,
                min_samples_leaf=cfg.rf_min_samples_leaf,
                random_state=cfg.random_state,
                n_jobs=-1,
            ))
        ])

        # Linear model with preprocessing
        lr = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=cfg.ridge_alpha))
        ])

        return {
            'hgb': hgb,
            'rf': rf,
            'lr': lr,
        }

    def train_voting_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        weights: Optional[List[float]] = None,
    ) -> TrainingResult:
        """
        Train a VotingRegressor ensemble.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            weights: Optional custom weights [hgb, rf, lr]

        Returns:
            TrainingResult with trained model and metrics
        """
        cfg = self.config.model

        if weights is None:
            weights = [cfg.ensemble_hgb_weight, cfg.ensemble_rf_weight, cfg.ensemble_lr_weight]

        base_models = self.create_base_models()

        with log_timing(logger, "voting ensemble training"):
            ensemble = VotingRegressor(
                estimators=[
                    ('hgb', base_models['hgb']),
                    ('rf', base_models['rf']),
                    ('lr', base_models['lr']),
                ],
                weights=weights,
            )

            ensemble.fit(X_train, y_train)

        # Evaluate
        train_pred = ensemble.predict(X_train)
        test_pred = ensemble.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        # Get baseline for comparison
        baseline_mae = self._get_baseline_mae(X_train, y_train, X_test, y_test)
        improvement = baseline_mae - test_mae

        structured_logger.log_model_performance(
            model_name="VotingEnsemble",
            mae=test_mae,
            samples=len(X_test),
            improvement=improvement,
        )

        return TrainingResult(
            model=ensemble,
            train_mae=train_mae,
            test_mae=test_mae,
            feature_names=list(X_train.columns),
            improvement=improvement,
        )

    def train_stacking_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_folds: Optional[int] = None,
    ) -> TrainingResult:
        """
        Train a StackingRegressor with meta-learner.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            cv_folds: Number of CV folds for meta-features

        Returns:
            TrainingResult with trained model and metrics
        """
        cfg = self.config.model
        cv_folds = cv_folds or cfg.stacking_cv_folds

        base_models = self.create_base_models()

        # Meta-learner: RidgeCV learns optimal weights
        meta_learner = RidgeCV(alphas=cfg.ridge_cv_alphas)

        with log_timing(logger, "stacking ensemble training"):
            stacking = StackingRegressor(
                estimators=[
                    ('gradient', base_models['hgb']),
                    ('forest', base_models['rf']),
                    ('linear', base_models['lr']),
                ],
                final_estimator=meta_learner,
                cv=cv_folds,
                n_jobs=-1,
            )

            stacking.fit(X_train, y_train)

        # Log meta-learner alpha
        if hasattr(meta_learner, 'alpha_'):
            logger.info(f"Meta-learner selected alpha: {meta_learner.alpha_}")

        # Evaluate
        train_pred = stacking.predict(X_train)
        test_pred = stacking.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        # Get baseline for comparison
        baseline_mae = self._get_baseline_mae(X_train, y_train, X_test, y_test)
        improvement = baseline_mae - test_mae

        structured_logger.log_model_performance(
            model_name="StackingEnsemble",
            mae=test_mae,
            samples=len(X_test),
            improvement=improvement,
        )

        return TrainingResult(
            model=stacking,
            train_mae=train_mae,
            test_mae=test_mae,
            feature_names=list(X_train.columns),
            improvement=improvement,
        )

    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> List[float]:
        """
        Perform time-series cross-validation.

        Args:
            model: Model to validate
            X: Features
            y: Targets
            n_splits: Number of CV splits

        Returns:
            List of MAE scores for each fold
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(
            model, X, y,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
        )
        return [-s for s in scores]  # Convert to positive MAE

    def _get_baseline_mae(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> float:
        """Get baseline MAE from single HGB model."""
        cfg = self.config.model

        baseline = HistGradientBoostingRegressor(
            max_iter=cfg.hgb_max_iter,
            max_depth=cfg.hgb_max_depth,
            learning_rate=cfg.hgb_learning_rate,
            random_state=cfg.random_state,
        )
        baseline.fit(X_train, y_train)
        pred = baseline.predict(X_test)
        return mean_absolute_error(y_test, pred)

    def save_model(
        self,
        model: Any,
        filename: str,
        feature_names: Optional[List[str]] = None,
    ) -> Path:
        """
        Save a trained model.

        Args:
            model: Trained model
            filename: Output filename
            feature_names: Optional feature names to save

        Returns:
            Path to saved model
        """
        output_path = self.config.paths.base_dir / filename

        # Save model with metadata
        save_data = {
            'model': model,
            'feature_names': feature_names,
            'version': '2.0',
        }

        joblib.dump(save_data, output_path)
        logger.info(f"Model saved to {output_path}")

        return output_path


class StackingModel:
    """
    Wrapper for loading and using stacking models.

    Provides:
    - Model loading with fallbacks
    - Feature validation
    - Variance calculation for confusion filter
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the stacking model.

        Args:
            model_path: Optional path to model file
        """
        self.config = get_config()
        self._model = None
        self._feature_names: Optional[List[str]] = None

        if model_path:
            self.load(model_path)

    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load model from file with fallbacks.

        Args:
            model_path: Path to model file

        Returns:
            True if loaded successfully
        """
        paths_to_try = []

        if model_path:
            paths_to_try.append(Path(model_path))

        # Add default paths
        base = self.config.paths.base_dir
        paths_to_try.extend([
            base / self.config.paths.stacking_model_file,
            base / self.config.paths.ensemble_model_file,
            base / self.config.paths.fallback_model_file,
        ])

        for path in paths_to_try:
            if path.exists():
                try:
                    data = joblib.load(path)

                    # Handle both old (just model) and new (dict with metadata) formats
                    if isinstance(data, dict):
                        self._model = data['model']
                        self._feature_names = data.get('feature_names')
                    else:
                        self._model = data
                        self._feature_names = None

                    logger.info(f"Loaded model from {path}")
                    return True

                except Exception as e:
                    logger.warning(f"Failed to load model from {path}: {e}")

        logger.error("No model could be loaded")
        return False

    def predict(self, features: np.ndarray) -> float:
        """
        Make a prediction.

        Args:
            features: Feature array

        Returns:
            Predicted margin
        """
        if self._model is None:
            raise RuntimeError("No model loaded")

        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        return self._model.predict(features)[0]

    def get_model_variance(self, features: np.ndarray) -> Tuple[float, List[float]]:
        """
        Calculate variance between base model predictions.

        Used as a "Confusion Filter" - high variance means models disagree.

        Args:
            features: Feature array

        Returns:
            Tuple of (std_dev, individual_predictions)
        """
        if self._model is None:
            return 0.0, []

        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)

        predictions = []

        # Try to get individual estimator predictions
        if hasattr(self._model, 'estimators_'):
            # StackingRegressor
            for name, estimator in self._model.estimators_:
                try:
                    pred = estimator.predict(features)[0]
                    predictions.append(pred)
                except Exception:
                    pass

        elif hasattr(self._model, 'named_estimators_'):
            # VotingRegressor
            for name in self._model.named_estimators_:
                try:
                    est = self._model.named_estimators_[name]
                    pred = est.predict(features)[0]
                    predictions.append(pred)
                except Exception:
                    pass

        elif hasattr(self._model, 'named_steps'):
            # Pipeline with ensemble
            for step_name, step in self._model.named_steps.items():
                if hasattr(step, 'estimators_'):
                    for name, estimator in step.estimators_:
                        try:
                            pred = estimator.predict(features)[0]
                            predictions.append(pred)
                        except Exception:
                            pass

        if len(predictions) >= 2:
            return float(np.std(predictions)), predictions
        else:
            return 0.0, predictions

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Get the feature names used by this model."""
        return self._feature_names
