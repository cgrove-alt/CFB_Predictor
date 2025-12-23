"""
Model Optimization for Sharp Sports Predictor.

Provides:
- Ensemble weight optimization via cross-validation
- Model compression for faster loading
- Hyperparameter tuning
"""

import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from ..utils.config import get_config
from ..utils.logging_config import get_logger, log_timing

logger = get_logger(__name__)


class WeightOptimizer:
    """
    Optimizes ensemble weights via cross-validation.

    Instead of using fixed weights (0.5, 0.3, 0.2), this class
    finds optimal weights that minimize MAE on validation data.
    """

    def __init__(self):
        """Initialize the weight optimizer."""
        self.config = get_config()
        self._optimal_weights: Optional[List[float]] = None
        self._optimization_history: List[Dict] = []

    def optimize_weights(
        self,
        base_models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> Tuple[List[float], float]:
        """
        Find optimal weights for ensemble using cross-validation.

        Args:
            base_models: Dictionary of fitted base models
            X: Features
            y: Targets
            n_splits: Number of CV splits

        Returns:
            Tuple of (optimal_weights, best_mae)
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Get base model predictions for each fold
        model_names = list(base_models.keys())
        n_models = len(model_names)

        def objective(weights):
            """Objective function: weighted average MAE across folds."""
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize

            total_mae = 0.0
            n_folds = 0

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Get predictions from each model
                fold_preds = []
                for name in model_names:
                    model = base_models[name]
                    # Refit on this fold
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    fold_preds.append(pred)

                # Weighted average
                fold_preds = np.array(fold_preds)
                weighted_pred = np.average(fold_preds, axis=0, weights=weights)

                fold_mae = mean_absolute_error(y_val, weighted_pred)
                total_mae += fold_mae
                n_folds += 1

            return total_mae / n_folds

        # Initial weights
        x0 = np.ones(n_models) / n_models

        # Bounds: weights between 0.05 and 0.9
        bounds = [(0.05, 0.9) for _ in range(n_models)]

        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        with log_timing(logger, "weight optimization"):
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6},
            )

        optimal_weights = result.x / result.x.sum()  # Normalize
        best_mae = result.fun

        self._optimal_weights = list(optimal_weights)
        self._optimization_history.append({
            'weights': self._optimal_weights,
            'mae': best_mae,
            'model_names': model_names,
        })

        logger.info(f"Optimal weights: {dict(zip(model_names, optimal_weights))}")
        logger.info(f"Best CV MAE: {best_mae:.2f}")

        return list(optimal_weights), best_mae

    def grid_search_weights(
        self,
        base_models: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        step: float = 0.1,
    ) -> Tuple[List[float], float]:
        """
        Grid search for optimal 3-model weights.

        Args:
            base_models: Dictionary of fitted base models
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            step: Step size for grid search

        Returns:
            Tuple of (optimal_weights, best_mae)
        """
        model_names = list(base_models.keys())

        if len(model_names) != 3:
            raise ValueError("Grid search only supports exactly 3 models")

        # Fit models
        predictions = {}
        for name, model in base_models.items():
            model.fit(X_train, y_train)
            predictions[name] = model.predict(X_val)

        best_weights = None
        best_mae = float('inf')

        # Grid search
        weights_range = np.arange(0.1, 0.9 + step, step)

        for w1 in weights_range:
            for w2 in weights_range:
                w3 = 1.0 - w1 - w2
                if w3 < 0.1 or w3 > 0.9:
                    continue

                weights = [w1, w2, w3]

                # Weighted average prediction
                weighted_pred = sum(
                    w * predictions[name]
                    for w, name in zip(weights, model_names)
                )

                mae = mean_absolute_error(y_val, weighted_pred)

                if mae < best_mae:
                    best_mae = mae
                    best_weights = weights

        self._optimal_weights = best_weights
        logger.info(f"Grid search optimal weights: {dict(zip(model_names, best_weights))}")

        return best_weights, best_mae

    @property
    def optimal_weights(self) -> Optional[List[float]]:
        """Get the optimal weights from last optimization."""
        return self._optimal_weights


class ModelCompressor:
    """
    Compresses model artifacts for faster loading.

    Techniques:
    - Remove unnecessary estimator data
    - Use more efficient serialization
    - Quantize weights where possible
    """

    def __init__(self):
        """Initialize the model compressor."""
        self.config = get_config()

    def compress_model(
        self,
        model: Any,
        output_path: str,
        compression_level: int = 6,
    ) -> Tuple[Path, float]:
        """
        Compress a model to a smaller file.

        Args:
            model: Model to compress
            output_path: Output file path
            compression_level: gzip compression level (1-9)

        Returns:
            Tuple of (output_path, compression_ratio)
        """
        output_path = Path(output_path)

        # Get original size
        original_data = pickle.dumps(model)
        original_size = len(original_data)

        # Compress
        compressed_data = gzip.compress(original_data, compresslevel=compression_level)
        compressed_size = len(compressed_data)

        # Save compressed
        with open(output_path, 'wb') as f:
            f.write(compressed_data)

        compression_ratio = compressed_size / original_size

        logger.info(
            f"Compressed model: {original_size / 1024 / 1024:.1f}MB -> "
            f"{compressed_size / 1024 / 1024:.1f}MB "
            f"({compression_ratio:.1%})"
        )

        return output_path, compression_ratio

    def decompress_model(self, model_path: str) -> Any:
        """
        Load a compressed model.

        Args:
            model_path: Path to compressed model file

        Returns:
            Decompressed model
        """
        with open(model_path, 'rb') as f:
            compressed_data = f.read()

        decompressed_data = gzip.decompress(compressed_data)
        model = pickle.loads(decompressed_data)

        return model

    def strip_model(self, model: Any) -> Any:
        """
        Strip unnecessary data from model to reduce size.

        This removes training data cached in the model while
        preserving prediction capability.

        Args:
            model: Model to strip

        Returns:
            Stripped model (same object, modified in place)
        """
        # For HistGradientBoosting, remove validation data
        if hasattr(model, '_raw_predict'):
            if hasattr(model, '_X_binned_train'):
                del model._X_binned_train
            if hasattr(model, '_y_train'):
                del model._y_train

        # For RandomForest, we can't easily strip without losing functionality

        # For Pipeline, recurse
        if hasattr(model, 'named_steps'):
            for step_name, step in model.named_steps.items():
                self.strip_model(step)

        # For VotingRegressor/StackingRegressor, recurse
        if hasattr(model, 'estimators_'):
            for name, estimator in model.estimators_:
                self.strip_model(estimator)

        if hasattr(model, 'final_estimator_'):
            self.strip_model(model.final_estimator_)

        return model

    def optimize_and_save(
        self,
        model: Any,
        output_path: str,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full optimization pipeline: strip, compress, and save.

        Args:
            model: Model to optimize
            output_path: Output file path
            feature_names: Optional feature names to include

        Returns:
            Dict with optimization stats
        """
        # Get original size
        original_data = pickle.dumps(model)
        original_size = len(original_data)

        # Strip unnecessary data
        stripped_model = self.strip_model(model)

        # Save with metadata
        save_data = {
            'model': stripped_model,
            'feature_names': feature_names,
            'version': '2.0',
            'compressed': True,
        }

        # Compress and save
        output_path = Path(output_path)
        compressed_data = gzip.compress(
            pickle.dumps(save_data),
            compresslevel=6,
        )

        with open(output_path, 'wb') as f:
            f.write(compressed_data)

        final_size = len(compressed_data)

        stats = {
            'original_size_mb': original_size / 1024 / 1024,
            'final_size_mb': final_size / 1024 / 1024,
            'compression_ratio': final_size / original_size,
            'output_path': str(output_path),
        }

        logger.info(
            f"Model optimized: {stats['original_size_mb']:.1f}MB -> "
            f"{stats['final_size_mb']:.1f}MB"
        )

        return stats
