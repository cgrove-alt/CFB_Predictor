"""
Data Validation for Sharp Sports Predictor.

Provides schema validation for:
- Feature data
- Game data
- Predictions
- Betting lines
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Raised when data validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    fixed_data: Optional[Any] = None

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class FeatureValidator:
    """Validates feature arrays for model prediction."""

    # Valid ranges for features
    FEATURE_RANGES = {
        "home_pregame_elo": (800, 2200),
        "away_pregame_elo": (800, 2200),
        "home_last5_score_avg": (0, 80),
        "away_last5_score_avg": (0, 80),
        "home_last5_defense_avg": (0, 80),
        "away_last5_defense_avg": (0, 80),
        "home_comp_off_ppa": (-1.0, 1.0),
        "away_comp_off_ppa": (-1.0, 1.0),
        "home_comp_def_ppa": (-1.0, 1.0),
        "away_comp_def_ppa": (-1.0, 1.0),
        "net_epa": (-2.0, 2.0),
        "home_team_hfa": (-5.0, 10.0),
        "away_team_hfa": (-5.0, 10.0),
        "home_rest": (0, 365),
        "away_rest": (0, 365),
        "rest_advantage": (-365, 365),
        "rest_diff": (-365, 365),
        "elo_diff": (-1500, 1500),
        "pass_efficiency_diff": (-2.0, 2.0),
        "epa_elo_interaction": (-50, 50),
        "success_diff": (-1.0, 1.0),
    }

    # Default values for missing features
    FEATURE_DEFAULTS = {
        "home_pregame_elo": 1500.0,
        "away_pregame_elo": 1500.0,
        "home_last5_score_avg": 28.0,
        "away_last5_score_avg": 28.0,
        "home_last5_defense_avg": 24.0,
        "away_last5_defense_avg": 24.0,
        "home_comp_off_ppa": 0.0,
        "away_comp_off_ppa": 0.0,
        "home_comp_def_ppa": 0.0,
        "away_comp_def_ppa": 0.0,
        "net_epa": 0.0,
        "home_team_hfa": 2.0,
        "away_team_hfa": 0.0,
        "home_rest": 7,
        "away_rest": 7,
        "rest_advantage": 0,
        "rest_diff": 0,
        "elo_diff": 0,
        "pass_efficiency_diff": 0.0,
        "epa_elo_interaction": 0.0,
        "success_diff": 0.0,
    }

    def __init__(self, required_features: List[str]):
        """
        Initialize validator with required features.

        Args:
            required_features: List of feature names required by the model
        """
        self.required_features = required_features

    def validate(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        fix_invalid: bool = True,
    ) -> ValidationResult:
        """
        Validate a feature array.

        Args:
            features: Feature array (1D or 2D)
            feature_names: Optional list of feature names corresponding to columns
            fix_invalid: Whether to fix invalid values (NaN, out of range)

        Returns:
            ValidationResult with validation status and optionally fixed data
        """
        errors = []
        warnings = []

        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Check shape
        if features.shape[1] != len(self.required_features):
            errors.append(
                f"Expected {len(self.required_features)} features, "
                f"got {features.shape[1]}"
            )

        if feature_names is None:
            feature_names = self.required_features

        # Create a copy for fixing
        fixed = features.copy().astype(float)

        # Check each feature
        for i, (name, value) in enumerate(zip(feature_names, fixed.flatten())):
            # Check for NaN
            if np.isnan(value):
                warnings.append(f"NaN value for {name}")
                if fix_invalid:
                    fixed.flat[i] = self.FEATURE_DEFAULTS.get(name, 0.0)

            # Check range
            elif name in self.FEATURE_RANGES:
                min_val, max_val = self.FEATURE_RANGES[name]
                if value < min_val or value > max_val:
                    warnings.append(
                        f"{name}={value:.2f} out of range [{min_val}, {max_val}]"
                    )
                    if fix_invalid:
                        fixed.flat[i] = np.clip(value, min_val, max_val)

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            fixed_data=fixed if fix_invalid else None,
        )

    def sanitize_value(self, value: Any, feature_name: str) -> float:
        """
        Sanitize a single feature value.

        Args:
            value: Value to sanitize
            feature_name: Name of the feature

        Returns:
            Sanitized float value
        """
        default = self.FEATURE_DEFAULTS.get(feature_name, 0.0)

        if value is None:
            return default

        try:
            float_val = float(value)
            if np.isnan(float_val):
                return default
            return float_val
        except (TypeError, ValueError):
            return default


class GameDataValidator:
    """Validates game data from API or CSV."""

    REQUIRED_FIELDS = {
        "id": int,
        "season": int,
        "week": int,
        "home_team": str,
        "away_team": str,
    }

    OPTIONAL_FIELDS = {
        "home_points": (int, float),
        "away_points": (int, float),
        "home_pregame_elo": (int, float),
        "away_pregame_elo": (int, float),
        "Margin": (int, float),
    }

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a games DataFrame.

        Args:
            df: DataFrame with game data

        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []

        # Check required fields
        for field, expected_type in self.REQUIRED_FIELDS.items():
            if field not in df.columns:
                errors.append(f"Missing required field: {field}")

        # Check data types and missing values
        for field in df.columns:
            null_count = df[field].isna().sum()
            if null_count > 0:
                if field in self.REQUIRED_FIELDS:
                    errors.append(f"{field} has {null_count} null values")
                else:
                    warnings.append(f"{field} has {null_count} null values")

        # Check season range
        if "season" in df.columns:
            invalid_seasons = df[(df["season"] < 2000) | (df["season"] > 2030)]
            if len(invalid_seasons) > 0:
                warnings.append(f"{len(invalid_seasons)} games with invalid season")

        # Check week range
        if "week" in df.columns:
            invalid_weeks = df[(df["week"] < 0) | (df["week"] > 20)]
            if len(invalid_weeks) > 0:
                warnings.append(f"{len(invalid_weeks)} games with invalid week")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


class PredictionValidator:
    """Validates model predictions before use."""

    def __init__(
        self,
        min_margin: float = -100.0,
        max_margin: float = 100.0,
        warn_threshold: float = 50.0,
    ):
        """
        Initialize validator.

        Args:
            min_margin: Minimum valid predicted margin
            max_margin: Maximum valid predicted margin
            warn_threshold: Threshold for warning about extreme predictions
        """
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.warn_threshold = warn_threshold

    def validate_prediction(
        self,
        predicted_margin: float,
        vegas_line: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate a single prediction.

        Args:
            predicted_margin: Model's predicted margin
            vegas_line: Optional Vegas line for comparison

        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []

        # Check for NaN
        if np.isnan(predicted_margin):
            errors.append("Predicted margin is NaN")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Check range
        if predicted_margin < self.min_margin or predicted_margin > self.max_margin:
            errors.append(
                f"Predicted margin {predicted_margin:.1f} out of range "
                f"[{self.min_margin}, {self.max_margin}]"
            )

        # Warn about extreme predictions
        if abs(predicted_margin) > self.warn_threshold:
            warnings.append(f"Extreme prediction: {predicted_margin:.1f}")

        # Compare to Vegas if available
        if vegas_line is not None and not np.isnan(vegas_line):
            edge = abs(predicted_margin - (-vegas_line))
            if edge > 30:
                warnings.append(
                    f"Large disagreement with Vegas: edge={edge:.1f} points"
                )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


class BettingLineValidator:
    """Validates betting lines from API."""

    def __init__(
        self,
        max_spread: float = 60.0,
        max_total: float = 100.0,
    ):
        """
        Initialize validator.

        Args:
            max_spread: Maximum valid spread
            max_total: Maximum valid over/under total
        """
        self.max_spread = max_spread
        self.max_total = max_total

    def validate_spread(self, spread: float) -> ValidationResult:
        """Validate a spread line."""
        errors = []
        warnings = []

        if np.isnan(spread):
            errors.append("Spread is NaN")
        elif abs(spread) > self.max_spread:
            warnings.append(f"Unusual spread: {spread:+.1f}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    def validate_total(self, total: float) -> ValidationResult:
        """Validate an over/under total."""
        errors = []
        warnings = []

        if np.isnan(total):
            errors.append("Total is NaN")
        elif total < 20 or total > self.max_total:
            warnings.append(f"Unusual total: {total:.1f}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_and_fix_features(
    features: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to validate and fix a feature array.

    Args:
        features: Feature array
        feature_names: List of feature names

    Returns:
        Tuple of (fixed_features, warnings)
    """
    validator = FeatureValidator(feature_names)
    result = validator.validate(features, feature_names, fix_invalid=True)

    if not result.is_valid:
        raise ValidationError(
            f"Feature validation failed: {'; '.join(result.errors)}"
        )

    return result.fixed_data, result.warnings
