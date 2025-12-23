"""Tests for data validation module."""

import numpy as np
import pandas as pd
import pytest

from src.utils.validation import (
    FeatureValidator,
    GameDataValidator,
    PredictionValidator,
    BettingLineValidator,
    ValidationError,
    validate_and_fix_features,
)


class TestFeatureValidator:
    """Tests for FeatureValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.features = [
            'home_pregame_elo', 'away_pregame_elo',
            'home_last5_score_avg', 'away_last5_score_avg',
        ]
        self.validator = FeatureValidator(self.features)

    def test_valid_features(self):
        """Test validation of valid features."""
        features = np.array([[1600, 1400, 30, 25]])
        result = self.validator.validate(features)

        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_nan_handling(self):
        """Test that NaN values are detected and fixed."""
        features = np.array([[np.nan, 1400, 30, np.nan]])
        result = self.validator.validate(features, fix_invalid=True)

        assert result.is_valid
        assert len(result.warnings) == 2
        assert result.fixed_data is not None
        # Check that NaN values were replaced with defaults
        assert not np.isnan(result.fixed_data).any()

    def test_out_of_range_values(self):
        """Test that out-of-range values are detected."""
        # Elo of 3000 is way out of range (800-2200)
        features = np.array([[3000, 1400, 30, 25]])
        result = self.validator.validate(features, fix_invalid=True)

        assert result.is_valid
        assert len(result.warnings) >= 1
        # Check that value was clipped
        assert result.fixed_data[0, 0] == 2200

    def test_wrong_shape(self):
        """Test validation fails with wrong number of features."""
        # Only 3 features instead of 4
        features = np.array([[1600, 1400, 30]])
        result = self.validator.validate(features)

        assert not result.is_valid
        assert len(result.errors) == 1

    def test_sanitize_value(self):
        """Test sanitization of individual values."""
        # None should return default
        assert self.validator.sanitize_value(None, 'home_pregame_elo') == 1500.0

        # NaN should return default
        assert self.validator.sanitize_value(float('nan'), 'home_pregame_elo') == 1500.0

        # Valid value should be returned
        assert self.validator.sanitize_value(1600, 'home_pregame_elo') == 1600.0


class TestGameDataValidator:
    """Tests for GameDataValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = GameDataValidator()

    def test_valid_dataframe(self):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'season': [2024, 2024, 2024],
            'week': [1, 1, 2],
            'home_team': ['Alabama', 'Ohio State', 'Georgia'],
            'away_team': ['Auburn', 'Michigan', 'Florida'],
            'home_points': [35, 28, 42],
            'away_points': [21, 24, 14],
        })
        result = self.validator.validate_dataframe(df)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_required_field(self):
        """Test that missing required fields are detected."""
        df = pd.DataFrame({
            'id': [1, 2],
            'season': [2024, 2024],
            'week': [1, 1],
            # Missing home_team and away_team
        })
        result = self.validator.validate_dataframe(df)

        assert not result.is_valid
        assert len(result.errors) >= 2

    def test_null_values_in_required_field(self):
        """Test that null values in required fields are detected."""
        df = pd.DataFrame({
            'id': [1, 2],
            'season': [2024, None],
            'week': [1, 1],
            'home_team': ['Alabama', 'Ohio State'],
            'away_team': ['Auburn', 'Michigan'],
        })
        result = self.validator.validate_dataframe(df)

        assert not result.is_valid


class TestPredictionValidator:
    """Tests for PredictionValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PredictionValidator()

    def test_valid_prediction(self):
        """Test validation of valid prediction."""
        result = self.validator.validate_prediction(10.5, vegas_line=-7.0)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_nan_prediction(self):
        """Test that NaN predictions are rejected."""
        result = self.validator.validate_prediction(float('nan'))

        assert not result.is_valid
        assert len(result.errors) == 1

    def test_extreme_prediction_warning(self):
        """Test that extreme predictions generate warnings."""
        result = self.validator.validate_prediction(60.0)

        assert result.is_valid
        assert result.has_warnings

    def test_large_disagreement_warning(self):
        """Test that large disagreement with Vegas generates warning."""
        # Model predicts home by 40, Vegas has home -3
        result = self.validator.validate_prediction(40.0, vegas_line=-3.0)

        assert result.is_valid
        assert result.has_warnings


class TestBettingLineValidator:
    """Tests for BettingLineValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = BettingLineValidator()

    def test_valid_spread(self):
        """Test validation of valid spread."""
        result = self.validator.validate_spread(-7.5)

        assert result.is_valid

    def test_nan_spread(self):
        """Test that NaN spread is rejected."""
        result = self.validator.validate_spread(float('nan'))

        assert not result.is_valid

    def test_unusual_spread_warning(self):
        """Test that unusual spread generates warning."""
        result = self.validator.validate_spread(-65.0)

        assert result.is_valid
        assert result.has_warnings


class TestValidateAndFixFeatures:
    """Tests for validate_and_fix_features helper function."""

    def test_successful_validation(self):
        """Test successful validation and fixing."""
        features = np.array([[1600, np.nan, 30, 25]])
        names = ['home_pregame_elo', 'away_pregame_elo',
                 'home_last5_score_avg', 'away_last5_score_avg']

        fixed, warnings = validate_and_fix_features(features, names)

        assert not np.isnan(fixed).any()
        assert len(warnings) >= 1

    def test_validation_error_on_wrong_shape(self):
        """Test that ValidationError is raised for invalid data."""
        features = np.array([[1600, 1400]])  # Too few features
        names = ['home_pregame_elo', 'away_pregame_elo',
                 'home_last5_score_avg', 'away_last5_score_avg']

        with pytest.raises(ValidationError):
            validate_and_fix_features(features, names)
