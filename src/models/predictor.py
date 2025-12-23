"""
Game Predictor for Sharp Sports Predictor.

High-level prediction interface that combines:
- Model loading and prediction
- Feature validation
- Monte Carlo simulation
- Kelly Criterion sizing
- Bet recommendations
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.config import get_config
from ..utils.logging_config import get_logger, StructuredLogger
from ..utils.validation import FeatureValidator, validate_and_fix_features
from .ensemble import StackingModel

logger = get_logger(__name__)
structured_logger = StructuredLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a game prediction."""

    home_team: str
    away_team: str

    # Model prediction
    predicted_margin: float
    model_line: float  # Negative of predicted_margin

    # Vegas comparison
    vegas_line: float
    edge: float  # Positive = value on home, Negative = value on away

    # Confidence metrics
    model_variance: float
    individual_predictions: List[float]
    high_variance: bool

    # Monte Carlo results
    cover_probability: float
    signal: str  # 'BUY', 'FADE', or 'PASS'

    # Kelly sizing
    bet_size: float
    kelly_fraction: float
    win_probability: float

    # Recommendation
    recommended_bet: str
    confidence_tier: str

    @property
    def is_actionable(self) -> bool:
        """Whether this prediction warrants a bet."""
        return (
            not self.high_variance and
            self.bet_size > 0 and
            self.signal != 'PASS'
        )


class GamePredictor:
    """
    High-level game prediction interface.

    Combines model prediction with Monte Carlo simulation
    and Kelly Criterion for complete bet recommendations.
    """

    def __init__(self, model: Optional[StackingModel] = None):
        """
        Initialize the game predictor.

        Args:
            model: Optional pre-loaded model
        """
        self.config = get_config()

        if model is not None:
            self._model = model
        else:
            self._model = StackingModel()
            self._model.load()

        # Feature validator
        self._feature_validator = FeatureValidator(
            self.config.features.v6_features
        )

    def predict(
        self,
        features: np.ndarray,
        vegas_line: float,
        home_team: str = "Home",
        away_team: str = "Away",
        bankroll: Optional[float] = None,
    ) -> PredictionResult:
        """
        Make a complete prediction for a game.

        Args:
            features: Feature array
            vegas_line: Current Vegas spread
            home_team: Home team name
            away_team: Away team name
            bankroll: Optional bankroll override

        Returns:
            PredictionResult with full prediction details
        """
        cfg = self.config.betting
        bankroll = bankroll or cfg.default_bankroll

        # Validate and fix features
        validation = self._feature_validator.validate(features, fix_invalid=True)
        if validation.warnings:
            for warning in validation.warnings[:3]:
                logger.warning(f"Feature validation: {warning}")
        features = validation.fixed_data

        # Get model prediction
        predicted_margin = self._model.predict(features)
        model_line = -predicted_margin
        edge = vegas_line - model_line

        # Get model variance
        model_variance, individual_predictions = self._model.get_model_variance(features)
        high_variance = model_variance > cfg.high_variance_threshold

        # Monte Carlo simulation
        cover_prob = self._simulate_game(predicted_margin, vegas_line)
        signal = self._get_signal(cover_prob)

        # Kelly sizing
        kelly_result = self._calculate_kelly(
            predicted_margin, vegas_line, bankroll
        )

        # Override for high variance
        if high_variance:
            signal = "PASS"
            kelly_result['bet_size'] = 0

        # Build recommendation
        recommended_bet = self._get_recommendation(
            edge, home_team, away_team, vegas_line, high_variance
        )

        confidence_tier = self._get_confidence_tier(cover_prob, model_variance)

        result = PredictionResult(
            home_team=home_team,
            away_team=away_team,
            predicted_margin=predicted_margin,
            model_line=model_line,
            vegas_line=vegas_line,
            edge=edge,
            model_variance=model_variance,
            individual_predictions=individual_predictions,
            high_variance=high_variance,
            cover_probability=cover_prob,
            signal=signal,
            bet_size=kelly_result['bet_size'],
            kelly_fraction=kelly_result['kelly_fraction'],
            win_probability=kelly_result['win_prob'],
            recommended_bet=recommended_bet,
            confidence_tier=confidence_tier,
        )

        structured_logger.log_prediction(
            game=f"{away_team} @ {home_team}",
            predicted_margin=predicted_margin,
            vegas_line=vegas_line,
            edge=edge,
            confidence=cover_prob,
        )

        if result.is_actionable:
            structured_logger.log_bet_recommendation(
                game=f"{away_team} @ {home_team}",
                side=recommended_bet,
                bet_size=kelly_result['bet_size'],
                kelly_pct=kelly_result['kelly_fraction'],
                win_prob=kelly_result['win_prob'],
            )

        return result

    def _simulate_game(
        self,
        predicted_margin: float,
        vegas_line: float,
        simulations: Optional[int] = None,
        std_dev: Optional[float] = None,
    ) -> float:
        """
        Monte Carlo simulation for cover probability.

        Args:
            predicted_margin: Model's predicted margin
            vegas_line: Vegas spread
            simulations: Number of simulations
            std_dev: Standard deviation for margin distribution

        Returns:
            Probability of covering the spread
        """
        cfg = self.config.betting
        simulations = simulations or cfg.monte_carlo_simulations
        std_dev = std_dev or cfg.monte_carlo_std_dev

        # Generate simulated margins
        simulated_margins = np.random.normal(predicted_margin, std_dev, simulations)

        # Count covers
        # Home covers if margin > -spread
        covers = simulated_margins > (-vegas_line)

        return float(covers.mean())

    def _get_signal(self, cover_prob: float) -> str:
        """Get betting signal from cover probability."""
        cfg = self.config.betting

        if cover_prob >= cfg.buy_threshold:
            return "BUY"
        elif cover_prob <= cfg.fade_threshold:
            return "FADE"
        else:
            return "PASS"

    def _calculate_kelly(
        self,
        predicted_margin: float,
        vegas_line: float,
        bankroll: float,
    ) -> Dict[str, float]:
        """
        Calculate Kelly Criterion bet sizing.

        Args:
            predicted_margin: Model's predicted margin
            vegas_line: Vegas spread
            bankroll: Current bankroll

        Returns:
            Dict with bet_size, kelly_fraction, win_prob, edge
        """
        cfg = self.config.betting

        # Estimate win probability from edge
        edge = vegas_line - (-predicted_margin)

        if edge > 0:
            win_prob = 0.50 + (edge / 100)
        else:
            win_prob = 0.50 + (edge / 100)

        win_prob = max(0.01, min(0.99, win_prob))

        # Calculate Kelly fraction
        # Standard -110 odds
        decimal_odds = 1 + (100 / abs(cfg.default_american_odds))
        b = decimal_odds - 1
        q = 1 - win_prob

        kelly_fraction = (b * win_prob - q) / b
        kelly_fraction = max(0, kelly_fraction)

        # Apply fractional Kelly
        fractional = cfg.kelly_fraction
        bet_size = bankroll * kelly_fraction * fractional

        # Apply max bet constraint
        max_bet = bankroll * cfg.max_bankroll_pct
        bet_size = min(bet_size, max_bet, cfg.max_bet)

        # Apply min bet threshold
        if bet_size < cfg.min_bet:
            bet_size = 0

        return {
            'bet_size': round(bet_size, 2),
            'kelly_fraction': kelly_fraction,
            'win_prob': win_prob,
            'edge': edge,
        }

    def _get_recommendation(
        self,
        edge: float,
        home_team: str,
        away_team: str,
        vegas_line: float,
        high_variance: bool,
    ) -> str:
        """Generate bet recommendation string."""
        if high_variance:
            return "PASS (High Variance)"

        cfg = self.config.betting

        if abs(edge) < cfg.min_edge_to_bet:
            return "Pass"
        elif edge > 0:
            return f"HOME {home_team} {vegas_line:+.1f}"
        else:
            return f"AWAY {away_team} {-vegas_line:+.1f}"

    def _get_confidence_tier(
        self,
        cover_prob: float,
        variance: float,
    ) -> str:
        """Get confidence tier for display."""
        cfg = self.config.betting

        if variance > cfg.high_variance_threshold:
            return "UNCERTAIN"
        elif cover_prob >= 0.60 or cover_prob <= 0.40:
            return "HIGH"
        elif cover_prob >= 0.55 or cover_prob <= 0.45:
            return "MEDIUM"
        else:
            return "LOW"

    def batch_predict(
        self,
        games: List[Dict],
        bankroll: Optional[float] = None,
    ) -> List[PredictionResult]:
        """
        Make predictions for multiple games.

        Args:
            games: List of game dicts with 'features', 'vegas_line', etc.
            bankroll: Optional bankroll override

        Returns:
            List of PredictionResult objects
        """
        results = []

        for game in games:
            result = self.predict(
                features=game['features'],
                vegas_line=game['vegas_line'],
                home_team=game.get('home_team', 'Home'),
                away_team=game.get('away_team', 'Away'),
                bankroll=bankroll,
            )
            results.append(result)

        return results
