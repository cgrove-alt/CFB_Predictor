"""
Kelly Criterion Money Management for Sharp Sports Predictor.

Provides optimal bet sizing using the Kelly Criterion:
- Full Kelly for maximum growth
- Fractional Kelly (0.25x default) for reduced volatility
"""

from dataclasses import dataclass
from typing import Optional

from scipy import stats

from ..utils.config import get_config


@dataclass
class BetRecommendation:
    """Complete bet recommendation with sizing."""

    bet_side: str  # 'HOME', 'AWAY', or 'PASS'
    bet_size: float
    kelly_fraction: float
    win_probability: float
    implied_probability: float
    edge_percentage: float
    decimal_odds: float
    confidence_tier: str


class KellyCalculator:
    """
    Kelly Criterion calculator for sports betting.

    The Kelly Criterion determines the optimal bet size to maximize
    long-term growth while minimizing risk of ruin.

    Formula: Kelly % = (p * b - q) / b
    Where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = net odds (decimal_odds - 1)
    """

    def __init__(
        self,
        fraction: Optional[float] = None,
        bankroll: Optional[float] = None,
        min_bet: Optional[float] = None,
        max_bet: Optional[float] = None,
    ):
        """
        Initialize the Kelly calculator.

        Args:
            fraction: Kelly fraction (default 0.25 = quarter Kelly)
            bankroll: Starting bankroll
            min_bet: Minimum bet size
            max_bet: Maximum bet size
        """
        config = get_config()
        betting = config.betting

        self.fraction = fraction or betting.kelly_fraction
        self.bankroll = bankroll or betting.default_bankroll
        self.min_bet = min_bet or betting.min_bet
        self.max_bet = max_bet or betting.max_bet
        self.default_odds = betting.default_american_odds

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """
        Convert American odds to Decimal odds.

        Examples:
            -110 -> 1.909
            +150 -> 2.500
        """
        if american_odds >= 100:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    @staticmethod
    def decimal_to_implied_prob(decimal_odds: float) -> float:
        """
        Convert decimal odds to implied probability.

        Example:
            1.909 -> 52.4% (standard -110 vig)
        """
        return 1 / decimal_odds

    def margin_to_win_prob(
        self,
        predicted_margin: float,
        spread: float,
        std_dev: float = 13.5,
    ) -> float:
        """
        Convert predicted margin and spread to win probability.

        Uses normal distribution to estimate probability of covering.

        Args:
            predicted_margin: Model's predicted home margin
            spread: Vegas spread (negative = home favored)
            std_dev: Standard deviation of prediction errors

        Returns:
            Probability of the bet winning
        """
        # Edge is how much better we think team will do vs spread
        edge = predicted_margin - (-spread)

        if edge > 0:
            # Betting HOME to cover
            win_prob = stats.norm.cdf(edge / std_dev)
        else:
            # Betting AWAY to cover
            win_prob = stats.norm.cdf(-edge / std_dev)

        return float(win_prob)

    def kelly_fraction_calc(
        self,
        win_prob: float,
        decimal_odds: float,
    ) -> float:
        """
        Calculate Kelly Criterion bet fraction.

        Args:
            win_prob: Probability of winning (0-1)
            decimal_odds: Decimal odds

        Returns:
            Kelly fraction (0-1), or 0 if no edge
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        q = 1 - win_prob
        b = decimal_odds - 1  # Net odds

        kelly = (win_prob * b - q) / b

        # Never bet negative Kelly (no edge)
        return max(0.0, kelly)

    def calculate_bet_size(
        self,
        win_prob: float,
        decimal_odds: Optional[float] = None,
        bankroll: Optional[float] = None,
    ) -> float:
        """
        Calculate actual bet size in dollars.

        Args:
            win_prob: Probability of winning
            decimal_odds: Decimal odds
            bankroll: Current bankroll

        Returns:
            Bet size in dollars
        """
        decimal_odds = decimal_odds or self.american_to_decimal(self.default_odds)
        bankroll = bankroll or self.bankroll

        full_kelly = self.kelly_fraction_calc(win_prob, decimal_odds)
        fractional_kelly = full_kelly * self.fraction

        bet_size = bankroll * fractional_kelly

        # Apply constraints
        if bet_size < self.min_bet:
            return 0.0

        return min(bet_size, self.max_bet)

    def get_recommendation(
        self,
        predicted_margin: float,
        spread: float,
        american_odds: Optional[int] = None,
        bankroll: Optional[float] = None,
        std_dev: float = 13.5,
    ) -> BetRecommendation:
        """
        Get complete bet recommendation.

        Args:
            predicted_margin: Model's predicted home margin
            spread: Vegas spread
            american_odds: American odds (default -110)
            bankroll: Current bankroll
            std_dev: Model standard deviation

        Returns:
            BetRecommendation with full details
        """
        american_odds = american_odds or self.default_odds
        bankroll = bankroll or self.bankroll

        decimal_odds = self.american_to_decimal(american_odds)
        win_prob = self.margin_to_win_prob(predicted_margin, spread, std_dev)

        # Determine bet side
        edge = predicted_margin - (-spread)
        if edge > 0:
            bet_side = "HOME"
        else:
            bet_side = "AWAY"

        # Calculate Kelly
        full_kelly = self.kelly_fraction_calc(win_prob, decimal_odds)
        bet_size = self.calculate_bet_size(win_prob, decimal_odds, bankroll)

        # If no edge, pass
        if bet_size == 0:
            bet_side = "PASS"

        # Calculate edge percentage
        implied_prob = self.decimal_to_implied_prob(decimal_odds)
        edge_pct = (win_prob - implied_prob) * 100

        # Confidence tier
        if win_prob >= 0.60:
            confidence = "HIGH"
        elif win_prob >= 0.55:
            confidence = "MEDIUM"
        elif win_prob >= 0.52:
            confidence = "LOW"
        else:
            confidence = "PASS"

        return BetRecommendation(
            bet_side=bet_side,
            bet_size=round(bet_size, 2),
            kelly_fraction=full_kelly * self.fraction,
            win_probability=win_prob,
            implied_probability=implied_prob,
            edge_percentage=edge_pct,
            decimal_odds=decimal_odds,
            confidence_tier=confidence,
        )

    def format_bet_size(self, bet_size: float) -> str:
        """Format bet size for display."""
        if bet_size == 0:
            return "No Bet"
        elif bet_size < 25:
            return f"${bet_size:.0f} (Small)"
        elif bet_size < 75:
            return f"${bet_size:.0f} (Medium)"
        elif bet_size < 150:
            return f"${bet_size:.0f} (Large)"
        else:
            return f"${bet_size:.0f} (MAX)"


# Convenience functions for compatibility
def kelly_recommendation(
    model_margin: float,
    vegas_line: float,
    bankroll: float = 1000,
    odds: int = -110,
) -> dict:
    """
    Calculate Kelly Criterion bet recommendation (compatibility function).

    Args:
        model_margin: Model's predicted margin
        vegas_line: Vegas spread
        bankroll: Current bankroll
        odds: American odds

    Returns:
        Dict with bet_size, kelly_fraction, win_prob, edge
    """
    calc = KellyCalculator(bankroll=bankroll)
    rec = calc.get_recommendation(model_margin, vegas_line, odds, bankroll)

    return {
        'bet_size': rec.bet_size,
        'kelly_fraction': rec.kelly_fraction,
        'win_prob': rec.win_probability,
        'edge': rec.edge_percentage,
    }
