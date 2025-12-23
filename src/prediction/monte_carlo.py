"""
Monte Carlo Simulation for Sharp Sports Predictor.

Simulates game outcomes to calculate cover probabilities
and betting confidence levels.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..utils.config import get_config


@dataclass
class SimulationResult:
    """Result of a Monte Carlo simulation."""

    cover_probability: float
    simulated_margins: np.ndarray
    mean_margin: float
    std_margin: float
    confidence_interval: Tuple[float, float]

    @property
    def signal(self) -> str:
        """Get betting signal based on cover probability."""
        if self.cover_probability >= 0.55:
            return "BUY"
        elif self.cover_probability <= 0.45:
            return "FADE"
        else:
            return "PASS"


class MonteCarloSimulator:
    """
    Monte Carlo simulator for spread betting.

    Simulates thousands of game outcomes to estimate
    the probability of covering the spread.
    """

    def __init__(
        self,
        simulations: Optional[int] = None,
        std_dev: Optional[float] = None,
    ):
        """
        Initialize the simulator.

        Args:
            simulations: Number of simulations (default from config)
            std_dev: Standard deviation for margin distribution
        """
        config = get_config()

        self.simulations = simulations or config.betting.monte_carlo_simulations
        self.std_dev = std_dev or config.betting.monte_carlo_std_dev
        self.buy_threshold = config.betting.buy_threshold
        self.fade_threshold = config.betting.fade_threshold

    def simulate(
        self,
        predicted_margin: float,
        spread_line: float,
        std_dev: Optional[float] = None,
    ) -> SimulationResult:
        """
        Simulate a game to calculate cover probability.

        Args:
            predicted_margin: Model's predicted margin (positive = home wins)
            spread_line: Vegas spread (negative = home favored)
            std_dev: Optional std dev override

        Returns:
            SimulationResult with probability and distribution info

        Example:
            If model predicts home wins by 7 (margin=7) and spread is -3
            (home favored by 3), home covers if they win by MORE than 3.
            So we check if simulated margin > 3.
        """
        std_dev = std_dev or self.std_dev

        # Generate simulated margins
        simulated_margins = np.random.normal(
            predicted_margin,
            std_dev,
            self.simulations
        )

        # Calculate cover probability
        # Home covers if their margin beats the spread
        # If spread is -7 (home favored), home needs margin > 7 to cover
        # If spread is +7 (home underdog), home needs margin > -7 to cover
        covers = simulated_margins > (-spread_line)
        cover_probability = float(covers.mean())

        # Calculate confidence interval (95%)
        sorted_margins = np.sort(simulated_margins)
        ci_lower = sorted_margins[int(0.025 * self.simulations)]
        ci_upper = sorted_margins[int(0.975 * self.simulations)]

        return SimulationResult(
            cover_probability=cover_probability,
            simulated_margins=simulated_margins,
            mean_margin=float(simulated_margins.mean()),
            std_margin=float(simulated_margins.std()),
            confidence_interval=(ci_lower, ci_upper),
        )

    def get_signal(
        self,
        cover_prob: float,
        buy_threshold: Optional[float] = None,
        fade_threshold: Optional[float] = None,
    ) -> Tuple[str, str]:
        """
        Get betting signal based on cover probability.

        Args:
            cover_prob: Probability of covering the spread
            buy_threshold: Probability above which to BUY
            fade_threshold: Probability below which to FADE

        Returns:
            Tuple of (signal_code, signal_label)
        """
        buy_threshold = buy_threshold or self.buy_threshold
        fade_threshold = fade_threshold or self.fade_threshold

        if cover_prob >= buy_threshold:
            return "BUY", "BUY"
        elif cover_prob <= fade_threshold:
            return "FADE", "FADE"
        else:
            return "PASS", "PASS"

    @staticmethod
    def format_probability(prob: float) -> str:
        """Format probability as percentage string."""
        return f"{prob * 100:.0f}%"

    def simulate_totals(
        self,
        predicted_total: float,
        vegas_total: float,
        std_dev: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Simulate over/under probabilities.

        Args:
            predicted_total: Model's predicted total points
            vegas_total: Vegas over/under line
            std_dev: Standard deviation (default 12.5 for totals)

        Returns:
            Tuple of (over_probability, under_probability)
        """
        std_dev = std_dev or 12.5  # Totals typically have lower variance

        simulated_totals = np.random.normal(
            predicted_total,
            std_dev,
            self.simulations
        )

        over_prob = float((simulated_totals > vegas_total).mean())
        under_prob = 1 - over_prob

        return over_prob, under_prob


# Convenience functions for compatibility
def simulate_game(
    predicted_margin: float,
    spread_line: float,
    std_dev: float = 14.0,
    simulations: int = 10000,
) -> dict:
    """
    Simulate a game (compatibility function).

    Args:
        predicted_margin: Model's predicted margin
        spread_line: Vegas spread line
        std_dev: Standard deviation
        simulations: Number of simulations

    Returns:
        Dict with cover_probability, simulated_margins, mean_margin, std_margin
    """
    simulator = MonteCarloSimulator(simulations=simulations, std_dev=std_dev)
    result = simulator.simulate(predicted_margin, spread_line)

    return {
        'cover_probability': result.cover_probability,
        'simulated_margins': result.simulated_margins,
        'mean_margin': result.mean_margin,
        'std_margin': result.std_margin,
    }


def get_bet_signal(
    cover_prob: float,
    buy_threshold: float = 0.55,
    fade_threshold: float = 0.45,
) -> Tuple[str, str]:
    """
    Get betting signal (compatibility function).

    Args:
        cover_prob: Probability of covering
        buy_threshold: Threshold for BUY signal
        fade_threshold: Threshold for FADE signal

    Returns:
        Tuple of (signal_code, signal_label)
    """
    simulator = MonteCarloSimulator()
    return simulator.get_signal(cover_prob, buy_threshold, fade_threshold)


def format_win_prob(cover_prob: float) -> str:
    """Format cover probability as percentage string."""
    return MonteCarloSimulator.format_probability(cover_prob)
