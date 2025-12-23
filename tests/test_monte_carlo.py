"""Tests for Monte Carlo simulation module."""

import numpy as np
import pytest

from src.prediction.monte_carlo import (
    MonteCarloSimulator,
    SimulationResult,
    simulate_game,
    get_bet_signal,
    format_win_prob,
)


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use fixed seed for reproducibility
        np.random.seed(42)
        self.simulator = MonteCarloSimulator(simulations=10000, std_dev=14.0)

    def test_simulation_returns_result(self):
        """Test that simulate returns a SimulationResult."""
        result = self.simulator.simulate(10.0, -7.0)

        assert isinstance(result, SimulationResult)
        assert 0 <= result.cover_probability <= 1
        assert len(result.simulated_margins) == 10000

    def test_home_favorite_should_cover(self):
        """Test that a large home favorite edge results in high cover prob."""
        # Model predicts home by 20, Vegas has home -7
        # Home should easily cover
        result = self.simulator.simulate(20.0, -7.0)

        assert result.cover_probability > 0.7

    def test_away_value_low_cover_prob(self):
        """Test that away value results in low home cover prob."""
        # Model predicts home by 3, Vegas has home -14
        # Home is unlikely to cover
        result = self.simulator.simulate(3.0, -14.0)

        assert result.cover_probability < 0.3

    def test_pick_em_close_to_50(self):
        """Test that pick'em with no edge is close to 50%."""
        # Model predicts home by 0, spread is 0
        result = self.simulator.simulate(0.0, 0.0)

        assert 0.45 <= result.cover_probability <= 0.55

    def test_confidence_interval(self):
        """Test that confidence interval is calculated correctly."""
        result = self.simulator.simulate(10.0, -7.0)

        ci_lower, ci_upper = result.confidence_interval
        assert ci_lower < result.mean_margin < ci_upper
        # 95% CI should be roughly ±1.96 * std
        assert ci_upper - ci_lower < 4 * result.std_margin

    def test_signal_property(self):
        """Test the signal property on SimulationResult."""
        # High cover prob = BUY
        result = self.simulator.simulate(20.0, -7.0)
        assert result.signal == "BUY"

        # Low cover prob = FADE
        result = self.simulator.simulate(-10.0, 7.0)
        assert result.signal == "FADE"


class TestGetSignal:
    """Tests for get_bet_signal function."""

    def test_buy_signal(self):
        """Test that high probability returns BUY."""
        signal, label = get_bet_signal(0.60)

        assert signal == "BUY"
        assert label == "BUY"

    def test_fade_signal(self):
        """Test that low probability returns FADE."""
        signal, label = get_bet_signal(0.40)

        assert signal == "FADE"
        assert label == "FADE"

    def test_pass_signal(self):
        """Test that middle probability returns PASS."""
        signal, label = get_bet_signal(0.50)

        assert signal == "PASS"
        assert label == "PASS"

    def test_custom_thresholds(self):
        """Test signal with custom thresholds."""
        # 0.52 is above default fade (0.45) but below custom fade (0.53)
        signal, label = get_bet_signal(0.52, buy_threshold=0.60, fade_threshold=0.53)

        assert signal == "FADE"


class TestSimulateGame:
    """Tests for simulate_game compatibility function."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)

    def test_returns_dict(self):
        """Test that function returns expected dict structure."""
        result = simulate_game(10.0, -7.0)

        assert isinstance(result, dict)
        assert 'cover_probability' in result
        assert 'simulated_margins' in result
        assert 'mean_margin' in result
        assert 'std_margin' in result

    def test_simulations_parameter(self):
        """Test that simulations parameter is respected."""
        result = simulate_game(10.0, -7.0, simulations=1000)

        assert len(result['simulated_margins']) == 1000

    def test_std_dev_parameter(self):
        """Test that std_dev parameter affects results."""
        # With very low std_dev, result should be very close to predicted
        result = simulate_game(10.0, -7.0, std_dev=0.1)

        assert abs(result['mean_margin'] - 10.0) < 0.1


class TestFormatWinProb:
    """Tests for format_win_prob function."""

    def test_format_50_percent(self):
        """Test formatting 50%."""
        assert format_win_prob(0.50) == "50%"

    def test_format_high_percent(self):
        """Test formatting high percentage."""
        assert format_win_prob(0.75) == "75%"

    def test_format_low_percent(self):
        """Test formatting low percentage."""
        assert format_win_prob(0.25) == "25%"

    def test_format_rounds(self):
        """Test that formatting rounds correctly."""
        assert format_win_prob(0.555) == "56%"
        assert format_win_prob(0.554) == "55%"


class TestSimulateTotals:
    """Tests for over/under simulation."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.simulator = MonteCarloSimulator()

    def test_over_value(self):
        """Test that predicted total above Vegas favors over."""
        # Model predicts 55, Vegas is 50
        over_prob, under_prob = self.simulator.simulate_totals(55.0, 50.0)

        assert over_prob > under_prob
        assert over_prob + under_prob == 1.0

    def test_under_value(self):
        """Test that predicted total below Vegas favors under."""
        # Model predicts 45, Vegas is 50
        over_prob, under_prob = self.simulator.simulate_totals(45.0, 50.0)

        assert under_prob > over_prob

    def test_no_edge(self):
        """Test that equal predictions give ~50/50."""
        over_prob, under_prob = self.simulator.simulate_totals(50.0, 50.0)

        assert 0.45 <= over_prob <= 0.55
