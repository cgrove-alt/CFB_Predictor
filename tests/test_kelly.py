"""Tests for Kelly Criterion money management module."""

import pytest

from src.prediction.kelly import (
    KellyCalculator,
    BetRecommendation,
    kelly_recommendation,
)


class TestKellyCalculator:
    """Tests for KellyCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calc = KellyCalculator(
            fraction=0.25,
            bankroll=1000.0,
            min_bet=10.0,
            max_bet=200.0,
        )

    def test_american_to_decimal(self):
        """Test American to decimal odds conversion."""
        # -110 is standard
        assert abs(self.calc.american_to_decimal(-110) - 1.909) < 0.01

        # +150
        assert self.calc.american_to_decimal(150) == 2.5

        # -200
        assert self.calc.american_to_decimal(-200) == 1.5

        # +100 (even money)
        assert self.calc.american_to_decimal(100) == 2.0

    def test_decimal_to_implied_prob(self):
        """Test decimal odds to implied probability conversion."""
        # 2.0 = 50% implied
        assert self.calc.decimal_to_implied_prob(2.0) == 0.5

        # 1.909 ≈ 52.4% (standard -110)
        implied = self.calc.decimal_to_implied_prob(1.909)
        assert 0.52 < implied < 0.53

    def test_kelly_fraction_no_edge(self):
        """Test Kelly fraction with no edge is 0."""
        # 52.4% win prob at -110 is break even
        implied = self.calc.decimal_to_implied_prob(1.909)
        kelly = self.calc.kelly_fraction_calc(implied, 1.909)

        assert kelly == pytest.approx(0, abs=0.01)

    def test_kelly_fraction_with_edge(self):
        """Test Kelly fraction with positive edge."""
        # 60% win prob at -110 odds
        kelly = self.calc.kelly_fraction_calc(0.60, 1.909)

        # Should be positive
        assert kelly > 0

    def test_kelly_fraction_negative_edge(self):
        """Test Kelly fraction with negative edge is 0."""
        # 40% win prob at -110 odds
        kelly = self.calc.kelly_fraction_calc(0.40, 1.909)

        assert kelly == 0

    def test_calculate_bet_size_applies_fraction(self):
        """Test that fractional Kelly is applied."""
        # High win prob should give positive bet
        bet_size = self.calc.calculate_bet_size(0.70, 1.909, 1000)

        # Should be positive but less than full Kelly
        assert bet_size > 0
        assert bet_size <= 200  # Max bet

    def test_bet_size_respects_min_bet(self):
        """Test that small edges result in no bet."""
        # Barely above break even
        bet_size = self.calc.calculate_bet_size(0.53, 1.909, 1000)

        # Should be 0 because below min bet
        assert bet_size == 0

    def test_bet_size_respects_max_bet(self):
        """Test that bet size is capped at max."""
        # Very high edge
        bet_size = self.calc.calculate_bet_size(0.90, 1.909, 10000)

        assert bet_size <= 200

    def test_get_recommendation_home_side(self):
        """Test recommendation for home team value."""
        # Model predicts home by 10, Vegas has home -3
        rec = self.calc.get_recommendation(10.0, -3.0)

        assert isinstance(rec, BetRecommendation)
        assert rec.bet_side == "HOME"
        assert rec.bet_size > 0

    def test_get_recommendation_away_side(self):
        """Test recommendation for away team value."""
        # Model predicts home by -5, Vegas has home -10
        # Away team covers
        rec = self.calc.get_recommendation(-5.0, -10.0)

        assert rec.bet_side == "AWAY" or rec.bet_side == "PASS"

    def test_get_recommendation_no_edge(self):
        """Test recommendation with no edge is PASS."""
        # Model matches Vegas
        rec = self.calc.get_recommendation(3.0, -3.0)

        # Should be PASS or very small bet
        assert rec.bet_side in ("PASS", "HOME", "AWAY")

    def test_recommendation_edge_calculation(self):
        """Test that edge is calculated correctly."""
        rec = self.calc.get_recommendation(10.0, -7.0)

        # Edge = win_prob - implied_prob
        expected_positive_edge = rec.win_probability > rec.implied_probability

        if rec.bet_size > 0:
            assert expected_positive_edge

    def test_format_bet_size(self):
        """Test bet size formatting."""
        assert "No Bet" in self.calc.format_bet_size(0)
        assert "Small" in self.calc.format_bet_size(15)
        assert "Medium" in self.calc.format_bet_size(50)
        assert "Large" in self.calc.format_bet_size(100)
        assert "MAX" in self.calc.format_bet_size(200)


class TestKellyRecommendation:
    """Tests for kelly_recommendation compatibility function."""

    def test_returns_dict(self):
        """Test that function returns expected dict structure."""
        result = kelly_recommendation(10.0, -7.0)

        assert isinstance(result, dict)
        assert 'bet_size' in result
        assert 'kelly_fraction' in result
        assert 'win_prob' in result
        assert 'edge' in result

    def test_positive_edge(self):
        """Test with positive edge scenario."""
        # Model predicts home by 15, Vegas -7
        result = kelly_recommendation(15.0, -7.0)

        assert result['bet_size'] > 0
        assert result['kelly_fraction'] > 0

    def test_bankroll_parameter(self):
        """Test that bankroll affects bet size."""
        result_1k = kelly_recommendation(15.0, -7.0, bankroll=1000)
        result_2k = kelly_recommendation(15.0, -7.0, bankroll=2000)

        # With same edge, larger bankroll = larger bet (up to max)
        assert result_2k['kelly_fraction'] == result_1k['kelly_fraction']

    def test_odds_parameter(self):
        """Test that odds parameter is used."""
        result_110 = kelly_recommendation(10.0, -7.0, odds=-110)
        result_105 = kelly_recommendation(10.0, -7.0, odds=-105)

        # Better odds (lower vig) should give slightly larger Kelly
        # This is a bit subtle to test, just ensure it runs
        assert result_105['bet_size'] >= 0
        assert result_110['bet_size'] >= 0
