"""Tests for momentum tracking module."""

import pandas as pd
import pytest

from src.data.momentum import MomentumTracker, TeamMomentum


class TestTeamMomentum:
    """Tests for TeamMomentum dataclass."""

    def test_is_hot(self):
        """Test hot streak detection."""
        # 3+ wins = hot
        momentum = TeamMomentum(
            team="Alabama",
            season=2024,
            week=5,
            straight_up_streak=3,
        )
        assert momentum.is_hot

        # 2 wins = not hot
        momentum = TeamMomentum(
            team="Alabama",
            season=2024,
            week=5,
            straight_up_streak=2,
        )
        assert not momentum.is_hot

    def test_is_cold(self):
        """Test cold streak detection."""
        # 3+ losses = cold
        momentum = TeamMomentum(
            team="Alabama",
            season=2024,
            week=5,
            straight_up_streak=-3,
        )
        assert momentum.is_cold

        # 2 losses = not cold
        momentum = TeamMomentum(
            team="Alabama",
            season=2024,
            week=5,
            straight_up_streak=-2,
        )
        assert not momentum.is_cold

    def test_is_ats_hot(self):
        """Test ATS hot streak detection."""
        momentum = TeamMomentum(
            team="Alabama",
            season=2024,
            week=5,
            ats_streak=4,
        )
        assert momentum.is_ats_hot

    def test_momentum_score(self):
        """Test momentum score calculation."""
        # Hot team should have positive score
        hot_momentum = TeamMomentum(
            team="Alabama",
            season=2024,
            week=5,
            straight_up_streak=5,
            ats_streak=4,
            last5_record=(5, 0),
            last5_ats_record=(4, 1),
        )
        assert hot_momentum.momentum_score > 0

        # Cold team should have negative score
        cold_momentum = TeamMomentum(
            team="Alabama",
            season=2024,
            week=5,
            straight_up_streak=-4,
            ats_streak=-3,
            last5_record=(1, 4),
            last5_ats_record=(1, 4),
        )
        assert cold_momentum.momentum_score < 0


class TestMomentumTracker:
    """Tests for MomentumTracker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = MomentumTracker()

    def test_add_game(self):
        """Test adding a game to the tracker."""
        self.tracker.add_game(
            team="Alabama",
            opponent="Auburn",
            season=2024,
            week=1,
            points_scored=35,
            points_allowed=21,
            is_home=True,
            spread=-14.0,
        )

        momentum = self.tracker.get_momentum("Alabama", 2024, 2)
        assert momentum.team == "Alabama"

    def test_win_streak_calculation(self):
        """Test that win streaks are calculated correctly."""
        # Add 3 consecutive wins
        for week in range(1, 4):
            self.tracker.add_game(
                team="Alabama",
                opponent=f"Team{week}",
                season=2024,
                week=week,
                points_scored=35,
                points_allowed=21,
                is_home=True,
            )

        momentum = self.tracker.get_momentum("Alabama", 2024, 4)
        assert momentum.straight_up_streak == 3
        assert momentum.is_hot

    def test_loss_streak_calculation(self):
        """Test that loss streaks are calculated correctly."""
        # Add 3 consecutive losses
        for week in range(1, 4):
            self.tracker.add_game(
                team="Alabama",
                opponent=f"Team{week}",
                season=2024,
                week=week,
                points_scored=14,
                points_allowed=35,
                is_home=True,
            )

        momentum = self.tracker.get_momentum("Alabama", 2024, 4)
        assert momentum.straight_up_streak == -3
        assert momentum.is_cold

    def test_ats_streak(self):
        """Test ATS streak calculation."""
        # Win by 20 with spread -14 = cover
        self.tracker.add_game(
            team="Alabama",
            opponent="Auburn",
            season=2024,
            week=1,
            points_scored=35,
            points_allowed=15,
            is_home=True,
            spread=-14.0,  # Favored by 14, won by 20 = cover
        )

        momentum = self.tracker.get_momentum("Alabama", 2024, 2)
        assert momentum.ats_streak == 1

    def test_last5_record(self):
        """Test last 5 games record calculation."""
        # Add 3 wins and 2 losses
        results = [(35, 21), (28, 14), (42, 35), (14, 21), (10, 24)]
        for week, (scored, allowed) in enumerate(results, 1):
            self.tracker.add_game(
                team="Alabama",
                opponent=f"Team{week}",
                season=2024,
                week=week,
                points_scored=scored,
                points_allowed=allowed,
                is_home=True,
            )

        momentum = self.tracker.get_momentum("Alabama", 2024, 6)
        wins, losses = momentum.last5_record
        assert wins == 3
        assert losses == 2

    def test_scoring_trend(self):
        """Test scoring trend calculation."""
        # Add games with increasing scores
        scores = [20, 25, 30, 35, 40]
        for week, score in enumerate(scores, 1):
            self.tracker.add_game(
                team="Alabama",
                opponent=f"Team{week}",
                season=2024,
                week=week,
                points_scored=score,
                points_allowed=21,
                is_home=True,
            )

        momentum = self.tracker.get_momentum("Alabama", 2024, 6)
        # Trend should be positive (scores increasing)
        assert momentum.scoring_trend > 0

    def test_momentum_differential(self):
        """Test momentum differential between two teams."""
        # Add wins for Alabama
        for week in range(1, 4):
            self.tracker.add_game(
                team="Alabama",
                opponent=f"Team{week}",
                season=2024,
                week=week,
                points_scored=35,
                points_allowed=21,
                is_home=True,
            )

        # Add losses for Auburn
        for week in range(1, 4):
            self.tracker.add_game(
                team="Auburn",
                opponent=f"Team{week}",
                season=2024,
                week=week,
                points_scored=14,
                points_allowed=28,
                is_home=True,
            )

        diff = self.tracker.get_momentum_differential(
            "Alabama", "Auburn", 2024, 4
        )

        assert diff['momentum_diff'] > 0
        assert diff['su_streak_diff'] == 6  # 3 - (-3)
        assert diff['home_is_hot'] == 1
        assert diff['away_is_cold'] == 1

    def test_build_from_dataframe(self):
        """Test building tracker from DataFrame."""
        df = pd.DataFrame({
            'season': [2024, 2024, 2024],
            'week': [1, 2, 3],
            'home_team': ['Alabama', 'Alabama', 'Alabama'],
            'away_team': ['Auburn', 'Georgia', 'LSU'],
            'home_points': [35, 28, 42],
            'away_points': [21, 14, 35],
        })

        self.tracker.build_from_dataframe(df)

        momentum = self.tracker.get_momentum("Alabama", 2024, 4)
        assert momentum.straight_up_streak == 3

    def test_add_momentum_features(self):
        """Test adding momentum features to DataFrame."""
        # First build some history
        for week in range(1, 4):
            self.tracker.add_game(
                team="Alabama",
                opponent=f"Team{week}",
                season=2024,
                week=week,
                points_scored=35,
                points_allowed=21,
                is_home=True,
            )
            self.tracker.add_game(
                team="Auburn",
                opponent=f"OtherTeam{week}",
                season=2024,
                week=week,
                points_scored=21,
                points_allowed=28,
                is_home=True,
            )

        # Now add features to a games DataFrame
        df = pd.DataFrame({
            'season': [2024],
            'week': [4],
            'home_team': ['Alabama'],
            'away_team': ['Auburn'],
        })

        result = self.tracker.add_momentum_features(df)

        assert 'momentum_diff' in result.columns
        assert 'su_streak_diff' in result.columns
        assert result['momentum_diff'].iloc[0] > 0  # Alabama has better momentum
