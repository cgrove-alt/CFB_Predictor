"""
Momentum and Streak Tracking for Sharp Sports Predictor.

Tracks team momentum factors:
- Win/loss streaks
- ATS (Against The Spread) streaks
- Scoring trends
- Performance vs expectations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TeamMomentum:
    """Momentum metrics for a single team."""

    team: str
    season: int
    week: int

    # Win/Loss streaks (positive = wins, negative = losses)
    straight_up_streak: int = 0

    # ATS streaks (positive = covers, negative = non-covers)
    ats_streak: int = 0

    # Recent performance
    last5_record: Tuple[int, int] = (0, 0)  # (wins, losses)
    last5_ats_record: Tuple[int, int] = (0, 0)  # (covers, non-covers)

    # Scoring momentum
    scoring_trend: float = 0.0  # Positive = trending up
    defense_trend: float = 0.0  # Negative = improving (fewer points allowed)

    # Performance vs expectations
    avg_margin_vs_spread: float = 0.0  # How much team beats/misses spread by

    # Situational
    home_streak: int = 0  # Consecutive home wins
    away_streak: int = 0  # Consecutive away wins
    conference_streak: int = 0  # Conference game streak

    @property
    def is_hot(self) -> bool:
        """Team is on a hot streak (3+ wins)."""
        return self.straight_up_streak >= 3

    @property
    def is_cold(self) -> bool:
        """Team is on a cold streak (3+ losses)."""
        return self.straight_up_streak <= -3

    @property
    def is_ats_hot(self) -> bool:
        """Team is hot against the spread (3+ covers)."""
        return self.ats_streak >= 3

    @property
    def is_ats_cold(self) -> bool:
        """Team is cold against the spread (3+ non-covers)."""
        return self.ats_streak <= -3

    @property
    def momentum_score(self) -> float:
        """
        Calculate overall momentum score.

        Positive = good momentum, Negative = bad momentum
        """
        score = 0.0

        # Straight up streak contribution
        score += self.straight_up_streak * 0.5

        # ATS streak contribution (more important for betting)
        score += self.ats_streak * 1.0

        # Recent record contribution
        wins, losses = self.last5_record
        if wins + losses > 0:
            score += (wins - losses) / (wins + losses) * 2

        # ATS record contribution
        covers, non_covers = self.last5_ats_record
        if covers + non_covers > 0:
            score += (covers - non_covers) / (covers + non_covers) * 3

        # Scoring trends
        score += self.scoring_trend * 0.1
        score -= self.defense_trend * 0.1  # Lower is better for defense

        return score


class MomentumTracker:
    """
    Tracks momentum and streaks for all teams.

    This class maintains running momentum calculations as games are played,
    enabling momentum features for predictions.
    """

    def __init__(self):
        """Initialize the momentum tracker."""
        self._team_games: Dict[str, List[dict]] = {}
        self._team_momentum: Dict[str, TeamMomentum] = {}

    def add_game(
        self,
        team: str,
        opponent: str,
        season: int,
        week: int,
        points_scored: int,
        points_allowed: int,
        is_home: bool,
        spread: Optional[float] = None,
        is_conference: bool = False,
    ) -> None:
        """
        Add a completed game to the tracker.

        Args:
            team: Team name
            opponent: Opponent name
            season: Season year
            week: Week number
            points_scored: Points scored by team
            points_allowed: Points allowed by team
            is_home: Whether team was home
            spread: Vegas spread (from team's perspective)
            is_conference: Whether this was a conference game
        """
        if team not in self._team_games:
            self._team_games[team] = []

        # Calculate game result
        margin = points_scored - points_allowed
        won = margin > 0

        # Calculate ATS result if spread available
        covered = None
        if spread is not None:
            # Spread is from team's perspective
            # If spread is -7, team favored by 7, needs to win by more than 7
            # If spread is +7, team is underdog, needs to not lose by more than 7
            covered = margin > -spread

        game_record = {
            'season': season,
            'week': week,
            'opponent': opponent,
            'points_scored': points_scored,
            'points_allowed': points_allowed,
            'margin': margin,
            'won': won,
            'spread': spread,
            'covered': covered,
            'is_home': is_home,
            'is_conference': is_conference,
        }

        self._team_games[team].append(game_record)
        self._update_momentum(team, season, week)

    def _update_momentum(self, team: str, season: int, week: int) -> None:
        """Update momentum calculations for a team."""
        games = self._get_recent_games(team, season, week, n=10)

        if not games:
            self._team_momentum[team] = TeamMomentum(team=team, season=season, week=week)
            return

        # Calculate streaks
        su_streak = self._calculate_streak(games, 'won')
        ats_streak = self._calculate_streak(
            [g for g in games if g['covered'] is not None],
            'covered'
        )

        # Calculate last 5 records
        last5 = games[:5]
        last5_wins = sum(1 for g in last5 if g['won'])
        last5_losses = len(last5) - last5_wins

        ats_games = [g for g in last5 if g['covered'] is not None]
        last5_covers = sum(1 for g in ats_games if g['covered'])
        last5_non_covers = len(ats_games) - last5_covers

        # Calculate scoring trends (reverse for chronological order - oldest to newest)
        scoring_trend = self._calculate_trend([g['points_scored'] for g in reversed(last5)])
        defense_trend = self._calculate_trend([g['points_allowed'] for g in reversed(last5)])

        # Calculate avg margin vs spread
        spread_margins = [
            g['margin'] - (-g['spread']) for g in games
            if g['spread'] is not None
        ]
        avg_vs_spread = np.mean(spread_margins) if spread_margins else 0.0

        # Home/away streaks
        home_games = [g for g in games if g['is_home']]
        away_games = [g for g in games if not g['is_home']]
        home_streak = self._calculate_streak(home_games, 'won') if home_games else 0
        away_streak = self._calculate_streak(away_games, 'won') if away_games else 0

        # Conference streak
        conf_games = [g for g in games if g['is_conference']]
        conf_streak = self._calculate_streak(conf_games, 'won') if conf_games else 0

        self._team_momentum[team] = TeamMomentum(
            team=team,
            season=season,
            week=week,
            straight_up_streak=su_streak,
            ats_streak=ats_streak,
            last5_record=(last5_wins, last5_losses),
            last5_ats_record=(last5_covers, last5_non_covers),
            scoring_trend=scoring_trend,
            defense_trend=defense_trend,
            avg_margin_vs_spread=avg_vs_spread,
            home_streak=home_streak,
            away_streak=away_streak,
            conference_streak=conf_streak,
        )

    def _get_recent_games(
        self,
        team: str,
        season: int,
        week: int,
        n: int = 10,
    ) -> List[dict]:
        """Get team's most recent n games before the given week."""
        if team not in self._team_games:
            return []

        # Filter to games before this week
        previous = [
            g for g in self._team_games[team]
            if g['season'] < season or (g['season'] == season and g['week'] < week)
        ]

        # Sort by recency
        previous.sort(key=lambda x: (x['season'], x['week']), reverse=True)

        return previous[:n]

    def _calculate_streak(self, games: List[dict], field: str) -> int:
        """
        Calculate current streak.

        Returns positive for win streak, negative for loss streak.
        """
        if not games:
            return 0

        streak = 0
        first_result = games[0].get(field)

        if first_result is None:
            return 0

        for game in games:
            result = game.get(field)
            if result is None:
                break
            if result == first_result:
                streak += 1 if first_result else -1
            else:
                break

        return streak

    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend using linear regression slope.

        Positive slope = trending up, Negative = trending down.
        """
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Simple linear regression
        n = len(values)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)

        return float(slope)

    def get_momentum(self, team: str, season: int, week: int) -> TeamMomentum:
        """
        Get momentum metrics for a team at a specific point in time.

        Args:
            team: Team name
            season: Season year
            week: Week number

        Returns:
            TeamMomentum object with current momentum metrics
        """
        # Recalculate to ensure accuracy at this point in time
        self._update_momentum(team, season, week)
        return self._team_momentum.get(
            team,
            TeamMomentum(team=team, season=season, week=week)
        )

    def get_momentum_differential(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
    ) -> Dict[str, float]:
        """
        Get momentum differential between two teams.

        Args:
            home_team: Home team name
            away_team: Away team name
            season: Season year
            week: Week number

        Returns:
            Dictionary with momentum differential features
        """
        home_momentum = self.get_momentum(home_team, season, week)
        away_momentum = self.get_momentum(away_team, season, week)

        return {
            'momentum_diff': home_momentum.momentum_score - away_momentum.momentum_score,
            'su_streak_diff': home_momentum.straight_up_streak - away_momentum.straight_up_streak,
            'ats_streak_diff': home_momentum.ats_streak - away_momentum.ats_streak,
            'scoring_trend_diff': home_momentum.scoring_trend - away_momentum.scoring_trend,
            'home_is_hot': 1 if home_momentum.is_hot else 0,
            'home_is_cold': 1 if home_momentum.is_cold else 0,
            'away_is_hot': 1 if away_momentum.is_hot else 0,
            'away_is_cold': 1 if away_momentum.is_cold else 0,
            'home_ats_hot': 1 if home_momentum.is_ats_hot else 0,
            'home_ats_cold': 1 if home_momentum.is_ats_cold else 0,
            'away_ats_hot': 1 if away_momentum.is_ats_hot else 0,
            'away_ats_cold': 1 if away_momentum.is_ats_cold else 0,
        }

    def build_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Build momentum data from a historical games DataFrame.

        Args:
            df: DataFrame with game data (must have season, week, teams, points, spread)
        """
        # Sort by time
        df = df.sort_values(['season', 'week'])

        for _, row in df.iterrows():
            # Add home team game
            self.add_game(
                team=row['home_team'],
                opponent=row['away_team'],
                season=row['season'],
                week=row['week'],
                points_scored=row['home_points'],
                points_allowed=row['away_points'],
                is_home=True,
                spread=row.get('spread_line'),
                is_conference=row.get('is_conference', False),
            )

            # Add away team game
            self.add_game(
                team=row['away_team'],
                opponent=row['home_team'],
                season=row['season'],
                week=row['week'],
                points_scored=row['away_points'],
                points_allowed=row['home_points'],
                is_home=False,
                spread=-row.get('spread_line') if row.get('spread_line') else None,
                is_conference=row.get('is_conference', False),
            )

        logger.info(f"Built momentum data for {len(self._team_games)} teams")

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum features to a games DataFrame.

        Args:
            df: DataFrame with games to add features to

        Returns:
            DataFrame with momentum features added
        """
        momentum_features = []

        for _, row in df.iterrows():
            features = self.get_momentum_differential(
                home_team=row['home_team'],
                away_team=row['away_team'],
                season=row['season'],
                week=row['week'],
            )
            momentum_features.append(features)

        momentum_df = pd.DataFrame(momentum_features)
        return pd.concat([df.reset_index(drop=True), momentum_df], axis=1)
