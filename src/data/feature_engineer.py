"""
Feature Engineering for Sharp Sports Predictor.

Consolidates all feature engineering logic:
- Rolling stats
- Rest days
- Situational factors
- Interaction features
- Line movement features
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.config import get_config
from ..utils.logging_config import get_logger
from .momentum import MomentumTracker

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for CFB prediction.

    Combines all feature types:
    - Base features (Elo, EPA/PPA)
    - Rolling stats (L5 averages)
    - Rest/travel factors
    - Situational flags
    - Interaction features
    - Momentum features
    - Line movement features
    """

    def __init__(self):
        """Initialize the feature engineer."""
        self.config = get_config()
        self.momentum_tracker = MomentumTracker()
        self._team_games: Dict[str, List[dict]] = {}

    def process_dataframe(
        self,
        df: pd.DataFrame,
        include_momentum: bool = True,
        include_line_movement: bool = True,
    ) -> pd.DataFrame:
        """
        Process a raw games DataFrame and add all features.

        Args:
            df: Raw games DataFrame with basic columns
            include_momentum: Whether to add momentum features
            include_line_movement: Whether to add line movement features

        Returns:
            DataFrame with all features added
        """
        logger.info(f"Processing {len(df)} games for feature engineering")

        # Sort by time for proper rolling calculations
        df = df.sort_values(['season', 'week', 'id']).reset_index(drop=True)

        # Build team game history
        self._build_team_history(df)

        # Add rolling stats
        df = self._add_rolling_stats(df)

        # Add rest days
        df = self._add_rest_days(df)

        # Add situational factors
        df = self._add_situational_factors(df)

        # Add derived features
        df = self._add_derived_features(df)

        # Add interaction features
        df = self._add_interaction_features(df)

        # Add momentum features
        if include_momentum:
            self.momentum_tracker.build_from_dataframe(df)
            df = self.momentum_tracker.add_momentum_features(df)

        # Add line movement features
        if include_line_movement and 'spread_line' in df.columns:
            df = self._add_line_movement_features(df)

        logger.info(f"Feature engineering complete. Total columns: {len(df.columns)}")
        return df

    def _build_team_history(self, df: pd.DataFrame) -> None:
        """Build team game history for rolling calculations."""
        self._team_games = {}

        for _, row in df.iterrows():
            season = row['season']
            week = row['week']

            # Home team
            home_team = row['home_team']
            if home_team not in self._team_games:
                self._team_games[home_team] = []
            self._team_games[home_team].append({
                'season': season,
                'week': week,
                'points_scored': row['home_points'],
                'points_allowed': row['away_points'],
                'opponent': row['away_team'],
                'opponent_elo': row.get('away_pregame_elo', 1500),
                'is_home': True,
            })

            # Away team
            away_team = row['away_team']
            if away_team not in self._team_games:
                self._team_games[away_team] = []
            self._team_games[away_team].append({
                'season': season,
                'week': week,
                'points_scored': row['away_points'],
                'points_allowed': row['home_points'],
                'opponent': row['home_team'],
                'opponent_elo': row.get('home_pregame_elo', 1500),
                'is_home': False,
            })

    def _get_last_n_games(
        self,
        team: str,
        season: int,
        week: int,
        n: int = 5,
    ) -> List[dict]:
        """Get team's last n games before given season/week."""
        if team not in self._team_games:
            return []

        previous = [
            g for g in self._team_games[team]
            if g['season'] < season or (g['season'] == season and g['week'] < week)
        ]

        previous.sort(key=lambda x: (x['season'], x['week']), reverse=True)
        return previous[:n]

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling average stats for offense and defense."""
        n = self.config.features.rolling_window

        home_scores = []
        home_defense = []
        away_scores = []
        away_defense = []

        for _, row in df.iterrows():
            # Home team stats
            home_games = self._get_last_n_games(
                row['home_team'], row['season'], row['week'], n
            )
            if home_games:
                home_scores.append(np.mean([g['points_scored'] for g in home_games]))
                home_defense.append(np.mean([g['points_allowed'] for g in home_games]))
            else:
                home_scores.append(np.nan)
                home_defense.append(np.nan)

            # Away team stats
            away_games = self._get_last_n_games(
                row['away_team'], row['season'], row['week'], n
            )
            if away_games:
                away_scores.append(np.mean([g['points_scored'] for g in away_games]))
                away_defense.append(np.mean([g['points_allowed'] for g in away_games]))
            else:
                away_scores.append(np.nan)
                away_defense.append(np.nan)

        df['home_last5_score_avg'] = home_scores
        df['home_last5_defense_avg'] = home_defense
        df['away_last5_score_avg'] = away_scores
        df['away_last5_defense_avg'] = away_defense

        logger.debug(f"Added rolling stats. Valid: {df['home_last5_score_avg'].notna().sum()}")
        return df

    def _add_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rest days calculations."""
        home_rest = []
        away_rest = []

        for _, row in df.iterrows():
            home_rest.append(self._calculate_rest(
                row['home_team'], row['season'], row['week']
            ))
            away_rest.append(self._calculate_rest(
                row['away_team'], row['season'], row['week']
            ))

        df['home_rest_days'] = home_rest
        df['away_rest_days'] = away_rest
        df['home_rest'] = df['home_rest_days']
        df['away_rest'] = df['away_rest_days']
        df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']

        return df

    def _calculate_rest(self, team: str, season: int, week: int) -> int:
        """Calculate days since last game."""
        if team not in self._team_games:
            return 7  # Default

        previous = [
            g for g in self._team_games[team]
            if g['season'] < season or (g['season'] == season and g['week'] < week)
        ]

        if not previous:
            return 14  # Season opener

        previous.sort(key=lambda x: (x['season'], x['week']), reverse=True)
        last_game = previous[0]

        if last_game['season'] == season:
            week_diff = week - last_game['week']
        else:
            week_diff = 30  # Off-season

        return week_diff * 7

    def _add_situational_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational flags (West Coast travel, lookahead spots)."""
        west_coast_teams = set(self.config.features.west_coast_teams)
        lookahead_threshold = self.config.features.lookahead_elo_threshold

        west_coast_early = []
        home_lookahead = []
        away_lookahead = []

        for _, row in df.iterrows():
            # West Coast early game disadvantage
            away_is_west = row['away_team'] in west_coast_teams
            home_is_east = row['home_team'] not in west_coast_teams
            west_coast_early.append(1 if (away_is_west and home_is_east) else 0)

            # Lookahead spots
            home_lookahead.append(self._is_lookahead(
                row['home_team'], row['season'], row['week'], lookahead_threshold
            ))
            away_lookahead.append(self._is_lookahead(
                row['away_team'], row['season'], row['week'], lookahead_threshold
            ))

        df['west_coast_early'] = west_coast_early
        df['home_lookahead'] = home_lookahead
        df['away_lookahead'] = away_lookahead

        return df

    def _is_lookahead(
        self,
        team: str,
        season: int,
        week: int,
        threshold: float,
    ) -> int:
        """Check if team has a big game coming up (potential lookahead spot)."""
        if team not in self._team_games:
            return 0

        future = [
            g for g in self._team_games[team]
            if g['season'] == season and g['week'] > week
        ]

        if not future:
            return 0

        future.sort(key=lambda x: x['week'])
        next_game = future[0]

        return 1 if next_game.get('opponent_elo', 1500) > threshold else 0

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features (net EPA, HFA diff, etc.)."""
        # Net EPA (if PPA columns exist)
        if 'home_comp_off_ppa' in df.columns and 'away_comp_def_ppa' in df.columns:
            df['net_epa'] = df['home_comp_off_ppa'] - df['away_comp_def_ppa']
        else:
            df['net_epa'] = 0.0

        # HFA diff (if columns exist)
        if 'home_team_hfa' not in df.columns:
            df['home_team_hfa'] = self.config.features.default_hfa
        if 'away_team_hfa' not in df.columns:
            df['away_team_hfa'] = 0.0

        df['hfa_diff'] = df['home_team_hfa'] - df['away_team_hfa']

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features for non-linear relationships."""
        # Rest differential
        df['rest_diff'] = df['home_rest_days'] - df['away_rest_days']

        # Elo differential
        if 'home_pregame_elo' in df.columns and 'away_pregame_elo' in df.columns:
            df['elo_diff'] = df['home_pregame_elo'] - df['away_pregame_elo']
        else:
            df['elo_diff'] = 0.0

        # Pass efficiency differential (if exists)
        if 'home_comp_pass_ppa' in df.columns and 'away_comp_pass_ppa' in df.columns:
            df['pass_efficiency_diff'] = df['home_comp_pass_ppa'] - df['away_comp_pass_ppa']
        else:
            df['pass_efficiency_diff'] = 0.0

        # EPA x Elo interaction
        if 'home_comp_epa' in df.columns and 'away_comp_epa' in df.columns:
            epa_diff = df['home_comp_epa'] - df['away_comp_epa']
            elo_normalized = df['elo_diff'] / 100
            df['epa_elo_interaction'] = epa_diff * elo_normalized
        else:
            df['epa_elo_interaction'] = 0.0

        # Success rate differential
        if 'home_off_pass_success' in df.columns and 'away_off_pass_success' in df.columns:
            df['success_diff'] = df['home_off_pass_success'] - df['away_off_pass_success']
        else:
            df['success_diff'] = 0.0

        return df

    def _add_line_movement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add line movement features for betting signals."""
        if 'spread_open' not in df.columns:
            df['spread_open'] = df['spread_line']

        # Line movement
        df['line_movement'] = df['spread_line'] - df['spread_open']

        # Significant line move (threshold from config)
        threshold = self.config.betting.significant_line_move
        df['significant_line_move'] = (df['line_movement'].abs() > threshold).astype(int)

        # Direction of move (1 = moved toward home, -1 = moved toward away)
        df['line_move_direction'] = np.sign(df['line_movement'])

        # Reverse line movement (line moved against public)
        # This is a simplified heuristic - could be enhanced with actual betting % data
        df['reverse_line_move'] = (
            (df['line_movement'] > 0) & (df['spread_open'] < 0)  # Home was favorite, line moved toward them
        ).astype(int)

        return df

    def extract_features_for_prediction(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        home_stats: Optional[dict] = None,
        away_stats: Optional[dict] = None,
        vegas_line: Optional[float] = None,
        opening_line: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extract feature array for a single game prediction.

        Args:
            home_team: Home team name
            away_team: Away team name
            season: Season year
            week: Week number
            home_stats: Optional pre-computed home team stats
            away_stats: Optional pre-computed away team stats
            vegas_line: Current Vegas spread
            opening_line: Opening spread

        Returns:
            Feature array ready for model prediction
        """
        defaults = self.config.features

        # Get or compute team stats
        if home_stats is None:
            home_stats = self._get_default_stats()
        if away_stats is None:
            away_stats = self._get_default_stats()

        # Calculate derived features
        net_epa = home_stats.get('comp_off_ppa', 0) - away_stats.get('comp_def_ppa', 0)
        rest_advantage = home_stats.get('rest_days', 7) - away_stats.get('rest_days', 7)

        # Build feature array (16 features for V6)
        features = np.array([[
            home_stats.get('pregame_elo', defaults.default_elo),
            away_stats.get('pregame_elo', defaults.default_elo),
            home_stats.get('last5_score_avg', defaults.default_score_avg),
            away_stats.get('last5_score_avg', defaults.default_score_avg),
            home_stats.get('last5_defense_avg', defaults.default_defense_avg),
            away_stats.get('last5_defense_avg', defaults.default_defense_avg),
            home_stats.get('comp_off_ppa', defaults.default_ppa),
            away_stats.get('comp_off_ppa', defaults.default_ppa),
            home_stats.get('comp_def_ppa', defaults.default_ppa),
            away_stats.get('comp_def_ppa', defaults.default_ppa),
            net_epa,
            home_stats.get('hfa', defaults.default_hfa),
            away_stats.get('hfa', 0.0),
            home_stats.get('rest_days', defaults.default_rest_days),
            away_stats.get('rest_days', defaults.default_rest_days),
            rest_advantage,
        ]])

        return features

    def _get_default_stats(self) -> dict:
        """Get default stats for a team with no history."""
        defaults = self.config.features
        return {
            'pregame_elo': defaults.default_elo,
            'last5_score_avg': defaults.default_score_avg,
            'last5_defense_avg': defaults.default_defense_avg,
            'comp_off_ppa': defaults.default_ppa,
            'comp_def_ppa': defaults.default_ppa,
            'hfa': defaults.default_hfa,
            'rest_days': defaults.default_rest_days,
        }

    def get_feature_names(self, version: str = 'v6') -> List[str]:
        """Get feature names for a specific version."""
        if version == 'v6':
            return self.config.features.v6_features
        elif version == 'v7':
            return self.config.features.v7_features
        elif version == 'stacking':
            return self.config.features.stacking_features
        else:
            raise ValueError(f"Unknown feature version: {version}")
