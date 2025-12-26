"""
Prediction Core - Shared prediction logic for Streamlit and Backend.

This module contains ALL the core prediction functions extracted from app_v10.py.
Both the Streamlit app and FastAPI backend MUST import from this module to ensure
identical predictions across platforms.

DO NOT DUPLICATE THIS LOGIC. Import from here.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# API Configuration
CFBD_BASE_URL = "https://api.collegefootballdata.com"
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")

# Betting thresholds (V19)
SPREAD_ERROR_THRESHOLD = 4.5
KELLY_FRACTION = 0.25
VARIANCE_THRESHOLD = 7.0

# Game type filters
SKIP_PICK_EM_GAMES = True
SKIP_AVG_VS_AVG_GAMES = True
SKIP_EARLY_SEASON_GAMES = True
SHOW_PASS_RECOMMENDATIONS = True

# Preferred sportsbooks for odds (in order of preference)
PREFERRED_BOOKS = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'Bovada']

# V19/V20 Feature set (58 features - includes weather)
# V21 adds 4 QB availability features = 62 total
V19_FEATURES = [
    'home_pregame_elo', 'away_pregame_elo', 'elo_diff',
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',
    'home_team_hfa', 'hfa_diff',
    'rest_diff', 'home_rest_days', 'away_rest_days',
    'home_short_rest', 'away_short_rest',
    'vegas_spread', 'line_movement', 'spread_open',
    'large_favorite', 'large_underdog', 'close_game',
    'elo_vs_spread', 'rest_spread_interaction',
    'home_streak', 'away_streak', 'streak_diff',
    'home_ats', 'away_ats', 'ats_diff',
    'home_elo_momentum', 'away_elo_momentum', 'elo_momentum_diff',
    'home_scoring_trend', 'away_scoring_trend',
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',
    'home_comp_pass_ppa', 'away_comp_pass_ppa',
    'home_comp_rush_ppa', 'away_comp_rush_ppa',
    'pass_efficiency_diff',
    'matchup_efficiency',
    'home_pass_rush_balance', 'away_pass_rush_balance',
    'elo_efficiency_interaction', 'momentum_strength',
    'dominant_home', 'dominant_away',
    'rest_favorite_interaction', 'has_line_movement',
    'expected_total',
    # V20: Weather features (from CFBD Patreon API)
    'wind_speed', 'temperature', 'is_dome', 'high_wind',
    'cold_game', 'wind_pass_impact',
    # V21: QB availability features
    'home_qb_status', 'away_qb_status', 'qb_advantage', 'qb_uncertainty',
]

# Dome stadiums (weather doesn't affect these games)
DOME_STADIUMS = [
    'syracuse', 'georgia state', 'tulane', 'unlv',
    'new mexico', 'louisiana tech', 'northern illinois',
    # Bowl game dome venues
    'ford field', 'lucas oil stadium', 'at&t stadium',
    'mercedes-benz stadium', 'caesars superdome',
    'allegiant stadium', 'nrg stadium', 'u.s. bank stadium',
]

logger = logging.getLogger(__name__)


# =============================================================================
# API HELPERS
# =============================================================================

def get_api_headers() -> Dict[str, str]:
    """Get headers for CFBD API requests."""
    return {
        "Authorization": f"Bearer {CFBD_API_KEY}",
        "Accept": "application/json"
    }


# =============================================================================
# V19 DUAL-TARGET MODEL CLASS
# =============================================================================

class V19DualTargetModel:
    """
    Dual-target model combining margin prediction and cover classification.

    This is the EXACT same class definition from train_v19_dual.py.
    DO NOT modify the predict logic without also updating train_v19_dual.py.
    """

    def __init__(self):
        self.margin_model = None
        self.cover_model = None
        self.calibrator = None
        self.feature_names = None
        self.config = None

    def predict(self, X, vegas_spread=None) -> List[Dict[str, Any]]:
        """
        Make predictions with both models.

        Args:
            X: Features array/DataFrame
            vegas_spread: Optional vegas spread (used if X is numpy array)

        Returns list of dicts with:
        - predicted_margin: Predicted home team margin
        - predicted_edge: Edge over Vegas (margin - (-spread))
        - cover_probability: Calibrated probability of covering
        - confidence_tier: HIGH/MEDIUM-HIGH/MEDIUM/LOW/VERY LOW
        - bet_recommendation: BET/LEAN/PASS
        - pick_side: HOME/AWAY
        - game_quality_score: Quality score for the game type
        """
        # Handle numpy arrays vs DataFrames
        if isinstance(X, np.ndarray):
            X_filled = np.nan_to_num(X, nan=0.0)
            is_array = True
        else:
            X_filled = X.fillna(X.median())
            is_array = False

        # Get predictions
        predicted_margin = self.margin_model.predict(X_filled)

        # Get calibrated cover probability
        uncal_prob = self.cover_model.predict_proba(X_filled)[:, 1]
        if hasattr(self, 'calibrator') and self.calibrator is not None:
            cover_prob = self.calibrator.transform(uncal_prob)
        else:
            cover_prob = uncal_prob

        # Clamp probabilities to prevent unrealistic 100% or 0% confidence
        # The isotonic calibrator can clip to exactly 1.0 for out-of-bounds inputs
        cover_prob = np.clip(cover_prob, 0.05, 0.95)

        results = []
        for i in range(len(X_filled)):
            margin = predicted_margin[i]
            prob = cover_prob[i]

            # Get spread
            if vegas_spread is not None:
                spread = vegas_spread
            elif is_array:
                spread = 0
            else:
                spread = X.iloc[i].get('vegas_spread', 0)

            # Calculate edge
            edge = margin - (-spread)

            # Classify game type
            if is_array:
                game_type = {'bet_quality_score': 0}
            else:
                game_type = classify_game_type_for_prediction(X.iloc[i])

            # Determine confidence tier based on calibrated probability
            if prob >= 0.65 or prob <= 0.35:
                conf_tier = 'HIGH'
            elif prob >= 0.60 or prob <= 0.40:
                conf_tier = 'MEDIUM-HIGH'
            elif prob >= 0.55 or prob <= 0.45:
                conf_tier = 'MEDIUM'
            elif prob >= 0.52 or prob <= 0.48:
                conf_tier = 'LOW'
            else:
                conf_tier = 'VERY LOW'

            # Bet recommendation
            bet_home = edge > 0
            edge_abs = abs(edge)
            prob_confidence = abs(prob - 0.5)

            # PASS if game type is bad
            if game_type['bet_quality_score'] <= -3:
                recommendation = 'PASS'
            # BET only if strong confidence AND good edge
            elif edge_abs >= 4.5 and prob_confidence >= 0.15:
                recommendation = 'BET'
            # LEAN if moderate confidence
            elif edge_abs >= 3.0 and prob_confidence >= 0.10:
                recommendation = 'LEAN'
            else:
                recommendation = 'PASS'

            results.append({
                'predicted_margin': margin,
                'predicted_edge': edge,
                'cover_probability': prob,
                'confidence_tier': conf_tier,
                'bet_recommendation': recommendation,
                'pick_side': 'HOME' if bet_home else 'AWAY',
                'game_quality_score': game_type['bet_quality_score'],
            })

        return results

    @classmethod
    def load(cls, path_prefix: str = 'cfb_v19') -> 'V19DualTargetModel':
        """Load model from file."""
        model = cls()

        with open(f'{path_prefix}_dual.pkl', 'rb') as f:
            data = pickle.load(f)
            model.margin_model = data['margin_model']
            model.cover_model = data['cover_model']
            model.calibrator = data.get('calibrator', None)
            model.feature_names = data['feature_names']

        with open(f'{path_prefix}_dual_config.pkl', 'rb') as f:
            model.config = pickle.load(f)

        return model


# =============================================================================
# GAME TYPE CLASSIFICATION
# =============================================================================

def classify_game_type_for_prediction(row, week: Optional[int] = None) -> Dict[str, Any]:
    """
    Classify game type to identify PASS recommendations.

    This is the EXACT same logic as app_v10.py classify_game_type.
    """
    elo_diff = abs(row.get('elo_diff', 0) if pd.notna(row.get('elo_diff')) else 0)
    spread = abs(row.get('vegas_spread', 0) if pd.notna(row.get('vegas_spread')) else 0)
    home_elo = row.get('home_pregame_elo', 1500) if pd.notna(row.get('home_pregame_elo')) else 1500
    away_elo = row.get('away_pregame_elo', 1500) if pd.notna(row.get('away_pregame_elo')) else 1500

    game_type = {
        'is_pick_em': spread < 3,
        'is_avg_vs_avg': elo_diff < 100,
        'is_early_season': week is not None and week <= 2,
        'is_large_mismatch': elo_diff > 300,
        'is_elite_matchup': home_elo > 1700 and away_elo > 1700,
    }

    # Calculate bet quality score
    bet_quality = 0
    if game_type['is_pick_em']:
        bet_quality -= 2
    if game_type['is_avg_vs_avg']:
        bet_quality -= 3
    if game_type['is_early_season']:
        bet_quality -= 1
    if game_type['is_large_mismatch']:
        bet_quality += 1
    if game_type['is_elite_matchup']:
        bet_quality += 2

    game_type['bet_quality_score'] = bet_quality

    # Determine if this game should be PASSED
    pass_reasons = []
    if SKIP_PICK_EM_GAMES and game_type['is_pick_em']:
        pass_reasons.append('pick-em')
    if SKIP_AVG_VS_AVG_GAMES and game_type['is_avg_vs_avg']:
        pass_reasons.append('avg-vs-avg')
    if SKIP_EARLY_SEASON_GAMES and game_type['is_early_season']:
        pass_reasons.append('early-season')

    game_type['should_pass'] = len(pass_reasons) > 0
    game_type['pass_reasons'] = pass_reasons

    return game_type


def get_confidence_tier(spread_error: float) -> Tuple[str, str, str]:
    """Get confidence tier based on spread error magnitude."""
    error_mag = abs(spread_error)
    if error_mag >= 6.0:
        return 'HIGH', 'confidence-high', '🔥'
    elif error_mag >= 4.5:
        return 'MEDIUM-HIGH', 'confidence-medium-high', '✅'
    elif error_mag >= 3.0:
        return 'MEDIUM', 'confidence-medium', '⚠️'
    elif error_mag >= 1.5:
        return 'LOW', 'confidence-low', '⚡'
    else:
        return 'VERY LOW', 'confidence-very-low', '❄️'


# =============================================================================
# KELLY BET SIZING
# =============================================================================

def kelly_bet_size(spread_error: float, bankroll: float = 1000, odds: int = -110,
                   interval_width: Optional[float] = None,
                   interval_crosses_zero: bool = True) -> Dict[str, Any]:
    """
    Calculate bet size based on predicted spread error with interval adjustment.

    This is the EXACT same logic as app_v10.py kelly_bet_size.
    """
    edge = abs(spread_error) / 100

    # Convert American odds to decimal
    decimal_odds = 1 + (100 / abs(odds)) if odds < 0 else 1 + (odds / 100)
    b = decimal_odds - 1

    # Base probability from spread error
    base_prob = 0.5 + (abs(spread_error) / 50)

    # Dynamic adjustment based on quantile intervals
    if interval_width is not None:
        if not interval_crosses_zero:
            certainty_boost = max(0, 0.05 - (interval_width / 300))
            base_prob += certainty_boost
        if interval_width > 20:
            uncertainty_penalty = min(0.05, (interval_width - 20) / 200)
            base_prob -= uncertainty_penalty

    win_prob = min(0.75, max(0.52, base_prob))

    q = 1 - win_prob
    kelly_fraction = max(0, (b * win_prob - q) / b)

    # Apply fractional Kelly and cap
    bet_size = min(bankroll * kelly_fraction * KELLY_FRACTION, bankroll * 0.05)

    return {
        'bet_size': round(bet_size, 2),
        'kelly_fraction': kelly_fraction,
        'win_prob': win_prob,
        'edge': edge,
        'interval_adjusted': interval_width is not None
    }


# =============================================================================
# V19 FEATURE CALCULATION
# =============================================================================

def calculate_v19_features_for_game(
    home: str,
    away: str,
    history_df: pd.DataFrame,
    season: int,
    week: int,
    vegas_spread: float,
    spread_open: Optional[float] = None,
    line_movement: float = 0,
    wind_speed: float = 0,
    temperature: float = 65,
    venue: Optional[str] = None,
    qb_features: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Calculate features for V19/V20/V21 model (62 features including weather + QB).

    V21 adds QB availability features for injury-aware predictions.
    """

    def get_team_recent_stats(team: str, is_home: bool) -> Dict[str, Any]:
        """Get most recent stats for a team from historical data."""
        if is_home:
            games = history_df[(history_df['home_team'] == team) &
                              ((history_df['season'] < season) |
                               ((history_df['season'] == season) & (history_df['week'] < week)))]
            prefix = 'home'
        else:
            games = history_df[(history_df['away_team'] == team) &
                              ((history_df['season'] < season) |
                               ((history_df['season'] == season) & (history_df['week'] < week)))]
            prefix = 'away'

        if len(games) > 0:
            recent = games.sort_values(['season', 'week'], ascending=False).iloc[0]
            return {
                'pregame_elo': recent.get(f'{prefix}_pregame_elo', 1500),
                'last5_score_avg': recent.get(f'{prefix}_last5_score_avg', 28),
                'last5_defense_avg': recent.get(f'{prefix}_last5_defense_avg', 24),
                'team_hfa': recent.get(f'{prefix}_team_hfa', 2.0 if is_home else 0),
                'rest_days': recent.get(f'{prefix}_rest_days', 7),
                'streak': recent.get(f'{prefix}_streak', 0),
                'ats': recent.get(f'{prefix}_ats', 0.5),
                'elo_momentum': recent.get(f'{prefix}_elo_momentum', 0),
                'scoring_trend': recent.get(f'{prefix}_scoring_trend', 0),
                'short_rest': recent.get(f'{prefix}_short_rest', 0),
                'comp_off_ppa': recent.get(f'{prefix}_comp_off_ppa', 0),
                'comp_def_ppa': recent.get(f'{prefix}_comp_def_ppa', 0),
                'comp_pass_ppa': recent.get(f'{prefix}_comp_pass_ppa', 0),
                'comp_rush_ppa': recent.get(f'{prefix}_comp_rush_ppa', 0),
            }
        else:
            # Check alternate role
            alt_prefix = 'away' if is_home else 'home'
            alt_games = history_df[(history_df[f'{alt_prefix}_team'] == team) &
                                   ((history_df['season'] < season) |
                                    ((history_df['season'] == season) & (history_df['week'] < week)))]
            if len(alt_games) > 0:
                recent = alt_games.sort_values(['season', 'week'], ascending=False).iloc[0]
                return {
                    'pregame_elo': recent.get(f'{alt_prefix}_pregame_elo', 1500),
                    'last5_score_avg': recent.get(f'{alt_prefix}_last5_score_avg', 28),
                    'last5_defense_avg': recent.get(f'{alt_prefix}_last5_defense_avg', 24),
                    'team_hfa': 2.0 if is_home else 0,
                    'rest_days': recent.get(f'{alt_prefix}_rest_days', 7),
                    'streak': recent.get(f'{alt_prefix}_streak', 0),
                    'ats': recent.get(f'{alt_prefix}_ats', 0.5),
                    'elo_momentum': recent.get(f'{alt_prefix}_elo_momentum', 0),
                    'scoring_trend': recent.get(f'{alt_prefix}_scoring_trend', 0),
                    'short_rest': recent.get(f'{alt_prefix}_short_rest', 0),
                    'comp_off_ppa': recent.get(f'{alt_prefix}_comp_off_ppa', 0),
                    'comp_def_ppa': recent.get(f'{alt_prefix}_comp_def_ppa', 0),
                    'comp_pass_ppa': recent.get(f'{alt_prefix}_comp_pass_ppa', 0),
                    'comp_rush_ppa': recent.get(f'{alt_prefix}_comp_rush_ppa', 0),
                }

        # Default values
        return {
            'pregame_elo': 1500,
            'last5_score_avg': 28,
            'last5_defense_avg': 24,
            'team_hfa': 2.0 if is_home else 0,
            'rest_days': 7,
            'streak': 0,
            'ats': 0.5,
            'elo_momentum': 0,
            'scoring_trend': 0,
            'short_rest': 0,
            'comp_off_ppa': 0,
            'comp_def_ppa': 0,
            'comp_pass_ppa': 0,
            'comp_rush_ppa': 0,
        }

    home_stats = get_team_recent_stats(home, is_home=True)
    away_stats = get_team_recent_stats(away, is_home=False)

    # Derived features
    elo_diff = home_stats['pregame_elo'] - away_stats['pregame_elo']
    hfa_diff = home_stats['team_hfa'] - away_stats['team_hfa']
    rest_diff = home_stats['rest_days'] - away_stats['rest_days']
    streak_diff = home_stats['streak'] - away_stats['streak']
    ats_diff = home_stats['ats'] - away_stats['ats']
    elo_momentum_diff = home_stats['elo_momentum'] - away_stats['elo_momentum']

    # Vegas features
    if spread_open is None:
        spread_open = vegas_spread
    large_favorite = 1 if vegas_spread < -14 else 0
    large_underdog = 1 if vegas_spread > 14 else 0
    close_game = 1 if abs(vegas_spread) < 7 else 0
    elo_vs_spread = (elo_diff / 25) - vegas_spread
    rest_spread_interaction = rest_diff * abs(vegas_spread) / 10

    # PPA-derived
    pass_efficiency_diff = home_stats['comp_pass_ppa'] - away_stats['comp_pass_ppa']
    matchup_efficiency = (home_stats['comp_off_ppa'] - away_stats['comp_off_ppa']) + \
                         (away_stats['comp_def_ppa'] - home_stats['comp_def_ppa'])
    home_pass_rush_balance = home_stats['comp_pass_ppa'] - home_stats['comp_rush_ppa']
    away_pass_rush_balance = away_stats['comp_pass_ppa'] - away_stats['comp_rush_ppa']
    elo_efficiency_interaction = elo_diff * (home_stats['comp_off_ppa'] - away_stats['comp_off_ppa'])
    momentum_strength = (home_stats['streak'] - away_stats['streak']) * \
                        (home_stats['last5_score_avg'] - away_stats['last5_score_avg']) / 100
    dominant_home = 1 if (elo_diff > 150 and (home_stats['comp_off_ppa'] - away_stats['comp_off_ppa']) > 0.5) else 0
    dominant_away = 1 if (elo_diff < -150 and (away_stats['comp_off_ppa'] - home_stats['comp_off_ppa']) > 0.5) else 0
    rest_favorite_interaction = rest_diff * (1 if vegas_spread < -7 else 0)
    has_line_movement = 1 if (line_movement and abs(line_movement) > 0) else 0
    expected_total = home_stats['last5_score_avg'] + away_stats['last5_score_avg']

    # Weather features (V20)
    # Check if game is in a dome stadium
    home_lower = home.lower()
    venue_lower = (venue or '').lower()
    is_dome = 1 if (home_lower in DOME_STADIUMS or
                    any(dome in venue_lower for dome in DOME_STADIUMS)) else 0

    # Weather impact features (nullified for dome games)
    if is_dome:
        high_wind = 0
        cold_game = 0
        wind_pass_impact = 0.0
    else:
        high_wind = 1 if wind_speed >= 15 else 0
        cold_game = 1 if temperature <= 40 else 0
        # Wind impact on passing game
        pass_advantage = abs(home_stats['comp_pass_ppa'] - away_stats['comp_pass_ppa'])
        wind_pass_impact = wind_speed * pass_advantage / 10

    # Build V19/V20 feature array (58 features)
    features = np.array([[
        # Core power ratings (3)
        home_stats['pregame_elo'],
        away_stats['pregame_elo'],
        elo_diff,

        # Rolling performance (4)
        home_stats['last5_score_avg'],
        away_stats['last5_score_avg'],
        home_stats['last5_defense_avg'],
        away_stats['last5_defense_avg'],

        # Home field advantage (2)
        home_stats['team_hfa'],
        hfa_diff,

        # Scheduling factors (5)
        rest_diff,
        home_stats['rest_days'],
        away_stats['rest_days'],
        home_stats['short_rest'],
        away_stats['short_rest'],

        # Vegas features (8)
        vegas_spread,
        line_movement if line_movement else 0,
        spread_open,
        large_favorite,
        large_underdog,
        close_game,
        elo_vs_spread,
        rest_spread_interaction,

        # Momentum features (11)
        home_stats['streak'],
        away_stats['streak'],
        streak_diff,
        home_stats['ats'],
        away_stats['ats'],
        ats_diff,
        home_stats['elo_momentum'],
        away_stats['elo_momentum'],
        elo_momentum_diff,
        home_stats['scoring_trend'],
        away_stats['scoring_trend'],

        # PPA efficiency (9)
        home_stats['comp_off_ppa'],
        away_stats['comp_off_ppa'],
        home_stats['comp_def_ppa'],
        away_stats['comp_def_ppa'],
        home_stats['comp_pass_ppa'],
        away_stats['comp_pass_ppa'],
        home_stats['comp_rush_ppa'],
        away_stats['comp_rush_ppa'],
        pass_efficiency_diff,

        # Composite features (9)
        matchup_efficiency,
        home_pass_rush_balance,
        away_pass_rush_balance,
        elo_efficiency_interaction,
        momentum_strength,
        dominant_home,
        dominant_away,
        rest_favorite_interaction,
        has_line_movement,

        # Expected total (1)
        expected_total,

        # Weather features (6) - V20
        wind_speed,
        temperature,
        is_dome,
        high_wind,
        cold_game,
        wind_pass_impact,

        # QB availability features - REMOVED for V19 compatibility
        # V21+ would add these features when using V21 model:
        # qb_features.get('home_qb_status', 0) if qb_features else 0,
        # qb_features.get('away_qb_status', 0) if qb_features else 0,
        # qb_features.get('qb_advantage', 0) if qb_features else 0,
        # qb_features.get('qb_uncertainty_diff', 0) if qb_features else 0,
    ]])

    return features


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_schedule(season: int, week: int, season_type: str = 'regular') -> List[Dict]:
    """Fetch game schedule from CFBD."""
    try:
        if season_type == 'postseason':
            url = f"{CFBD_BASE_URL}/games?year={season}&seasonType={season_type}"
        else:
            url = f"{CFBD_BASE_URL}/games?year={season}&week={week}&seasonType={season_type}"

        resp = requests.get(url, headers=get_api_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"CFBD schedule fetch failed: {resp.status_code}")
        return []
    except Exception as e:
        logger.error(f"Error fetching schedule: {e}")
        return []


def fetch_lines(season: int, week: int, season_type: str = 'regular') -> List[Dict]:
    """Fetch betting lines from CFBD."""
    try:
        if season_type == 'postseason':
            url = f"{CFBD_BASE_URL}/lines?year={season}&seasonType={season_type}"
        else:
            url = f"{CFBD_BASE_URL}/lines?year={season}&week={week}&seasonType={season_type}"

        resp = requests.get(url, headers=get_api_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"CFBD lines fetch failed: {resp.status_code}")
        return []
    except Exception as e:
        logger.error(f"Error fetching lines: {e}")
        return []


def fetch_weather(season: int, week: int, season_type: str = 'regular') -> Dict[int, Dict]:
    """
    Fetch weather data from CFBD API (requires Patreon Pro tier).

    Returns dict keyed by game_id with wind_speed and temperature.
    """
    weather_dict = {}

    try:
        if season_type == 'postseason':
            url = f"{CFBD_BASE_URL}/games/weather?year={season}&seasonType={season_type}"
        else:
            url = f"{CFBD_BASE_URL}/games/weather?year={season}&week={week}"

        resp = requests.get(url, headers=get_api_headers(), timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            for game in data:
                game_id = game.get('id')
                if game_id:
                    weather_dict[game_id] = {
                        'wind_speed': game.get('windSpeed', 0) or 0,
                        'temperature': game.get('temperature', 65) or 65,
                        'humidity': game.get('humidity'),
                        'weather_condition': game.get('weatherCondition'),
                    }
            logger.info(f"Fetched weather for {len(weather_dict)} games")
        elif resp.status_code == 403:
            logger.warning("Weather API requires CFBD Patreon Pro tier - using defaults")
        else:
            logger.warning(f"CFBD weather fetch returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Error fetching weather: {e}")

    return weather_dict


def build_lines_dict(lines_data: List[Dict]) -> Dict[str, Dict]:
    """
    Build a dictionary of lines keyed by home team.

    This is the EXACT same logic as app_v10.py.
    """
    lines_dict = {}

    for game in lines_data:
        home = game.get('homeTeam')
        if not home:
            continue

        game_lines = game.get('lines', [])
        if not game_lines:
            continue

        # Find best line from preferred books
        best_line = None
        for book in PREFERRED_BOOKS:
            for line in game_lines:
                if line.get('provider', '').lower() == book.lower():
                    best_line = line
                    break
            if best_line:
                break

        # Fall back to first available line
        if not best_line:
            best_line = game_lines[0]

        spread = best_line.get('spread')
        if spread is None:
            continue

        try:
            spread_current = float(spread)
        except (ValueError, TypeError):
            continue

        spread_opening = best_line.get('spreadOpen', spread_current)
        try:
            spread_opening = float(spread_opening)
        except (ValueError, TypeError):
            spread_opening = spread_current

        line_movement = spread_current - spread_opening

        lines_dict[home] = {
            'spread_current': spread_current,
            'spread_opening': spread_opening,
            'line_movement': line_movement,
            'provider': best_line.get('provider', 'Unknown'),
            'over_under': best_line.get('overUnder'),
        }

    return lines_dict


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_v19_dual_model(model_path: str = 'cfb_v19') -> V19DualTargetModel:
    """
    Load the V19 dual-target model.

    Args:
        model_path: Path prefix for model files (without _dual.pkl suffix)

    Returns:
        V19DualTargetModel instance
    """
    return V19DualTargetModel.load(model_path)


# =============================================================================
# PREDICTION GENERATION
# =============================================================================

def generate_v19_predictions(
    games: List[Dict],
    lines_dict: Dict[str, Dict],
    model: V19DualTargetModel,
    history_df: pd.DataFrame,
    season: int,
    week: int,
    bankroll: float,
    weather_dict: Optional[Dict[int, Dict]] = None,
    injury_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Generate predictions using V19/V21 dual-target model.

    V21 adds injury_df parameter for QB availability tracking.

    Args:
        games: List of game dicts from CFBD API
        lines_dict: Dict of betting lines keyed by home team
        model: V19DualTargetModel instance
        history_df: Historical games DataFrame for feature calculation
        season: Current season year
        week: Current week number
        bankroll: Bankroll for bet sizing
        weather_dict: Optional dict of weather data keyed by game_id
        injury_df: Optional DataFrame of injury data for QB tracking

    Returns DataFrame with predictions for each game.
    """
    predictions = []

    # Initialize weather dict if not provided
    if weather_dict is None:
        weather_dict = {}

    # V21: Helper to get QB features for a game
    def get_qb_features_for_game(home: str, away: str) -> Optional[Dict[str, float]]:
        """Get QB availability features if injury data available."""
        if injury_df is None or injury_df.empty:
            return None

        try:
            # Import here to avoid circular imports
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from fetch_injuries import generate_qb_features
            return generate_qb_features(home, away, injury_df)
        except Exception as e:
            logger.warning(f"Could not generate QB features: {e}")
            return None

    for game in games:
        try:
            # Support both camelCase (CFBD) and snake_case (normalized)
            home = game.get('homeTeam') or game.get('home_team')
            away = game.get('awayTeam') or game.get('away_team')

            if not home or not away or home not in lines_dict:
                continue

            vegas_spread = lines_dict[home]['spread_current']
            line_movement = lines_dict[home]['line_movement']
            spread_open = lines_dict[home].get('spread_opening', vegas_spread)

            # Get venue for dome detection
            venue = game.get('venue') or game.get('venue_name') or ''

            # Get weather data for this game (from CFBD Pro tier if available)
            game_id = game.get('id')
            game_weather = weather_dict.get(game_id, {})
            wind_speed = game_weather.get('wind_speed', 0)
            temperature = game_weather.get('temperature', 65)

            # V21: Get QB availability features
            qb_features = get_qb_features_for_game(home, away)

            # Calculate V19/V20/V21 features (62 features including weather + QB)
            features = calculate_v19_features_for_game(
                home, away, history_df, season, week, vegas_spread,
                spread_open=spread_open, line_movement=line_movement,
                wind_speed=wind_speed, temperature=temperature, venue=venue,
                qb_features=qb_features
            )

            # Get V19 prediction
            result = model.predict(features, vegas_spread=vegas_spread)[0]

            # Extract V19 fields
            pred_spread_error = result['predicted_edge']
            cover_prob = result['cover_probability']
            predicted_margin = result['predicted_margin']
            confidence_tier = result['confidence_tier']
            bet_recommendation = result['bet_recommendation']
            pick_side = result['pick_side']
            game_quality = result.get('game_quality_score', 0.5)

            # Determine signal from pick_side
            if pick_side == 'HOME':
                signal = 'BUY'
                team_to_bet = home
                opponent = away
                spread_to_bet = vegas_spread
            else:
                signal = 'FADE'
                team_to_bet = away
                opponent = home
                spread_to_bet = -vegas_spread

            # Map confidence tier to CSS class and emoji
            tier_mapping = {
                'HIGH': ('confidence-high', '🔥'),
                'MEDIUM-HIGH': ('confidence-medium-high', '✅'),
                'MEDIUM': ('confidence-medium', '⚠️'),
                'LOW': ('confidence-low', '⚡'),
                'VERY LOW': ('confidence-very-low', '❄️'),
            }
            confidence_class, confidence_emoji = tier_mapping.get(
                confidence_tier, ('confidence-medium', '⚠️')
            )

            # Calculate bet size using cover probability directly
            edge = cover_prob - 0.5
            if edge > 0:
                kelly_fraction = edge / 0.91  # Simplified Kelly
                kelly_fraction = min(kelly_fraction, 0.05)  # Cap at 5%
                bet_size = bankroll * kelly_fraction
            else:
                bet_size = 0

            # Adjust bet size based on recommendation
            if bet_recommendation == 'PASS':
                bet_size = 0
            elif bet_recommendation == 'LEAN':
                bet_size = bet_size * 0.5

            # Show probability for the PICKED side, not always home
            if pick_side == 'HOME':
                effective_prob = cover_prob
                effective_edge = pred_spread_error  # Edge already favors home
            else:
                effective_prob = 1 - cover_prob
                effective_edge = -pred_spread_error  # Flip edge for away pick

            predictions.append({
                'Home': home,
                'Away': away,
                'Game': f"{away} @ {home}",
                'Signal': signal,
                'team_to_bet': team_to_bet,
                'opponent': opponent,
                'spread_to_bet': spread_to_bet,
                'vegas_spread': vegas_spread,
                'spread_error': effective_edge,  # Now from picked team's perspective
                'predicted_margin': predicted_margin,
                'win_prob': effective_prob,
                'cover_probability': effective_prob,
                'bet_size': bet_size,
                'line_movement': line_movement,
                'confidence_tier': confidence_tier,
                'confidence_class': confidence_class,
                'confidence_emoji': confidence_emoji,
                'bet_recommendation': bet_recommendation,
                'game_quality': game_quality,
                'start_date': game.get('start_date') or game.get('startDate'),
                'completed': game.get('completed', False),
                'game_id': game.get('id'),
                # V19 doesn't use quantile intervals
                'lower_bound': None,
                'upper_bound': None,
                'interval_width': None,
                'interval_crosses_zero': None,
            })
        except Exception as e:
            logger.error(f"V19 error predicting {game.get('awayTeam', '?')} @ {game.get('homeTeam', '?')}: {e}")

    return pd.DataFrame(predictions)
