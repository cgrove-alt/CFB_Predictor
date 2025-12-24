"""
Sharp Sports Predictor - Prediction Logic

Extracted from app_v10.py for Railway backend deployment.
Contains V19 dual-target model and feature calculation logic.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Look for files in current directory first (bundled in Docker), then fallback to DATA_DIR
APP_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
KELLY_FRACTION = 0.25
SPREAD_ERROR_THRESHOLD = 4.5

# V19 features (52 features)
V19_FEATURES = [
    # Core power ratings (3)
    'home_pregame_elo', 'away_pregame_elo', 'elo_diff',
    # Rolling performance (4)
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',
    # Home field advantage (2)
    'home_team_hfa', 'hfa_diff',
    # Scheduling factors (5)
    'rest_diff', 'home_rest_days', 'away_rest_days',
    'home_short_rest', 'away_short_rest',
    # Vegas features (8)
    'vegas_spread', 'line_movement', 'spread_open',
    'large_favorite', 'large_underdog', 'close_game',
    'elo_vs_spread', 'rest_spread_interaction',
    # Momentum features (11)
    'home_streak', 'away_streak', 'streak_diff',
    'home_ats', 'away_ats', 'ats_diff',
    'home_elo_momentum', 'away_elo_momentum', 'elo_momentum_diff',
    'home_scoring_trend', 'away_scoring_trend',
    # PPA efficiency (9)
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',
    'home_comp_pass_ppa', 'away_comp_pass_ppa',
    'home_comp_rush_ppa', 'away_comp_rush_ppa',
    'pass_efficiency_diff',
    # Composite features (9)
    'matchup_efficiency', 'home_pass_rush_balance', 'away_pass_rush_balance',
    'elo_efficiency_interaction', 'momentum_strength',
    'dominant_home', 'dominant_away',
    'rest_favorite_interaction', 'has_line_movement',
    # Expected total (1)
    'expected_total',
]


# =============================================================================
# V19 DUAL TARGET MODEL
# =============================================================================
class V19DualTargetModel:
    """
    Dual-target model combining margin prediction and cover classification.
    Loaded from cfb_v19_dual.pkl and cfb_v19_dual_config.pkl
    """

    def __init__(self):
        self.margin_model = None
        self.cover_model = None
        self.calibrator = None
        self.feature_names = None
        self.config = None

    def predict(self, X, vegas_spread=None):
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
        if self.calibrator is not None:
            cover_prob = self.calibrator.transform(uncal_prob)
        else:
            cover_prob = uncal_prob

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

            # Determine confidence tier
            if prob >= 0.65 or prob <= 0.35:
                conf_tier = 'HIGH'
            elif prob >= 0.60 or prob <= 0.40:
                conf_tier = 'MEDIUM-HIGH'
            elif prob >= 0.55 or prob <= 0.45:
                conf_tier = 'MEDIUM'
            else:
                conf_tier = 'LOW'

            # Determine pick side and effective probability
            if margin > -spread:  # Model predicts home covers
                pick_side = 'HOME'
                effective_prob = prob
            else:
                pick_side = 'AWAY'
                effective_prob = 1 - prob

            # Determine bet recommendation
            edge_abs = abs(edge)
            prob_confidence = abs(prob - 0.5)

            # Calculate game quality score
            bet_quality_score = 0
            if edge_abs >= 5:
                bet_quality_score += 2
            elif edge_abs >= 3:
                bet_quality_score += 1

            if prob_confidence >= 0.15:
                bet_quality_score += 2
            elif prob_confidence >= 0.10:
                bet_quality_score += 1

            # Determine recommendation
            if bet_quality_score <= -3:
                recommendation = 'PASS'
            elif edge_abs < 3.0:
                recommendation = 'PASS'
            elif prob_confidence < 0.10:
                recommendation = 'PASS'
            elif conf_tier in ['HIGH', 'MEDIUM-HIGH'] and edge_abs >= 4.5:
                recommendation = 'BET'
            elif conf_tier in ['HIGH', 'MEDIUM-HIGH'] or edge_abs >= 3.5:
                recommendation = 'LEAN'
            else:
                recommendation = 'PASS'

            results.append({
                'predicted_margin': float(margin),
                'predicted_edge': float(edge),
                'cover_probability': float(effective_prob),
                'confidence_tier': conf_tier,
                'bet_recommendation': recommendation,
                'pick_side': pick_side,
                'game_quality_score': bet_quality_score,
            })

        return results

    def save(self, path_prefix='cfb_v19'):
        """Save model and config."""
        with open(f'{path_prefix}_dual.pkl', 'wb') as f:
            pickle.dump({
                'margin_model': self.margin_model,
                'cover_model': self.cover_model,
                'calibrator': self.calibrator,
                'feature_names': self.feature_names,
            }, f)

        with open(f'{path_prefix}_dual_config.pkl', 'wb') as f:
            pickle.dump(self.config, f)

    @classmethod
    def load(cls, path_prefix='cfb_v19'):
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
# MODEL LOADING
# =============================================================================
_v19_model = None
_history_df = None


def load_v19_model():
    """Load V19 dual-target model."""
    global _v19_model
    if _v19_model is None:
        # Try bundled files first (in app directory), then DATA_DIR
        app_model_path = APP_DIR / 'cfb_v19_dual.pkl'
        data_model_path = DATA_DIR / 'cfb_v19_dual.pkl'

        if app_model_path.exists():
            # Pass full path prefix (without _dual.pkl suffix)
            model_path = str(APP_DIR / 'cfb_v19')
            logger.info(f"Loading V19 model from bundled path: {model_path}")
        elif data_model_path.exists():
            model_path = str(DATA_DIR / 'cfb_v19')
            logger.info(f"Loading V19 model from data path: {model_path}")
        else:
            raise FileNotFoundError(f"V19 model not found at {app_model_path} or {data_model_path}")

        try:
            _v19_model = V19DualTargetModel.load(model_path)
            logger.info("Loaded V19 dual-target model successfully")
        except Exception as e:
            logger.error(f"Failed to load V19 model: {e}")
            raise
    return _v19_model


def load_history_data():
    """Load historical data for feature calculation."""
    global _history_df
    if _history_df is None:
        # Try bundled files first (in app directory), then DATA_DIR
        app_csv_path = APP_DIR / 'cfb_data_safe.csv'
        data_csv_path = DATA_DIR / 'cfb_data_safe.csv'

        if app_csv_path.exists():
            csv_path = app_csv_path
            logger.info(f"Loading history from bundled path: {csv_path}")
        elif data_csv_path.exists():
            csv_path = data_csv_path
            logger.info(f"Loading history from data path: {csv_path}")
        else:
            raise FileNotFoundError(f"History CSV not found at {app_csv_path} or {data_csv_path}")

        try:
            _history_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(_history_df)} games from {csv_path}")
        except Exception as e:
            logger.error(f"Failed to load history data: {e}")
            raise
    return _history_df


# =============================================================================
# FEATURE CALCULATION
# =============================================================================
def calculate_v19_features_for_game(home, away, history_df, season, week,
                                     vegas_spread, spread_open=None, line_movement=0):
    """Calculate features for V19 model (52 features)."""

    def get_team_recent_stats(team, is_home):
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

    # Build V19 feature array (52 features)
    features = np.array([[
        home_stats['pregame_elo'],
        away_stats['pregame_elo'],
        elo_diff,
        home_stats['last5_score_avg'],
        away_stats['last5_score_avg'],
        home_stats['last5_defense_avg'],
        away_stats['last5_defense_avg'],
        home_stats['team_hfa'],
        hfa_diff,
        rest_diff,
        home_stats['rest_days'],
        away_stats['rest_days'],
        home_stats['short_rest'],
        away_stats['short_rest'],
        vegas_spread,
        line_movement if line_movement else 0,
        spread_open,
        large_favorite,
        large_underdog,
        close_game,
        elo_vs_spread,
        rest_spread_interaction,
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
        home_stats['comp_off_ppa'],
        away_stats['comp_off_ppa'],
        home_stats['comp_def_ppa'],
        away_stats['comp_def_ppa'],
        home_stats['comp_pass_ppa'],
        away_stats['comp_pass_ppa'],
        home_stats['comp_rush_ppa'],
        away_stats['comp_rush_ppa'],
        pass_efficiency_diff,
        matchup_efficiency,
        home_pass_rush_balance,
        away_pass_rush_balance,
        elo_efficiency_interaction,
        momentum_strength,
        dominant_home,
        dominant_away,
        rest_favorite_interaction,
        has_line_movement,
        expected_total,
    ]])

    return features


# =============================================================================
# KELLY CRITERION
# =============================================================================
def kelly_bet_size(cover_probability, bankroll=1000, odds=-110):
    """
    Calculate bet size based on cover probability.

    Args:
        cover_probability: Probability of covering (0-1)
        bankroll: Total bankroll
        odds: American odds (default -110)

    Returns:
        dict with bet sizing info
    """
    # Convert American odds to decimal
    decimal_odds = 1 + (100 / abs(odds)) if odds < 0 else 1 + (odds / 100)
    b = decimal_odds - 1

    win_prob = cover_probability
    q = 1 - win_prob

    # Kelly formula: (bp - q) / b
    kelly_fraction = max(0, (b * win_prob - q) / b)

    # Apply fractional Kelly and cap
    bet_size = min(bankroll * kelly_fraction * KELLY_FRACTION, bankroll * 0.05)

    return {
        'bet_size': round(bet_size, 2),
        'kelly_fraction': kelly_fraction,
        'win_prob': win_prob,
    }


# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================
def generate_predictions(games, lines_dict, season, week, bankroll=1000):
    """
    Generate predictions for a list of games.

    Args:
        games: List of game dicts from CFBD API
        lines_dict: Dict mapping home team to line info
        season: Season year
        week: Week number
        bankroll: Bankroll for Kelly sizing

    Returns:
        List of prediction dicts
    """
    model = load_v19_model()
    history_df = load_history_data()
    predictions = []

    for game in games:
        try:
            home = game.get('homeTeam') or game.get('home_team')
            away = game.get('awayTeam') or game.get('away_team')

            if not home or not away or home not in lines_dict:
                continue

            line_info = lines_dict[home]
            vegas_spread = line_info['spread_current']
            line_movement = line_info.get('line_movement', 0)
            spread_open = line_info.get('spread_opening', vegas_spread)

            # Calculate features
            features = calculate_v19_features_for_game(
                home, away, history_df, season, week, vegas_spread,
                spread_open=spread_open, line_movement=line_movement
            )

            # Get V19 prediction
            result = model.predict(features, vegas_spread=vegas_spread)[0]

            # Determine signal and team to bet
            if result['pick_side'] == 'HOME':
                signal = 'BUY'
                team_to_bet = home
                opponent = away
                spread_to_bet = vegas_spread
            else:
                signal = 'FADE'
                team_to_bet = away
                opponent = home
                spread_to_bet = -vegas_spread

            # Calculate bet size
            kelly_result = kelly_bet_size(result['cover_probability'], bankroll)

            # Adjust bet size based on recommendation
            bet_size = kelly_result['bet_size']
            if result['bet_recommendation'] == 'PASS':
                bet_size = 0
            elif result['bet_recommendation'] == 'LEAN':
                bet_size = bet_size * 0.5

            predictions.append({
                'home_team': home,
                'away_team': away,
                'game': f"{away} @ {home}",
                'signal': signal,
                'team_to_bet': team_to_bet,
                'opponent': opponent,
                'spread_to_bet': spread_to_bet,
                'vegas_spread': vegas_spread,
                'predicted_margin': result['predicted_margin'],
                'predicted_edge': result['predicted_edge'],
                'cover_probability': result['cover_probability'],
                'bet_recommendation': result['bet_recommendation'],
                'confidence_tier': result['confidence_tier'],
                'bet_size': bet_size,
                'kelly_fraction': kelly_result['kelly_fraction'],
                'line_movement': line_movement,
                'game_quality_score': result['game_quality_score'],
                'start_date': game.get('start_date'),
                'completed': game.get('completed', False),
                'game_id': game.get('id'),
            })

        except Exception as e:
            logger.error(f"Error predicting {game}: {e}")
            continue

    return predictions
