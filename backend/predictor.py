"""
Sharp Sports Predictor - Backend Prediction Module

This module imports ALL prediction logic from the shared prediction_core module.
This ensures the backend produces IDENTICAL predictions to the Streamlit app.

DO NOT ADD PREDICTION LOGIC HERE. Use prediction_core.py.
"""

import os
import logging
from pathlib import Path

import pandas as pd

# Import everything from the shared prediction core module
# prediction_core.py should be in the same directory as this file
from prediction_core import (
    # Model class
    V19DualTargetModel,

    # Configuration
    CFBD_API_KEY,
    CFBD_BASE_URL,
    SPREAD_ERROR_THRESHOLD,
    KELLY_FRACTION,
    V19_FEATURES,
    PREFERRED_BOOKS,

    # Feature calculation
    calculate_v19_features_for_game,

    # Prediction generation
    generate_v19_predictions,

    # Data fetching
    fetch_schedule,
    fetch_lines,
    build_lines_dict,
    get_api_headers,

    # Kelly sizing
    kelly_bet_size,

    # Game classification
    classify_game_type_for_prediction,
    get_confidence_tier,

    # Model loading
    load_v19_dual_model,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Look for files in current directory first (bundled in Docker), then fallback
APP_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))


# =============================================================================
# MODEL AND DATA LOADING (Cached)
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
            model_path = str(APP_DIR / 'cfb_v19')
            logger.info(f"Loading V19 model from bundled path: {model_path}")
        elif data_model_path.exists():
            model_path = str(DATA_DIR / 'cfb_v19')
            logger.info(f"Loading V19 model from data path: {model_path}")
        else:
            raise FileNotFoundError(
                f"V19 model not found at {app_model_path} or {data_model_path}"
            )

        try:
            _v19_model = load_v19_dual_model(model_path)
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
            raise FileNotFoundError(
                f"History CSV not found at {app_csv_path} or {data_csv_path}"
            )

        try:
            _history_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(_history_df)} games from {csv_path}")
        except Exception as e:
            logger.error(f"Failed to load history data: {e}")
            raise
    return _history_df


# =============================================================================
# GENERATE PREDICTIONS (Uses shared logic)
# =============================================================================
def generate_predictions(games, lines_dict, season, week, bankroll=1000):
    """
    Generate predictions for a list of games.

    This function uses the SAME generate_v19_predictions() function
    as the Streamlit app to ensure identical predictions.

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

    # Use the shared generate_v19_predictions function
    df = generate_v19_predictions(
        games=games,
        lines_dict=lines_dict,
        model=model,
        history_df=history_df,
        season=season,
        week=week,
        bankroll=bankroll
    )

    if df.empty:
        return []

    # Convert DataFrame to list of dicts with API-friendly field names
    predictions = []
    for _, row in df.iterrows():
        predictions.append({
            'home_team': row['Home'],
            'away_team': row['Away'],
            'game': row['Game'],
            'signal': row['Signal'],
            'team_to_bet': row['team_to_bet'],
            'opponent': row['opponent'],
            'spread_to_bet': row['spread_to_bet'],
            'vegas_spread': row['vegas_spread'],
            'predicted_margin': row.get('predicted_margin', 0),
            'predicted_edge': row['spread_error'],
            'cover_probability': row['cover_probability'],
            'bet_recommendation': row['bet_recommendation'],
            'confidence_tier': row['confidence_tier'],
            'bet_size': row['bet_size'],
            'kelly_fraction': row.get('kelly_fraction', 0.0),
            'line_movement': row['line_movement'],
            'game_quality_score': row.get('game_quality', 0),
            'start_date': row.get('start_date'),
            'completed': row.get('completed', False),
            'game_id': row.get('game_id'),
        })

    return predictions
