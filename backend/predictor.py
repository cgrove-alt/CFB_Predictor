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
MODEL_DIR = DATA_DIR / "models"


# =============================================================================
# MODEL AND DATA LOADING (Cached)
# =============================================================================
_v19_model = None
_v19_model_version = None
_history_df = None


def get_latest_model_path():
    """
    Find the latest model to load.

    Priority:
    1. Latest retrained model in /data/models/ (check current_model.txt)
    2. Bundled model in /app/ directory
    3. Model in /data/ directory

    Returns tuple of (model_path_prefix, version_string)
    """
    # Check for retrained model in volume
    current_model_file = MODEL_DIR / 'current_model.txt'
    if current_model_file.exists():
        try:
            version = current_model_file.read_text().strip()
            model_path = MODEL_DIR / f'{version}.pkl'
            if model_path.exists():
                logger.info(f"Found retrained model: {version}")
                return str(MODEL_DIR / version), version
        except Exception as e:
            logger.warning(f"Error reading current_model.txt: {e}")

    # Fall back to bundled model
    app_model_path = APP_DIR / 'cfb_v19_dual.pkl'
    if app_model_path.exists():
        return str(APP_DIR / 'cfb_v19'), 'bundled'

    # Fall back to data directory
    data_model_path = DATA_DIR / 'cfb_v19_dual.pkl'
    if data_model_path.exists():
        return str(DATA_DIR / 'cfb_v19'), 'data'

    raise FileNotFoundError("No V19 model found in any location")


def load_v19_model(force_reload=False):
    """Load V19 dual-target model."""
    global _v19_model, _v19_model_version

    # Check if we should reload (new retrained model available)
    model_path, version = get_latest_model_path()

    if _v19_model is None or force_reload or version != _v19_model_version:
        try:
            logger.info(f"Loading V19 model from: {model_path} (version: {version})")
            _v19_model = load_v19_dual_model(model_path)
            _v19_model_version = version
            logger.info(f"Loaded V19 dual-target model successfully (version: {version})")
        except Exception as e:
            logger.error(f"Failed to load V19 model: {e}")
            raise

    return _v19_model


def reload_model_if_needed():
    """
    Check if a new retrained model is available and reload if so.
    Called periodically to pick up new weekly models.
    """
    global _v19_model_version

    try:
        _, latest_version = get_latest_model_path()
        if latest_version != _v19_model_version:
            logger.info(f"New model available: {latest_version} (current: {_v19_model_version})")
            load_v19_model(force_reload=True)
            return True
    except Exception as e:
        logger.warning(f"Error checking for new model: {e}")

    return False


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
