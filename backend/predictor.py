"""
Sharp Sports Predictor - Backend Prediction Module

This module imports ALL prediction logic from the shared prediction_core module.
This ensures the backend produces IDENTICAL predictions to the Streamlit app.

V22: Switched to Meta-Router Model with specialized sub-models for:
- Elite matchups (high Elo competitive games)
- Average vs Average games (improved from 55% to 62% cover accuracy)
- Mismatch games (FCS/blowout with 92% blowout classifier accuracy)
- Standard games

DO NOT ADD PREDICTION LOGIC HERE. Use prediction_core.py.
"""

import os
import logging
from pathlib import Path

import pandas as pd

# Import everything from the shared prediction core module
# prediction_core.py should be in the same directory as this file
from prediction_core import (
    # Model classes
    V19DualTargetModel,
    V22MetaRouterModel,  # V22: Meta-Router Model

    # Configuration
    CFBD_API_KEY,
    CFBD_BASE_URL,
    SPREAD_ERROR_THRESHOLD,
    KELLY_FRACTION,
    V19_FEATURES,
    PREFERRED_BOOKS,
    FCS_TEAMS,  # V22: FCS team detection

    # Feature calculation
    calculate_v19_features_for_game,

    # Prediction generation
    generate_v19_predictions,
    generate_v22_predictions,  # V22: Meta-Router predictions

    # Data fetching
    fetch_schedule,
    fetch_lines,
    fetch_weather,
    build_lines_dict,
    get_api_headers,

    # Kelly sizing
    kelly_bet_size,

    # Game classification
    classify_game_type_for_prediction,
    classify_game_cluster_v22,  # V22: Game type classification (renamed from classify_game_type_v22)
    get_confidence_tier,

    # Model loading
    load_v19_dual_model,
    load_v22_meta_model,  # V22: Load meta-router model
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Look for files in current directory first (bundled in Docker), then fallback
APP_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MODEL_DIR = DATA_DIR / "models"

# V22: Use Meta-Router Model by default
USE_V22_MODEL = True  # Set to False to revert to V19


# =============================================================================
# MODEL AND DATA LOADING (Cached)
# =============================================================================
_v22_model = None
_v22_model_version = None
_v19_model = None
_v19_model_version = None
_history_df = None


def get_latest_v22_model_path():
    """
    Find the latest V22 model to load.

    Priority:
    1. Bundled model in /app/ directory
    2. Model in /data/ directory

    Returns tuple of (model_path_prefix, version_string)
    """
    # Check for bundled V22 model
    app_model_path = APP_DIR / 'cfb_v22_meta.pkl'
    if app_model_path.exists():
        return str(APP_DIR / 'cfb'), 'v22-bundled'

    # Fall back to data directory
    data_model_path = DATA_DIR / 'cfb_v22_meta.pkl'
    if data_model_path.exists():
        return str(DATA_DIR / 'cfb'), 'v22-data'

    raise FileNotFoundError("No V22 model found in any location")


def get_latest_model_path():
    """
    Find the latest V19 model to load.

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


def load_v22_model(force_reload=False):
    """Load V22 meta-router model."""
    global _v22_model, _v22_model_version

    model_path, version = get_latest_v22_model_path()

    if _v22_model is None or force_reload or version != _v22_model_version:
        try:
            logger.info(f"Loading V22 model from: {model_path} (version: {version})")
            _v22_model = load_v22_meta_model(model_path)
            _v22_model_version = version
            logger.info(f"Loaded V22 Meta-Router model successfully (version: {version})")
            if _v22_model.metrics:
                logger.info(f"V22 metrics: {_v22_model.metrics.get('overall', {})}")
        except Exception as e:
            logger.error(f"Failed to load V22 model: {e}")
            raise

    return _v22_model


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
# GENERATE PREDICTIONS (Uses V22 Meta-Router by default)
# =============================================================================
def generate_predictions(games, lines_dict, season, week, bankroll=1000, season_type='regular'):
    """
    Generate predictions for a list of games.

    V22: Uses Meta-Router Model with specialized sub-models for different game types.

    Args:
        games: List of game dicts from CFBD API
        lines_dict: Dict mapping home team to line info
        season: Season year
        week: Week number
        bankroll: Bankroll for Kelly sizing
        season_type: 'regular' or 'postseason'

    Returns:
        List of prediction dicts
    """
    history_df = load_history_data()

    # Fetch weather data from CFBD Pro tier (if available)
    weather_dict = fetch_weather(season, week, season_type)
    if weather_dict:
        logger.info(f"Weather data available for {len(weather_dict)} games")
    else:
        logger.info("No weather data available (CFBD Pro tier required)")

    # Try V22 first, fall back to V19 if V22 not available
    if USE_V22_MODEL:
        try:
            model = load_v22_model()
            logger.info("Using V22 Meta-Router Model for predictions")

            # Use V22 prediction generation
            df = generate_v22_predictions(
                games=games,
                lines_dict=lines_dict,
                model=model,
                history_df=history_df,
                season=season,
                week=week,
                bankroll=bankroll,
                weather_dict=weather_dict,
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
                    'game_type': row.get('game_type', 'Unknown'),  # V22: Game type
                    'router_confidence': row.get('router_confidence', 0.5),  # V22
                    'uncertainty': row.get('uncertainty', 7.0),  # V22: NGBoost uncertainty
                    'blowout_prob': row.get('blowout_prob', 0.0),  # V22: Blowout probability
                    'start_date': row.get('start_date'),
                    'completed': row.get('completed', False),
                    'game_id': row.get('game_id'),
                })

            return predictions

        except FileNotFoundError as e:
            logger.warning(f"V22 model not found, falling back to V19: {e}")
        except Exception as e:
            logger.error(f"V22 prediction failed, falling back to V19: {e}")

    # Fall back to V19
    logger.info("Using V19 dual-target model for predictions")
    model = load_v19_model()

    # V21: Fetch injury data for QB availability tracking
    injury_df = None
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from fetch_injuries import fetch_all_injuries

        # Get list of teams in games
        teams = []
        for game in games:
            home = game.get('homeTeam') or game.get('home_team')
            away = game.get('awayTeam') or game.get('away_team')
            if home:
                teams.append(home)
            if away:
                teams.append(away)

        if teams:
            injury_df = fetch_all_injuries(teams)
            if not injury_df.empty:
                logger.info(f"Injury data available for {injury_df['team'].nunique()} teams")
    except Exception as e:
        logger.warning(f"Could not fetch injury data: {e}")

    # Use the shared generate_v19_predictions function
    df = generate_v19_predictions(
        games=games,
        lines_dict=lines_dict,
        model=model,
        history_df=history_df,
        season=season,
        week=week,
        bankroll=bankroll,
        weather_dict=weather_dict,
        injury_df=injury_df
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
