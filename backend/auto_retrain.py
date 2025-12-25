"""
Auto-Retrain Script for Railway.

Runs weekly to retrain the V19 dual-target model with latest data.
Triggered by scheduler_worker.py on Sundays.

Usage:
    python auto_retrain.py
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import requests

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MODEL_DIR = DATA_DIR / "models"
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")
CFBD_BASE_URL = "https://api.collegefootballdata.com"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_headers():
    """Get CFBD API headers."""
    return {'Authorization': f'Bearer {CFBD_API_KEY}'}


def fetch_weather_data(years):
    """Fetch weather data from CFBD API (Patreon required)."""
    logger.info("Fetching weather data...")
    weather_data = []

    try:
        for year in years:
            for week in range(1, 20):
                try:
                    url = f"{CFBD_BASE_URL}/games/weather?year={year}&week={week}"
                    resp = requests.get(url, headers=get_headers(), timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        for w in data:
                            weather_data.append({
                                'game_id': w.get('id'),
                                'wind_speed': w.get('windSpeed'),
                                'temperature': w.get('temperature'),
                            })
                except Exception:
                    continue
        logger.info(f"  Fetched {len(weather_data)} weather records")
    except Exception as e:
        logger.warning(f"  Weather API error: {e}")

    return pd.DataFrame(weather_data) if weather_data else None


def add_weather_features(df, weather_df=None):
    """Add weather-derived features."""
    logger.info("Adding weather features...")

    # Merge weather data if available
    if weather_df is not None and len(weather_df) > 0:
        # Remove existing weather columns
        for col in ['wind_speed', 'temperature', 'game_id']:
            if col in df.columns:
                df = df.drop(columns=[col], errors='ignore')

        df = df.merge(weather_df, left_on='id', right_on='game_id', how='left')
        df = df.drop(columns=['game_id'], errors='ignore')

    # Fill missing values
    if 'wind_speed' not in df.columns:
        df['wind_speed'] = 0
    else:
        df['wind_speed'] = df['wind_speed'].fillna(0)

    if 'temperature' not in df.columns:
        df['temperature'] = 65
    else:
        df['temperature'] = df['temperature'].fillna(65)

    # Dome stadiums
    dome_stadiums = [
        'syracuse', 'georgia state', 'tulane', 'unlv',
        'new mexico', 'louisiana tech', 'northern illinois',
        'ford field', 'lucas oil stadium', 'at&t stadium',
        'mercedes-benz stadium', 'caesars superdome',
    ]

    df['is_dome'] = df['home_team'].str.lower().isin(dome_stadiums).astype(int)
    df['high_wind'] = ((df['wind_speed'] >= 15) & (df['is_dome'] == 0)).astype(int)
    df['cold_game'] = ((df['temperature'] <= 40) & (df['is_dome'] == 0)).astype(int)

    # Wind-pass interaction
    if 'home_comp_pass_ppa' in df.columns:
        pass_advantage = (df['home_comp_pass_ppa'].fillna(0) -
                          df['away_comp_pass_ppa'].fillna(0)).abs()
        df['wind_pass_impact'] = df['wind_speed'] * pass_advantage / 10
        df.loc[df['is_dome'] == 1, 'wind_pass_impact'] = 0
    else:
        df['wind_pass_impact'] = 0

    return df


def prepare_training_data():
    """Load and prepare data for training."""
    logger.info("Preparing training data...")

    # Load base data
    data_file = DATA_DIR / 'cfb_data_safe.csv'
    if not data_file.exists():
        # Fall back to bundled data
        data_file = Path('/app/cfb_data_safe.csv')

    if not data_file.exists():
        raise FileNotFoundError("No training data found!")

    df = pd.read_csv(data_file)
    logger.info(f"  Loaded {len(df)} games from {data_file}")

    # Fetch and add weather data
    current_year = datetime.now().year
    years = [current_year - 2, current_year - 1, current_year]
    weather_df = fetch_weather_data(years)

    df = add_weather_features(df, weather_df)

    return df


def train_model(df, n_trials=15):
    """Train the V19 dual-target model with reduced trials for speed."""
    logger.info("Training model...")

    # Import training module
    from train_v19_dual import V19DualTargetModel

    model = V19DualTargetModel()

    # Prepare data
    X, y_margin, y_cover, df_valid = model.prepare_data(df)
    logger.info(f"  Training on {len(X)} games with {len(model.feature_names)} features")

    # Train with reduced trials
    model.train(X, y_margin, y_cover, n_trials=n_trials)

    return model


def save_model(model, timestamp):
    """Save model to Railway volume with versioning."""
    logger.info("Saving model...")

    # Ensure model directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Generate versioned filename
    version_str = timestamp.strftime('%Y%m%d_%H%M')
    model_prefix = MODEL_DIR / f'cfb_v19_dual_{version_str}'

    # Save model files
    model.save(str(model_prefix))

    # Update current model pointer
    current_file = MODEL_DIR / 'current_model.txt'
    with open(current_file, 'w') as f:
        f.write(f'cfb_v19_dual_{version_str}')

    logger.info(f"  Saved model: {model_prefix}")
    logger.info(f"  Updated current_model.txt")

    return str(model_prefix)


def update_status(success, message, model_path=None):
    """Write training status for monitoring."""
    status = {
        'last_train': datetime.now().isoformat(),
        'success': success,
        'message': message,
        'model_path': model_path,
    }

    status_file = DATA_DIR / '.train_status.json'
    try:
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write status: {e}")


def should_retrain():
    """Check if retraining is needed."""
    # Check if forced via environment
    if os.getenv("FORCE_RETRAIN", "").lower() == "true":
        logger.info("Force retrain enabled via environment")
        return True

    # Check if it's Sunday (weekly retrain day)
    today = datetime.now()
    if today.weekday() != 6:  # 6 = Sunday
        logger.info(f"Skipping retrain - not Sunday (today is {today.strftime('%A')})")
        return False

    # Check if already trained today
    status_file = DATA_DIR / '.train_status.json'
    if status_file.exists():
        try:
            with open(status_file) as f:
                status = json.load(f)
            last_train = datetime.fromisoformat(status.get('last_train', '2000-01-01'))
            if last_train.date() == today.date():
                logger.info("Already trained today, skipping")
                return False
        except Exception:
            pass

    return True


def main():
    """Main retraining entry point."""
    logger.info("=" * 60)
    logger.info("AUTO-RETRAIN SCRIPT")
    logger.info("=" * 60)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    logger.info(f"MODEL_DIR: {MODEL_DIR}")

    if not CFBD_API_KEY:
        logger.error("CFBD_API_KEY not set!")
        update_status(False, "CFBD_API_KEY not configured")
        sys.exit(1)

    if not should_retrain():
        logger.info("Retraining not needed")
        sys.exit(0)

    timestamp = datetime.now()

    try:
        # Prepare data
        df = prepare_training_data()

        # Train model (reduced trials for Railway timeout)
        model = train_model(df, n_trials=15)

        # Save model
        model_path = save_model(model, timestamp)

        # Update status
        update_status(True, "Training completed successfully", model_path)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        update_status(False, str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
