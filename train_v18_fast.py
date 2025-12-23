"""
V18 Fast Retraining Script

Retrains models using pre-optimized hyperparameters (no Optuna tuning).
Completes in 15-20 minutes instead of 3-4 hours.

Usage:
    python train_v18_fast.py
"""

import os
import sys
import shutil
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import optional libraries
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False
    LGBMRegressor = None
    logger.info("LightGBM not available, skipping...")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except (ImportError, OSError):
    HAS_CATBOOST = False
    CatBoostRegressor = None
    logger.info("CatBoost not available, skipping...")

# Paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / 'cfb_data_safe.csv'
CONFIG_FILE = BASE_DIR / 'cfb_v18_stacking_config.pkl'
FEATURE_CONFIG = BASE_DIR / 'cfb_v16_config.pkl'
OUTPUT_MODEL = BASE_DIR / 'cfb_v18_stacking.pkl'
OUTPUT_CONFIG = BASE_DIR / 'cfb_v18_stacking_config.pkl'
MODELS_DIR = BASE_DIR / 'models'

RANDOM_STATE = 42

# Default hyperparameters (if config not found)
DEFAULT_PARAMS = {
    'xgboost': {
        'n_estimators': 400,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    },
    'hgb': {
        'max_iter': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'min_samples_leaf': 20,
        'max_leaf_nodes': 31,
        'l2_regularization': 1.0,
    },
    'lgbm': {
        'n_estimators': 400,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    },
    'catboost': {
        'iterations': 300,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
    }
}


def load_params():
    """Load pre-optimized hyperparameters from config."""
    if CONFIG_FILE.exists():
        try:
            config = joblib.load(CONFIG_FILE)
            if 'params' in config:
                logger.info(f"Loaded hyperparameters from {CONFIG_FILE}")
                return config['params']
        except Exception as e:
            logger.warning(f"Could not load params from config: {e}")

    logger.info("Using default hyperparameters")
    return DEFAULT_PARAMS


def load_features():
    """Load feature list from config."""
    # Try V18 config first
    if CONFIG_FILE.exists():
        try:
            config = joblib.load(CONFIG_FILE)
            if 'features' in config:
                logger.info(f"Loaded {len(config['features'])} features from {CONFIG_FILE}")
                return config['features']
        except Exception:
            pass

    # Try V16 config
    if FEATURE_CONFIG.exists():
        try:
            config = joblib.load(FEATURE_CONFIG)
            if 'features' in config:
                logger.info(f"Loaded {len(config['features'])} features from {FEATURE_CONFIG}")
                return config['features']
        except Exception:
            pass

    # Fallback
    logger.warning("No feature config found, using defaults")
    return [
        'home_pregame_elo', 'away_pregame_elo', 'elo_diff',
        'home_last5_score_avg', 'away_last5_score_avg',
        'home_last5_defense_avg', 'away_last5_defense_avg',
        'home_team_hfa', 'hfa_diff', 'rest_diff',
        'vegas_spread', 'line_movement',
        'home_streak', 'away_streak', 'streak_diff',
        'home_ats', 'away_ats', 'ats_diff',
        'home_elo_momentum', 'away_elo_momentum',
        'home_scoring_trend', 'away_scoring_trend',
        'large_favorite', 'large_underdog', 'close_game',
        'elo_vs_spread', 'expected_total',
    ]


def load_data():
    """Load and prepare training data."""
    logger.info(f"Loading data from {DATA_FILE}...")

    df = pd.read_csv(DATA_FILE)

    # Only use games with spreads
    df = df[df['vegas_spread'].notna()].copy()

    # Calculate spread_error if missing
    if 'spread_error' not in df.columns:
        df['spread_error'] = df['Margin'] - (-df['vegas_spread'])

    # Sort chronologically
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    logger.info(f"Total games: {len(df)}")
    return df


def backup_current_model():
    """Backup current model before retraining."""
    if not OUTPUT_MODEL.exists():
        return None

    # Create models directory if needed
    MODELS_DIR.mkdir(exist_ok=True)

    # Generate backup name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"cfb_v18_stacking_{timestamp}.pkl"
    backup_path = MODELS_DIR / backup_name

    shutil.copy(OUTPUT_MODEL, backup_path)
    logger.info(f"Backed up current model to {backup_path}")

    # Also backup config
    if OUTPUT_CONFIG.exists():
        config_backup = MODELS_DIR / f"cfb_v18_stacking_config_{timestamp}.pkl"
        shutil.copy(OUTPUT_CONFIG, config_backup)

    # Clean old backups (keep last 3)
    backups = sorted(MODELS_DIR.glob('cfb_v18_stacking_*.pkl'))
    if len(backups) > 3:
        for old_backup in backups[:-3]:
            old_backup.unlink()
            logger.info(f"Deleted old backup: {old_backup}")

    return backup_path


def train_fast(df, features, params):
    """Train models with fixed hyperparameters (no tuning)."""
    logger.info("=" * 60)
    logger.info("FAST RETRAINING (no hyperparameter optimization)")
    logger.info("=" * 60)

    # Use all data up to current season for training
    current_season = df['season'].max()
    train_df = df[df['season'] < current_season].copy()

    # Filter features that exist in data
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        logger.warning(f"Missing features: {missing}")

    X_train = train_df[available_features].fillna(0)
    y_train = train_df['spread_error']

    logger.info(f"Training on {len(train_df)} games with {len(available_features)} features")

    base_models = {}

    # XGBoost
    logger.info("Training XGBoost...")
    xgb_params = params['xgboost'].copy()
    xgb_params['random_state'] = RANDOM_STATE
    xgb_params['n_jobs'] = -1
    xgb_params['verbosity'] = 0
    base_models['xgboost'] = XGBRegressor(**xgb_params)
    base_models['xgboost'].fit(X_train, y_train)

    # HistGradientBoosting
    logger.info("Training HistGradientBoosting...")
    hgb_params = params['hgb'].copy()
    hgb_params['random_state'] = RANDOM_STATE
    base_models['hgb'] = HistGradientBoostingRegressor(**hgb_params)
    base_models['hgb'].fit(X_train, y_train)

    # LightGBM (optional)
    if HAS_LGBM and 'lgbm' in params:
        logger.info("Training LightGBM...")
        lgbm_params = params['lgbm'].copy()
        lgbm_params['random_state'] = RANDOM_STATE
        lgbm_params['n_jobs'] = -1
        lgbm_params['verbose'] = -1
        base_models['lgbm'] = LGBMRegressor(**lgbm_params)
        base_models['lgbm'].fit(X_train, y_train)

    # CatBoost (optional)
    if HAS_CATBOOST and 'catboost' in params:
        logger.info("Training CatBoost...")
        cb_params = params['catboost'].copy()
        cb_params['random_state'] = RANDOM_STATE
        cb_params['verbose'] = False
        base_models['catboost'] = CatBoostRegressor(**cb_params)
        base_models['catboost'].fit(X_train, y_train)

    # Create meta-features
    logger.info("Creating meta-features...")
    meta_features = []
    for name, model in base_models.items():
        preds = model.predict(X_train)
        meta_features.append(preds)
        mae = mean_absolute_error(y_train, preds)
        logger.info(f"  {name}: MAE = {mae:.4f}")

    X_meta = np.column_stack(meta_features)
    X_meta_full = np.hstack([X_meta, X_train.values])

    # Train meta-learners
    logger.info("Training meta-learners...")
    meta_learner = BayesianRidge(max_iter=1000, compute_score=True)
    meta_learner.fit(X_meta_full, y_train)

    ridge_meta = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge_meta.fit(X_meta, y_train)

    return base_models, meta_learner, ridge_meta, available_features


def evaluate_model(df, base_models, meta_learner, ridge_meta, features):
    """Evaluate model on current season data."""
    current_season = df['season'].max()
    test_df = df[df['season'] == current_season].copy()

    if len(test_df) == 0:
        logger.warning("No current season data for evaluation")
        return None

    X_test = test_df[features].fillna(0)
    y_test = test_df['spread_error']

    logger.info(f"\nEvaluating on {len(test_df)} games from {current_season}...")

    # Get base predictions
    base_preds = {}
    for name, model in base_models.items():
        preds = model.predict(X_test)
        base_preds[name] = preds
        mae = mean_absolute_error(y_test, preds)
        direction = (np.sign(preds) == np.sign(y_test)).mean()
        logger.info(f"  {name}: MAE={mae:.3f}, Direction={direction:.1%}")

    # Meta-learner predictions
    X_meta = np.column_stack(list(base_preds.values()))
    X_meta_full = np.hstack([X_meta, X_test.values])

    bayes_preds = meta_learner.predict(X_meta_full)
    bayes_mae = mean_absolute_error(y_test, bayes_preds)
    bayes_dir = (np.sign(bayes_preds) == np.sign(y_test)).mean()
    logger.info(f"  Bayesian Meta: MAE={bayes_mae:.3f}, Direction={bayes_dir:.1%}")

    return {
        'mae': bayes_mae,
        'direction_accuracy': bayes_dir,
        'test_games': len(test_df)
    }


def save_model(base_models, meta_learner, ridge_meta, features, params, results):
    """Save the trained model bundle."""
    timestamp = datetime.now().strftime('%Y%m%d')
    version = f"v18_stacking_{timestamp}"

    model_bundle = {
        'base_models': base_models,
        'meta_learner': meta_learner,
        'ridge_meta': ridge_meta,
        'features': features,
        'model_names': list(base_models.keys()),
        'version': version,
        'trained_at': datetime.now().isoformat(),
        'retrain_type': 'fast'
    }

    joblib.dump(model_bundle, OUTPUT_MODEL)
    logger.info(f"Saved model to {OUTPUT_MODEL}")

    # Save config
    config = {
        'features': features,
        'params': params,
        'results': results,
        'version': version,
        'trained_at': datetime.now().isoformat()
    }
    joblib.dump(config, OUTPUT_CONFIG)
    logger.info(f"Saved config to {OUTPUT_CONFIG}")

    return version


def main():
    """Main fast retraining function."""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("STARTING FAST RETRAIN")
    logger.info(f"Time: {start_time}")
    logger.info("=" * 60)

    # Backup current model
    backup_current_model()

    # Load configuration
    params = load_params()
    features = load_features()
    df = load_data()

    # Train
    base_models, meta_learner, ridge_meta, used_features = train_fast(
        df, features, params
    )

    # Evaluate
    results = evaluate_model(df, base_models, meta_learner, ridge_meta, used_features)

    # Save
    version = save_model(
        base_models, meta_learner, ridge_meta,
        used_features, params, results
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info(f"FAST RETRAIN COMPLETE")
    logger.info(f"Version: {version}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    if results:
        logger.info(f"Test MAE: {results['mae']:.3f}")
        logger.info(f"Direction Accuracy: {results['direction_accuracy']:.1%}")
    logger.info("=" * 60)

    return {
        'success': True,
        'version': version,
        'results': results,
        'elapsed_minutes': elapsed / 60
    }


if __name__ == '__main__':
    result = main()
    sys.exit(0 if result['success'] else 1)
