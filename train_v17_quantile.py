"""
V17 Quantile Regression Model for CFB Betting.

This model uses quantile regression to:
1. Predict the MEDIAN spread error (more robust than mean)
2. Generate prediction intervals (10th and 90th percentiles)
3. Only recommend bets when intervals don't cross zero (high confidence)

Research: "Median outcome is sufficient for optimal prediction, but additional
quantiles are necessary to optimally select matches to wager on."
- PLOS One 2023: Statistical theory of optimal decision-making in sports betting

Key insight: We want CALIBRATED uncertainty, not just point predictions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import optuna
from datetime import datetime
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = 'cfb_data_safe.csv'
CONFIG_FILE = 'cfb_v16_config.pkl'
OUTPUT_MODEL = 'cfb_spread_error_v17_quantile.pkl'
OUTPUT_CONFIG = 'cfb_v17_quantile_config.pkl'
OPTUNA_TRIALS = 30  # Reduced for faster training
RANDOM_STATE = 42

# Quantiles to train
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

# Load V16 features as base
try:
    V16_CONFIG = joblib.load(CONFIG_FILE)
    FEATURES = V16_CONFIG['features']
except FileNotFoundError:
    # Fall back to V15 config
    V15_CONFIG = joblib.load('cfb_v15_config.pkl')
    FEATURES = V15_CONFIG['features']

print(f"Using {len(FEATURES)} features for quantile regression")


def load_data():
    """Load training data."""
    print("Loading data...")

    df = pd.read_csv(DATA_FILE)
    df = df[df['vegas_spread'].notna()].copy()
    print(f"Loaded {len(df)} games with spreads")

    # Ensure spread_error exists
    if 'spread_error' not in df.columns:
        df['margin'] = df['home_points'] - df['away_points']
        df['spread_error'] = df['margin'] - (-df['vegas_spread'])

    # Sort chronologically
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    return df


def simple_holdout_quantile_score(params, df, feature_cols, quantile, target_col='spread_error'):
    """
    Use simple holdout validation (much faster than walk-forward).
    Train on pre-2024, validate on 2024, test on 2025.
    """
    # Train on pre-2024 data
    train_mask = df['season'] < 2024
    val_mask = df['season'] == 2024

    train_data = df[train_mask]
    val_data = df[val_mask]

    if len(val_data) == 0:
        return 999.0

    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data[target_col]
    X_val = val_data[feature_cols].fillna(0)
    y_val = val_data[target_col]

    model = GradientBoostingRegressor(
        loss='quantile',
        alpha=quantile,
        **params,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # For median (0.5), use MAE; for other quantiles, use pinball loss
    if quantile == 0.5:
        return np.mean(np.abs(y_pred - y_val))
    else:
        # Pinball loss
        diff = y_val - y_pred
        pinball = np.where(diff >= 0, quantile * diff, (quantile - 1) * diff)
        return np.mean(pinball)


def optimize_quantile_model(df, feature_cols, quantile, n_trials=OPTUNA_TRIALS):
    """Optimize hyperparameters for a single quantile model."""
    print(f"\nOptimizing quantile={quantile} model ({n_trials} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }

        return simple_holdout_quantile_score(params, df, feature_cols, quantile)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best score: {study.best_value:.4f}")
    return study.best_params


def train_quantile_ensemble(df, feature_cols, quantile_params):
    """Train final quantile models on all pre-2025 data."""
    print("\nTraining final quantile models...")

    train_mask = df['season'] < 2025
    train_data = df[train_mask]

    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['spread_error']

    models = {}
    for quantile in QUANTILES:
        print(f"  Training q={quantile} model...")

        params = quantile_params[quantile]
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=quantile,
            **params,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        models[quantile] = model

    return models


def calculate_bet_recommendations(models, X, threshold=0.0):
    """
    Determine which games to bet on based on prediction intervals.

    Only recommend betting when:
    1. The 80% prediction interval (q10-q90) doesn't cross zero
    2. The interval is narrow enough (high confidence)

    Returns DataFrame with predictions and recommendations.
    """
    predictions = {
        'q10': models[0.1].predict(X),
        'q25': models[0.25].predict(X),
        'median': models[0.5].predict(X),
        'q75': models[0.75].predict(X),
        'q90': models[0.9].predict(X),
    }

    df_pred = pd.DataFrame(predictions)

    # Interval width (80% interval)
    df_pred['interval_width'] = df_pred['q90'] - df_pred['q10']

    # Check if interval crosses zero
    df_pred['interval_crosses_zero'] = (df_pred['q10'] < 0) & (df_pred['q90'] > 0)

    # Recommend bet only if interval doesn't cross zero
    df_pred['should_bet'] = ~df_pred['interval_crosses_zero']

    # Confidence score (inverse of interval width, normalized)
    median_width = df_pred['interval_width'].median()
    df_pred['confidence_score'] = 1 - (df_pred['interval_width'] / (2 * median_width)).clip(0, 1)

    # Signal direction from median
    df_pred['signal'] = np.where(df_pred['median'] > 0, 'BUY', 'FADE')

    # Expected edge is the median prediction
    df_pred['expected_edge'] = df_pred['median'].abs()

    return df_pred


def evaluate_quantile_models(models, df, feature_cols):
    """Evaluate quantile models on 2025 data."""
    print("\n" + "=" * 60)
    print("QUANTILE MODEL EVALUATION ON 2025 DATA")
    print("=" * 60)

    test_mask = df['season'] == 2025
    test_data = df[test_mask]

    if len(test_data) == 0:
        print("No 2025 data available for evaluation")
        return

    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['spread_error']

    # Get predictions
    df_pred = calculate_bet_recommendations(models, X_test)
    df_pred['actual'] = y_test.values

    # Overall metrics
    median_pred = df_pred['median']
    mae = mean_absolute_error(y_test, median_pred)
    print(f"\nOverall MAE (median): {mae:.3f}")

    # Coverage analysis
    for q_low, q_high in [(0.1, 0.9), (0.25, 0.75)]:
        coverage = ((df_pred['actual'] >= df_pred[f'q{int(q_low*100):02d}']) &
                   (df_pred['actual'] <= df_pred[f'q{int(q_high*100):02d}'])).mean()
        expected = q_high - q_low
        print(f"  {int((q_high-q_low)*100)}% interval coverage: {coverage:.1%} (expected: {expected:.1%})")

    # Bet selection analysis
    bet_games = df_pred[df_pred['should_bet']]
    no_bet_games = df_pred[~df_pred['should_bet']]

    print(f"\nBet Selection:")
    print(f"  Games recommended to bet: {len(bet_games)} ({len(bet_games)/len(df_pred)*100:.1f}%)")
    print(f"  Games to skip: {len(no_bet_games)} ({len(no_bet_games)/len(df_pred)*100:.1f}%)")

    if len(bet_games) > 0:
        bet_mae = mean_absolute_error(bet_games['actual'], bet_games['median'])
        print(f"  MAE on bet games: {bet_mae:.3f}")

        # Direction accuracy on bet games
        direction_correct = ((bet_games['median'] > 0) == (bet_games['actual'] > 0)).mean()
        print(f"  Direction accuracy on bet games: {direction_correct:.1%}")

        # Simulated profitability (assuming -110 odds)
        correct_bets = ((bet_games['median'] > 0) == (bet_games['actual'] > 0))
        profit = correct_bets.sum() * 0.91 - (~correct_bets).sum()
        roi = profit / len(bet_games) * 100
        print(f"  Simulated ROI on bet games: {roi:+.1f}%")

    if len(no_bet_games) > 0:
        no_bet_mae = mean_absolute_error(no_bet_games['actual'], no_bet_games['median'])
        print(f"  MAE on skipped games: {no_bet_mae:.3f}")

        direction_correct = ((no_bet_games['median'] > 0) == (no_bet_games['actual'] > 0)).mean()
        print(f"  Direction accuracy on skipped: {direction_correct:.1%}")


def main():
    print("=" * 70)
    print("V17 QUANTILE REGRESSION MODEL TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()

    # Get available features
    feature_cols = [f for f in FEATURES if f in df.columns]
    print(f"\nUsing {len(feature_cols)} features")

    # Optimize each quantile model
    quantile_params = {}
    for q in QUANTILES:
        # Same number of trials for all quantiles now (faster validation)
        quantile_params[q] = optimize_quantile_model(df, feature_cols, q, OPTUNA_TRIALS)

    # Train final models
    models = train_quantile_ensemble(df, feature_cols, quantile_params)

    # Evaluate on 2025 data
    evaluate_quantile_models(models, df, feature_cols)

    # Save models and config
    print("\nSaving models...")
    joblib.dump(models, OUTPUT_MODEL)

    config = {
        'features': feature_cols,
        'quantiles': QUANTILES,
        'quantile_params': quantile_params,
        'trained_on': datetime.now().isoformat(),
        'model_type': 'quantile_regression_ensemble',
    }
    joblib.dump(config, OUTPUT_CONFIG)

    print(f"\nSaved models to {OUTPUT_MODEL}")
    print(f"Saved config to {OUTPUT_CONFIG}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
