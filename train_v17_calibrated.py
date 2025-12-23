"""
V17 Calibration-Optimized Model for CFB Betting.

Key Research Insight (Walsh & Joshi 2024):
"Calibration-optimized models generate 69.86% higher returns than accuracy-optimized models."

Instead of optimizing for MAE (accuracy), we optimize for CALIBRATION:
- When model predicts 60% confidence, it should win ~60% of the time
- This leads to better betting decisions than raw accuracy

The model still predicts spread_error, but hyperparameters are tuned to minimize
calibration error rather than MAE.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.calibration import calibration_curve
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
OUTPUT_MODEL = 'cfb_spread_error_v17_calibrated.pkl'
OUTPUT_CONFIG = 'cfb_v17_calibrated_config.pkl'
OPTUNA_TRIALS = 150
RANDOM_STATE = 42

# Calibration vs MAE weight in objective
# Research suggests calibration is more important for betting
CALIBRATION_WEIGHT = 0.7
MAE_WEIGHT = 0.3

# Load V16 features
try:
    V16_CONFIG = joblib.load(CONFIG_FILE)
    FEATURES = V16_CONFIG['features']
except FileNotFoundError:
    V15_CONFIG = joblib.load('cfb_v15_config.pkl')
    FEATURES = V15_CONFIG['features']

print(f"Using {len(FEATURES)} features")


def load_data():
    """Load training data."""
    print("Loading data...")

    df = pd.read_csv(DATA_FILE)
    df = df[df['vegas_spread'].notna()].copy()
    print(f"Loaded {len(df)} games with spreads")

    if 'spread_error' not in df.columns:
        df['margin'] = df['home_points'] - df['away_points']
        df['spread_error'] = df['margin'] - (-df['vegas_spread'])

    df = df.sort_values(['season', 'week']).reset_index(drop=True)
    return df


def prediction_to_probability(pred, scale=5.0):
    """
    Convert spread error prediction to win probability using sigmoid.

    pred: predicted spread error (positive = home covers)
    scale: scaling factor (higher = sharper probability curve)

    Returns probability that prediction is correct (home covers when pred > 0).
    """
    # Sigmoid centered at 0
    prob = 1 / (1 + np.exp(-pred / scale))
    return prob


def calculate_calibration_error(y_true, y_pred, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    This measures how well predicted probabilities match actual outcomes.
    Lower is better.
    """
    # Convert to binary classification: did prediction direction match actual?
    y_binary = (y_true > 0).astype(int)
    y_prob = prediction_to_probability(y_pred)

    try:
        prob_true, prob_pred = calibration_curve(y_binary, y_prob, n_bins=n_bins, strategy='uniform')

        # ECE = weighted average of |accuracy - confidence| per bin
        bin_weights = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0] / len(y_prob)
        ece = np.sum(np.abs(prob_true - prob_pred) * bin_weights[:len(prob_true)])

        return ece
    except ValueError:
        # Not enough data for calibration curve
        return 1.0


def walk_forward_calibrated_score(model_class, params, df, feature_cols, target_col='spread_error'):
    """
    Calculate combined calibration + MAE score using walk-forward validation.

    Returns weighted combination of calibration error and MAE.
    """
    all_preds = []
    all_actuals = []
    errors = []

    for test_week in range(1, 16):
        train_mask = (df['season'] < 2025) | ((df['season'] == 2025) & (df['week'] < test_week))
        test_mask = (df['season'] == 2025) & (df['week'] == test_week)

        train_data = df[train_mask]
        test_data = df[test_mask]

        if len(test_data) == 0:
            continue

        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data[target_col]
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data[target_col]

        model = model_class(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_preds.extend(y_pred.tolist())
        all_actuals.extend(y_test.tolist())
        errors.extend(np.abs(y_pred - y_test).tolist())

    if not all_preds:
        return 999.0

    # Calculate metrics
    mae = np.mean(errors)
    calibration_error = calculate_calibration_error(np.array(all_actuals), np.array(all_preds))

    # Combine with weights
    # Normalize MAE to similar scale as calibration error (0-1)
    mae_normalized = mae / 20.0  # Typical MAE is ~10, so this gives ~0.5

    combined_score = CALIBRATION_WEIGHT * calibration_error + MAE_WEIGHT * mae_normalized

    return combined_score


def optimize_xgb_calibrated(df, feature_cols):
    """Optimize XGBoost for calibration."""
    print(f"\nOptimizing XGBoost for calibration ({OPTUNA_TRIALS} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'n_jobs': -1,
        }

        return walk_forward_calibrated_score(XGBRegressor, params, df, feature_cols)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    print(f"  Best calibrated score: {study.best_value:.4f}")
    return study.best_params


def optimize_hgb_calibrated(df, feature_cols):
    """Optimize HistGradientBoosting for calibration."""
    print(f"\nOptimizing HistGradientBoosting for calibration ({OPTUNA_TRIALS // 2} trials)...")

    def objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
        }

        return walk_forward_calibrated_score(HistGradientBoostingRegressor, params, df, feature_cols)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS // 2, show_progress_bar=True)

    print(f"  Best calibrated score: {study.best_value:.4f}")
    return study.best_params


def train_calibrated_ensemble(df, feature_cols, xgb_params, hgb_params):
    """Train final ensemble optimized for calibration."""
    print("\nTraining calibrated ensemble...")

    train_mask = df['season'] < 2025
    train_data = df[train_mask]

    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['spread_error']

    # Train XGBoost
    xgb_model = XGBRegressor(**xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    print("  XGBoost trained")

    # Train HGB
    hgb_model = HistGradientBoostingRegressor(**hgb_params, random_state=RANDOM_STATE)
    hgb_model.fit(X_train, y_train)
    print("  HistGradientBoosting trained")

    return {'xgb': xgb_model, 'hgb': hgb_model}


def find_optimal_ensemble_weights(models, df, feature_cols):
    """Find optimal ensemble weights based on calibration."""
    print("\nFinding optimal ensemble weights...")

    test_mask = df['season'] == 2025
    test_data = df[test_mask]

    if len(test_data) == 0:
        print("  No 2025 data, using equal weights")
        return {'xgb': 0.5, 'hgb': 0.5}

    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['spread_error']

    xgb_pred = models['xgb'].predict(X_test)
    hgb_pred = models['hgb'].predict(X_test)

    best_weights = {'xgb': 0.5, 'hgb': 0.5}
    best_score = float('inf')

    # Grid search over weights
    for xgb_w in np.arange(0.3, 0.8, 0.05):
        hgb_w = 1 - xgb_w
        ensemble_pred = xgb_w * xgb_pred + hgb_w * hgb_pred

        cal_error = calculate_calibration_error(y_test, ensemble_pred)

        if cal_error < best_score:
            best_score = cal_error
            best_weights = {'xgb': xgb_w, 'hgb': hgb_w}

    print(f"  Best weights: XGB={best_weights['xgb']:.2f}, HGB={best_weights['hgb']:.2f}")
    print(f"  Best calibration error: {best_score:.4f}")

    return best_weights


def evaluate_calibration(models, weights, df, feature_cols):
    """Evaluate calibration on 2025 data."""
    print("\n" + "=" * 60)
    print("CALIBRATION EVALUATION ON 2025 DATA")
    print("=" * 60)

    test_mask = df['season'] == 2025
    test_data = df[test_mask]

    if len(test_data) == 0:
        print("No 2025 data available")
        return

    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['spread_error']

    # Ensemble prediction
    xgb_pred = models['xgb'].predict(X_test)
    hgb_pred = models['hgb'].predict(X_test)
    y_pred = weights['xgb'] * xgb_pred + weights['hgb'] * hgb_pred

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    cal_error = calculate_calibration_error(y_test, y_pred)

    print(f"\nOverall Metrics:")
    print(f"  MAE: {mae:.3f}")
    print(f"  Calibration Error (ECE): {cal_error:.4f}")

    # Direction accuracy
    direction_acc = ((y_pred > 0) == (y_test > 0)).mean()
    print(f"  Direction Accuracy: {direction_acc:.1%}")

    # Calibration by confidence bin
    print("\nCalibration by Predicted Confidence:")
    y_prob = prediction_to_probability(y_pred)
    y_binary = (y_test > 0).astype(int)

    for low, high in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        mask = (y_prob >= low) & (y_prob < high)
        if mask.sum() > 0:
            actual_acc = y_binary[mask].mean()
            expected = (low + high) / 2
            n_games = mask.sum()
            print(f"  {low:.0%}-{high:.0%} confidence: {actual_acc:.1%} actual ({n_games} games, expected {expected:.0%})")

    # Simulated betting performance
    print("\nSimulated Betting Performance:")

    # Only bet on high confidence (>60%)
    high_conf_mask = y_prob > 0.6
    if high_conf_mask.sum() > 0:
        correct = ((y_pred[high_conf_mask] > 0) == (y_test.values[high_conf_mask] > 0)).sum()
        total = high_conf_mask.sum()
        profit = correct * 0.91 - (total - correct)
        roi = profit / total * 100
        print(f"  High confidence (>60%): {correct}/{total} = {correct/total:.1%}, ROI: {roi:+.1f}%")

    # Only bet on very high confidence (>70%)
    very_high_conf_mask = y_prob > 0.7
    if very_high_conf_mask.sum() > 0:
        correct = ((y_pred[very_high_conf_mask] > 0) == (y_test.values[very_high_conf_mask] > 0)).sum()
        total = very_high_conf_mask.sum()
        profit = correct * 0.91 - (total - correct)
        roi = profit / total * 100
        print(f"  Very high confidence (>70%): {correct}/{total} = {correct/total:.1%}, ROI: {roi:+.1f}%")


def main():
    print("=" * 70)
    print("V17 CALIBRATION-OPTIMIZED MODEL TRAINING")
    print("=" * 70)
    print(f"Calibration weight: {CALIBRATION_WEIGHT}, MAE weight: {MAE_WEIGHT}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()

    # Get available features
    feature_cols = [f for f in FEATURES if f in df.columns]
    print(f"\nUsing {len(feature_cols)} features")

    # Optimize models for calibration
    xgb_params = optimize_xgb_calibrated(df, feature_cols)
    hgb_params = optimize_hgb_calibrated(df, feature_cols)

    # Train final ensemble
    models = train_calibrated_ensemble(df, feature_cols, xgb_params, hgb_params)

    # Find optimal weights
    weights = find_optimal_ensemble_weights(models, df, feature_cols)

    # Evaluate calibration
    evaluate_calibration(models, weights, df, feature_cols)

    # Save models and config
    print("\nSaving models...")

    ensemble = {
        'models': models,
        'weights': weights,
    }
    joblib.dump(ensemble, OUTPUT_MODEL)

    config = {
        'features': feature_cols,
        'xgb_params': xgb_params,
        'hgb_params': hgb_params,
        'weights': weights,
        'calibration_weight': CALIBRATION_WEIGHT,
        'mae_weight': MAE_WEIGHT,
        'trained_on': datetime.now().isoformat(),
        'model_type': 'calibration_optimized_ensemble',
    }
    joblib.dump(config, OUTPUT_CONFIG)

    print(f"\nSaved models to {OUTPUT_MODEL}")
    print(f"Saved config to {OUTPUT_CONFIG}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
