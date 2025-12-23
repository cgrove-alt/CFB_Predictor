"""
V17 Situation-Specific Models for CFB Betting.

Based on error analysis, different game situations have vastly different prediction accuracy:
- Elite vs Elite (Elo > 1600): 7.50 MAE, 75.6% accuracy (BEST)
- Mismatch (Elo diff > 300): 7.93 MAE, 76.4% accuracy
- Average vs Average: 11.35 MAE, 55.4% accuracy (WORST - 51% of games!)
- Blowout (spread > 21): 10.21 MAE
- Early Season (week <= 3): Higher variance due to cold start

Solution: Train specialized models for each situation type, then route
predictions to the appropriate specialist.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
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
OUTPUT_MODEL = 'cfb_spread_error_v17_situational.pkl'
OUTPUT_CONFIG = 'cfb_v17_situational_config.pkl'
OPTUNA_TRIALS_PER_SITUATION = 50
RANDOM_STATE = 42

# Situation definitions
SITUATIONS = {
    'elite_matchup': {
        'description': 'Both teams Elo > 1600',
        'historical_mae': 7.50,
        'historical_acc': 0.756,
    },
    'mismatch': {
        'description': 'Elo diff > 300',
        'historical_mae': 7.93,
        'historical_acc': 0.764,
    },
    'blowout': {
        'description': 'Spread > 21 points',
        'historical_mae': 10.21,
        'historical_acc': 0.660,
    },
    'early_season': {
        'description': 'Week 1-3 (cold start)',
        'historical_mae': 9.64,
        'historical_acc': 0.670,
    },
    'average_vs_average': {
        'description': 'Neither elite nor mismatch',
        'historical_mae': 11.35,
        'historical_acc': 0.554,
    },
}

# Load features
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


def classify_situation(row):
    """
    Classify a game into a situation category.

    Priority order (first match wins):
    1. Early season (week <= 3)
    2. Elite matchup (both teams > 1600 Elo)
    3. Mismatch (Elo diff > 300)
    4. Blowout (spread > 21)
    5. Average vs Average (default)
    """
    week = row.get('week', 10)
    home_elo = row.get('home_pregame_elo', 1500)
    away_elo = row.get('away_pregame_elo', 1500)
    spread = abs(row.get('vegas_spread', 7))

    elo_diff = abs(home_elo - away_elo)

    # Early season games have different dynamics
    if week <= 3:
        return 'early_season'

    # Elite teams playing each other
    if home_elo > 1600 and away_elo > 1600:
        return 'elite_matchup'

    # Big mismatch
    if elo_diff > 300:
        return 'mismatch'

    # Blowout spread
    if spread > 21:
        return 'blowout'

    # Default: average teams
    return 'average_vs_average'


def add_situation_column(df):
    """Add situation classification to dataframe."""
    df['situation'] = df.apply(classify_situation, axis=1)
    return df


def get_situation_data(df, situation):
    """Get data for a specific situation."""
    return df[df['situation'] == situation].copy()


def walk_forward_mae(model_class, params, df, feature_cols, target_col='spread_error'):
    """Calculate walk-forward MAE."""
    errors = []

    for test_week in range(1, 16):
        train_mask = (df['season'] < 2025) | ((df['season'] == 2025) & (df['week'] < test_week))
        test_mask = (df['season'] == 2025) & (df['week'] == test_week)

        train_data = df[train_mask]
        test_data = df[test_mask]

        if len(test_data) == 0 or len(train_data) < 50:
            continue

        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data[target_col]
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data[target_col]

        model = model_class(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        errors.extend(np.abs(y_pred - y_test).tolist())

    return np.mean(errors) if errors else 999.0


def optimize_situation_model(df_situation, feature_cols, situation_name, n_trials):
    """Optimize model for a specific situation."""
    print(f"\nOptimizing {situation_name} model ({n_trials} trials)...")
    print(f"  Training games: {len(df_situation[df_situation['season'] < 2025])}")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'n_jobs': -1,
        }

        return walk_forward_mae(XGBRegressor, params, df_situation, feature_cols)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best MAE: {study.best_value:.4f}")
    return study.best_params


def train_situation_models(df, feature_cols, situation_params):
    """Train final models for each situation."""
    print("\nTraining situation-specific models...")

    train_mask = df['season'] < 2025
    train_data = df[train_mask]

    models = {}
    for situation in SITUATIONS.keys():
        situation_data = train_data[train_data['situation'] == situation]

        if len(situation_data) < 50:
            print(f"  {situation}: Not enough data ({len(situation_data)} games), using general model")
            # Use all training data for situations with insufficient data
            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data['spread_error']
        else:
            X_train = situation_data[feature_cols].fillna(0)
            y_train = situation_data['spread_error']

        params = situation_params.get(situation, {
            'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05
        })

        model = XGBRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
        models[situation] = model

        print(f"  {situation}: trained on {len(X_train)} games")

    return models


def train_general_fallback(df, feature_cols):
    """Train a general model as fallback."""
    print("\nTraining general fallback model...")

    train_mask = df['season'] < 2025
    train_data = df[train_mask]

    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['spread_error']

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model


def predict_with_routing(models, fallback_model, X, situations):
    """
    Route predictions to appropriate situation-specific model.

    Returns predictions using the specialist model for each game's situation.
    """
    predictions = np.zeros(len(X))

    for situation in SITUATIONS.keys():
        mask = situations == situation
        if mask.sum() > 0:
            if situation in models:
                predictions[mask] = models[situation].predict(X[mask])
            else:
                predictions[mask] = fallback_model.predict(X[mask])

    return predictions


def evaluate_situational(models, fallback_model, df, feature_cols):
    """Evaluate situational models on 2025 data."""
    print("\n" + "=" * 60)
    print("SITUATIONAL MODEL EVALUATION ON 2025 DATA")
    print("=" * 60)

    test_mask = df['season'] == 2025
    test_data = df[test_mask]

    if len(test_data) == 0:
        print("No 2025 data available")
        return

    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['spread_error'].values
    situations = test_data['situation'].values

    # Routed predictions
    y_pred_routed = predict_with_routing(models, fallback_model, X_test, situations)

    # General model predictions (for comparison)
    y_pred_general = fallback_model.predict(X_test)

    # Overall metrics
    print("\nOverall Performance:")
    mae_routed = mean_absolute_error(y_test, y_pred_routed)
    mae_general = mean_absolute_error(y_test, y_pred_general)
    print(f"  Situational MAE: {mae_routed:.3f}")
    print(f"  General MAE: {mae_general:.3f}")
    print(f"  Improvement: {(mae_general - mae_routed) / mae_general * 100:+.1f}%")

    direction_routed = ((y_pred_routed > 0) == (y_test > 0)).mean()
    direction_general = ((y_pred_general > 0) == (y_test > 0)).mean()
    print(f"  Situational Direction Acc: {direction_routed:.1%}")
    print(f"  General Direction Acc: {direction_general:.1%}")

    # Per-situation breakdown
    print("\nPer-Situation Performance:")
    print(f"{'Situation':<25} {'Games':>6} {'MAE':>8} {'Dir Acc':>10} {'Historical':>12}")
    print("-" * 65)

    for situation in SITUATIONS.keys():
        mask = situations == situation
        if mask.sum() > 0:
            sit_mae = mean_absolute_error(y_test[mask], y_pred_routed[mask])
            sit_acc = ((y_pred_routed[mask] > 0) == (y_test[mask] > 0)).mean()
            hist_mae = SITUATIONS[situation]['historical_mae']

            improvement = (hist_mae - sit_mae) / hist_mae * 100 if hist_mae > 0 else 0

            print(f"{situation:<25} {mask.sum():>6} {sit_mae:>8.2f} {sit_acc:>10.1%} {hist_mae:>8.2f} ({improvement:+.1f}%)")

    # Betting simulation
    print("\nBetting Simulation (routed predictions):")

    # All bets
    correct = ((y_pred_routed > 0) == (y_test > 0))
    profit = correct.sum() * 0.91 - (~correct).sum()
    roi = profit / len(correct) * 100
    print(f"  All games: {correct.sum()}/{len(correct)} = {correct.mean():.1%}, ROI: {roi:+.1f}%")

    # High-confidence bets (large predicted error)
    high_conf_mask = np.abs(y_pred_routed) >= 5.0
    if high_conf_mask.sum() > 0:
        correct_hc = ((y_pred_routed[high_conf_mask] > 0) == (y_test[high_conf_mask] > 0))
        profit_hc = correct_hc.sum() * 0.91 - (~correct_hc).sum()
        roi_hc = profit_hc / len(correct_hc) * 100
        print(f"  High confidence (|error| >= 5): {correct_hc.sum()}/{len(correct_hc)} = {correct_hc.mean():.1%}, ROI: {roi_hc:+.1f}%")


def main():
    print("=" * 70)
    print("V17 SITUATION-SPECIFIC MODEL TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()

    # Add situation classification
    df = add_situation_column(df)

    # Show situation distribution
    print("\nSituation Distribution:")
    situation_counts = df['situation'].value_counts()
    for sit, count in situation_counts.items():
        pct = count / len(df) * 100
        print(f"  {sit}: {count} games ({pct:.1f}%)")

    # Get available features
    feature_cols = [f for f in FEATURES if f in df.columns]
    print(f"\nUsing {len(feature_cols)} features")

    # Optimize model for each situation
    situation_params = {}
    for situation in SITUATIONS.keys():
        df_situation = get_situation_data(df, situation)

        if len(df_situation) >= 100:
            # Enough data to optimize
            params = optimize_situation_model(
                df_situation, feature_cols, situation,
                n_trials=OPTUNA_TRIALS_PER_SITUATION
            )
            situation_params[situation] = params
        else:
            print(f"\n{situation}: Not enough data ({len(df_situation)} games), using defaults")
            situation_params[situation] = {
                'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05,
                'n_jobs': -1
            }

    # Train final models
    models = train_situation_models(df, feature_cols, situation_params)
    fallback_model = train_general_fallback(df, feature_cols)

    # Evaluate
    evaluate_situational(models, fallback_model, df, feature_cols)

    # Save models and config
    print("\nSaving models...")

    model_package = {
        'situation_models': models,
        'fallback_model': fallback_model,
    }
    joblib.dump(model_package, OUTPUT_MODEL)

    config = {
        'features': feature_cols,
        'situation_params': situation_params,
        'situations': SITUATIONS,
        'trained_on': datetime.now().isoformat(),
        'model_type': 'situational_routing',
    }
    joblib.dump(config, OUTPUT_CONFIG)

    print(f"\nSaved models to {OUTPUT_MODEL}")
    print(f"Saved config to {OUTPUT_CONFIG}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
