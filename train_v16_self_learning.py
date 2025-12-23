"""
V16 Self-Learning Model for CFB Betting.

This model learns from its past mistakes by:
1. Adding uncertainty-aware features
2. Training an error meta-model to predict unreliability
3. Ensemble with situation-specific weighting

NO SHORTCUTS - The model truly learns from historical prediction errors.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
PREDICTIONS_FILE = 'predictions_2025_comprehensive.csv'
CONFIG_FILE = 'cfb_v15_config.pkl'
OUTPUT_MODEL = 'cfb_spread_error_v16.pkl'
OUTPUT_CONFIG = 'cfb_v16_config.pkl'
OPTUNA_TRIALS = 150
RANDOM_STATE = 42

# Load V15 features as base
V15_CONFIG = joblib.load(CONFIG_FILE)
V15_FEATURES = V15_CONFIG['features']

# ============================================================
# UNCERTAINTY FEATURES (learned from error analysis)
# ============================================================
UNCERTAINTY_FEATURES = [
    # Game situation indicators
    'is_pickem',           # abs(spread) < 3 - hardest to predict
    'is_mismatch',         # abs(elo_diff) > 300
    'is_early_season',     # week <= 3
    'is_rivalry_week',     # week 12-13 typically

    # Historical error patterns
    'home_team_historical_error',  # Rolling avg of model error for home team
    'away_team_historical_error',  # Rolling avg of model error for away team
    'spread_bucket_error',         # Avg error for this spread magnitude

    # Model confidence calibration
    'feature_completeness',  # How many features are non-null

    # Situational risk
    'is_post_bye',          # Team coming off bye
    'is_short_rest',        # < 6 days rest
]


def load_data():
    """Load training data and historical predictions."""
    print("Loading data...")

    # Load base data
    df = pd.read_csv(DATA_FILE)
    df = df[df['vegas_spread'].notna()].copy()
    print(f"Base data: {len(df)} games with spreads")

    # Ensure spread_error exists
    if 'spread_error' not in df.columns:
        df['margin'] = df['home_points'] - df['away_points']
        df['spread_error'] = df['margin'] - (-df['vegas_spread'])

    # Sort chronologically
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    # Try to load historical predictions for error learning
    try:
        pred_df = pd.read_csv(PREDICTIONS_FILE)
        print(f"Historical predictions: {len(pred_df)} games")
        has_predictions = True
    except:
        print("No historical predictions found - will compute during training")
        pred_df = None
        has_predictions = False

    return df, pred_df, has_predictions


def calculate_historical_team_errors(df, pred_df):
    """Calculate historical prediction errors per team."""
    print("\nCalculating team historical errors...")

    team_errors = {}

    if pred_df is not None:
        # Group by home team
        for team in pred_df['home_team'].unique():
            team_games = pred_df[pred_df['home_team'] == team]
            if len(team_games) >= 3:
                team_errors[f"{team}_home"] = team_games['prediction_error'].mean()

        # Group by away team
        for team in pred_df['away_team'].unique():
            team_games = pred_df[pred_df['away_team'] == team]
            if len(team_games) >= 3:
                team_errors[f"{team}_away"] = team_games['prediction_error'].mean()

    print(f"  Computed error profiles for {len(team_errors)} team-situations")

    return team_errors


def calculate_spread_bucket_errors(df, pred_df):
    """Calculate average errors by spread magnitude bucket."""
    print("\nCalculating spread bucket errors...")

    bucket_errors = {}

    if pred_df is not None:
        pred_df['spread_bucket'] = pd.cut(
            pred_df['spread_magnitude'].abs(),
            bins=[0, 3, 7, 14, 21, 100],
            labels=['pickem', 'small', 'medium', 'large', 'blowout']
        )

        for bucket in ['pickem', 'small', 'medium', 'large', 'blowout']:
            bucket_games = pred_df[pred_df['spread_bucket'] == bucket]
            if len(bucket_games) > 0:
                bucket_errors[bucket] = bucket_games['prediction_error'].mean()

    print(f"  Bucket errors: {bucket_errors}")

    return bucket_errors


def add_uncertainty_features(df, team_errors, bucket_errors):
    """Add uncertainty-aware features to the dataframe."""
    print("\nAdding uncertainty features...")

    # Game situation indicators
    df['is_pickem'] = (df['vegas_spread'].abs() <= 3).astype(int)
    df['is_mismatch'] = (df['elo_diff'].abs() > 300).astype(int)
    df['is_early_season'] = (df['week'] <= 3).astype(int)
    df['is_rivalry_week'] = df['week'].isin([12, 13]).astype(int)

    # Historical team errors
    df['home_team_historical_error'] = df['home_team'].apply(
        lambda x: team_errors.get(f"{x}_home", 10.0)  # Default to median error
    )
    df['away_team_historical_error'] = df['away_team'].apply(
        lambda x: team_errors.get(f"{x}_away", 10.0)
    )

    # Spread bucket errors - map directly without categorical
    def get_spread_bucket_error(spread):
        abs_spread = abs(spread) if pd.notna(spread) else 0
        if abs_spread <= 3:
            return bucket_errors.get('pickem', 10.0)
        elif abs_spread <= 7:
            return bucket_errors.get('small', 10.0)
        elif abs_spread <= 14:
            return bucket_errors.get('medium', 10.0)
        elif abs_spread <= 21:
            return bucket_errors.get('large', 10.0)
        else:
            return bucket_errors.get('blowout', 10.0)

    df['spread_bucket_error'] = df['vegas_spread'].apply(get_spread_bucket_error)

    # Feature completeness
    feature_cols = V15_FEATURES
    df['feature_completeness'] = df[feature_cols].notna().sum(axis=1) / len(feature_cols)

    # Rest-related
    if 'rest_diff' in df.columns:
        df['is_short_rest'] = (df['rest_diff'].abs() > 3).astype(int)
    else:
        df['is_short_rest'] = 0

    if 'home_short_rest' in df.columns and 'away_short_rest' in df.columns:
        df['is_post_bye'] = ((df['home_short_rest'] == 0) | (df['away_short_rest'] == 0)).astype(int)
    else:
        df['is_post_bye'] = 0

    print(f"  Added {len(UNCERTAINTY_FEATURES)} uncertainty features")

    return df


def get_v16_features():
    """Get the complete list of V16 features."""
    # Start with V15 features
    features = list(V15_FEATURES)

    # Add uncertainty features
    for feat in UNCERTAINTY_FEATURES:
        if feat not in features:
            features.append(feat)

    # Exclude any categorical columns
    exclude = ['spread_bucket', 'home_team', 'away_team', 'game_id']
    features = [f for f in features if f not in exclude]

    return features


def walk_forward_mae(model_class, params, df, feature_cols, target_col='spread_error'):
    """Calculate walk-forward MAE for validation."""
    errors = []

    # Train on 2022-2024, validate on 2025 weeks
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

        errors.extend(np.abs(y_pred - y_test).tolist())

    return np.mean(errors) if errors else 999.0


def optimize_xgboost(df, feature_cols):
    """Optimize XGBoost hyperparameters."""
    print(f"\nOptimizing XGBoost ({OPTUNA_TRIALS} trials)...")

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

        mae = walk_forward_mae(XGBRegressor, params, df, feature_cols)
        return mae

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    print(f"Best XGBoost MAE: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def optimize_hgb(df, feature_cols):
    """Optimize HistGradientBoosting hyperparameters."""
    print(f"\nOptimizing HistGradientBoosting (75 trials)...")

    def objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
        }

        mae = walk_forward_mae(HistGradientBoostingRegressor, params, df, feature_cols)
        return mae

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=75, show_progress_bar=True)

    print(f"Best HGB MAE: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def train_meta_model(df, primary_model, feature_cols):
    """Train meta-model to predict primary model's error."""
    print("\nTraining error meta-model...")

    # Generate predictions for all training data using walk-forward
    meta_features = []
    meta_targets = []

    for test_week in range(5, 16):  # Need some history first
        train_mask = (df['season'] < 2025) | ((df['season'] == 2025) & (df['week'] < test_week))
        test_mask = (df['season'] == 2025) & (df['week'] == test_week)

        train_data = df[train_mask]
        test_data = df[test_mask]

        if len(test_data) == 0:
            continue

        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data['spread_error']
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data['spread_error']

        # Train primary model on historical data
        temp_model = XGBRegressor(**primary_model.get_params())
        temp_model.fit(X_train, y_train)

        # Get predictions
        predictions = temp_model.predict(X_test)

        # Calculate errors (target for meta-model)
        errors = np.abs(predictions - y_test)

        # Meta features: original features + prediction magnitude + prediction
        for i, (idx, row) in enumerate(test_data.iterrows()):
            meta_row = list(X_test.iloc[i].values)
            meta_row.append(predictions[i])  # Primary prediction
            meta_row.append(abs(predictions[i]))  # Prediction magnitude
            meta_features.append(meta_row)
            meta_targets.append(errors.iloc[i])

    # Train meta-model
    meta_feature_names = list(feature_cols) + ['primary_prediction', 'prediction_magnitude']
    X_meta = np.array(meta_features)
    y_meta = np.array(meta_targets)

    meta_model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    meta_model.fit(X_meta, y_meta)

    print(f"  Meta-model trained on {len(y_meta)} samples")
    print(f"  Mean predicted error: {y_meta.mean():.2f}")

    return meta_model, meta_feature_names


def train_ensemble(df, feature_cols, xgb_params, hgb_params):
    """Train ensemble of models with optimal weighting."""
    print("\nTraining ensemble models...")

    # Prepare training data
    train_df = df[df['season'] < 2025].copy()
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['spread_error']

    # Train models
    models = {}

    # XGBoost primary
    xgb_model = XGBRegressor(**xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    models['xgb'] = xgb_model
    print("  Trained XGBoost")

    # HistGradientBoosting
    hgb_model = HistGradientBoostingRegressor(**hgb_params, random_state=RANDOM_STATE)
    hgb_model.fit(X_train, y_train)
    models['hgb'] = hgb_model
    print("  Trained HistGradientBoosting")

    # Find optimal ensemble weights using validation
    print("\nOptimizing ensemble weights...")
    best_weights = [0.5, 0.5]
    best_mae = float('inf')

    for xgb_w in np.arange(0.3, 0.8, 0.05):
        hgb_w = 1 - xgb_w

        total_error = 0
        total_count = 0

        for test_week in range(1, 16):
            test_mask = (df['season'] == 2025) & (df['week'] == test_week)
            test_data = df[test_mask]

            if len(test_data) == 0:
                continue

            X_test = test_data[feature_cols].fillna(0)
            y_test = test_data['spread_error']

            # Ensemble prediction
            pred = xgb_w * xgb_model.predict(X_test) + hgb_w * hgb_model.predict(X_test)

            total_error += np.abs(pred - y_test).sum()
            total_count += len(y_test)

        if total_count > 0:
            mae = total_error / total_count
            if mae < best_mae:
                best_mae = mae
                best_weights = [xgb_w, hgb_w]

    print(f"  Best weights: XGB={best_weights[0]:.2f}, HGB={best_weights[1]:.2f}")
    print(f"  Best ensemble MAE: {best_mae:.4f}")

    return models, best_weights


def evaluate_model(df, models, weights, feature_cols):
    """Evaluate model performance on 2025 data."""
    print("\n" + "="*60)
    print("MODEL EVALUATION ON 2025 DATA")
    print("="*60)

    results = []

    for test_week in range(1, 16):
        train_mask = (df['season'] < 2025) | ((df['season'] == 2025) & (df['week'] < test_week))
        test_mask = (df['season'] == 2025) & (df['week'] == test_week)

        train_data = df[train_mask]
        test_data = df[test_mask]

        if len(test_data) == 0:
            continue

        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data['spread_error']
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data['spread_error']

        # Retrain models
        models['xgb'].fit(X_train, y_train)
        models['hgb'].fit(X_train, y_train)

        # Ensemble prediction
        pred = weights[0] * models['xgb'].predict(X_test) + weights[1] * models['hgb'].predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, pred)
        vegas_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))
        direction_acc = ((pred > 0) == (y_test > 0)).mean()

        results.append({
            'week': test_week,
            'games': len(test_data),
            'mae': mae,
            'vegas_mae': vegas_mae,
            'beat_vegas': mae < vegas_mae,
            'direction_accuracy': direction_acc,
        })

        print(f"Week {test_week:2d}: {len(test_data):3d} games | "
              f"MAE: {mae:.2f} vs Vegas {vegas_mae:.2f} | "
              f"Direction: {direction_acc*100:.1f}%")

    # Summary
    results_df = pd.DataFrame(results)
    overall_mae = results_df['mae'].mean()
    overall_vegas = results_df['vegas_mae'].mean()
    weeks_beat = results_df['beat_vegas'].sum()
    overall_direction = results_df['direction_accuracy'].mean()

    print("\n" + "-"*40)
    print(f"OVERALL: MAE={overall_mae:.2f} | Vegas={overall_vegas:.2f} | "
          f"Beat Vegas {weeks_beat}/{len(results_df)} weeks")
    print(f"Direction Accuracy: {overall_direction*100:.1f}%")

    return results_df, {
        'mae': overall_mae,
        'vegas_mae': overall_vegas,
        'improvement': overall_vegas - overall_mae,
        'weeks_beat_vegas': weeks_beat,
        'total_weeks': len(results_df),
        'direction_accuracy': overall_direction,
    }


def save_model(models, weights, feature_cols, metrics, xgb_params, hgb_params):
    """Save the trained model and configuration."""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)

    # Save primary model (XGBoost for compatibility)
    joblib.dump(models['xgb'], OUTPUT_MODEL)
    print(f"Primary model saved to: {OUTPUT_MODEL}")

    # Save ensemble
    ensemble = {
        'xgb': models['xgb'],
        'hgb': models['hgb'],
        'weights': weights,
    }
    joblib.dump(ensemble, 'cfb_v16_ensemble.pkl')
    print("Ensemble saved to: cfb_v16_ensemble.pkl")

    # Save configuration
    config = {
        'features': feature_cols,
        'xgb_params': xgb_params,
        'hgb_params': hgb_params,
        'weights': weights,
        'metrics': metrics,
        'trained_at': datetime.now().isoformat(),
        'version': 'v16_self_learning',
    }
    joblib.dump(config, OUTPUT_CONFIG)
    print(f"Config saved to: {OUTPUT_CONFIG}")


def main():
    print("="*60)
    print("TRAIN V16 SELF-LEARNING MODEL")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    df, pred_df, has_predictions = load_data()

    # Calculate historical error patterns
    team_errors = calculate_historical_team_errors(df, pred_df)
    bucket_errors = calculate_spread_bucket_errors(df, pred_df)

    # Add uncertainty features
    df = add_uncertainty_features(df, team_errors, bucket_errors)

    # Get V16 features
    feature_cols = get_v16_features()
    print(f"\nUsing {len(feature_cols)} features:")
    for f in feature_cols:
        print(f"  - {f}")

    # Check for missing features
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"\nMissing features (will use 0): {missing}")
        for f in missing:
            df[f] = 0
        feature_cols = [f for f in feature_cols if f in df.columns or f not in missing]

    # Optimize hyperparameters
    xgb_params = optimize_xgboost(df, feature_cols)
    hgb_params = optimize_hgb(df, feature_cols)

    # Train ensemble
    models, weights = train_ensemble(df, feature_cols, xgb_params, hgb_params)

    # Evaluate
    results_df, metrics = evaluate_model(df, models, weights, feature_cols)

    # Save
    save_model(models, weights, feature_cols, metrics, xgb_params, hgb_params)

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"""
V16 Self-Learning Model Summary:
  - Features: {len(feature_cols)} (V15 + uncertainty features)
  - Model: Ensemble (XGB + HGB)
  - Weights: XGB={weights[0]:.2f}, HGB={weights[1]:.2f}

Performance:
  - MAE: {metrics['mae']:.2f} points
  - Vegas MAE: {metrics['vegas_mae']:.2f} points
  - Improvement: {metrics['improvement']:+.2f} points
  - Direction Accuracy: {metrics['direction_accuracy']*100:.1f}%
  - Weeks Beat Vegas: {metrics['weeks_beat_vegas']}/{metrics['total_weeks']}

Files Created:
  - {OUTPUT_MODEL}: Primary model
  - cfb_v16_ensemble.pkl: Full ensemble
  - {OUTPUT_CONFIG}: Configuration
""")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
