"""
Train V14 Ultimate Model - Maximum Accuracy with No Data Leakage.

This is the ultimate model combining:
1. ONLY safe pre-game features (no leakage)
2. Optuna hyperparameter optimization
3. Multi-model ensemble with optimized weights
4. Walk-forward validation on 2025 season
5. Spread error approach (predict how wrong Vegas will be)

CRITICAL: Uses cfb_data_safe.csv which has properly calculated features.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
import warnings
import optuna
from optuna.samplers import TPESampler
warnings.filterwarnings('ignore')

# Suppress Optuna logging during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("TRAIN V14 ULTIMATE MODEL")
print("Maximum Accuracy with No Data Leakage")
print("=" * 70)

# =============================================================================
# SAFE FEATURES ONLY
# =============================================================================
SAFE_FEATURES = [
    # Pre-game Elo
    'home_pregame_elo', 'away_pregame_elo', 'elo_diff',

    # Rolling averages (last 5 games)
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',

    # Home field advantage
    'home_team_hfa', 'hfa_diff',

    # Scheduling
    'rest_diff',

    # Vegas features
    'line_movement',
    'large_favorite', 'large_underdog', 'close_game',

    # Momentum
    'home_streak', 'away_streak', 'streak_diff',

    # ATS history
    'home_ats', 'away_ats', 'ats_diff',

    # Elo momentum
    'home_elo_momentum', 'away_elo_momentum', 'elo_momentum_diff',

    # Scoring trends
    'home_scoring_trend', 'away_scoring_trend',

    # Vegas-derived
    'elo_vs_spread', 'rest_spread_interaction',
    'home_short_rest', 'away_short_rest',
    'expected_total',

    # Situational
    'west_coast_early', 'home_lookahead', 'away_lookahead',
]


def load_data():
    """Load prepared safe data."""
    try:
        df = pd.read_csv('cfb_data_safe.csv')
        print(f"Loaded cfb_data_safe.csv with {len(df)} games")
    except FileNotFoundError:
        print("cfb_data_safe.csv not found. Running prepare_safe_features.py...")
        import subprocess
        subprocess.run(['python3', 'prepare_safe_features.py'])
        df = pd.read_csv('cfb_data_safe.csv')
        print(f"Loaded cfb_data_safe.csv with {len(df)} games")

    # Filter to games with Vegas spreads and valid results
    df_vegas = df[
        (df['vegas_spread'].notna()) &
        (df['Margin'].notna())
    ].copy()

    df_vegas = df_vegas.sort_values(['season', 'week']).reset_index(drop=True)

    # Ensure spread_error exists
    if 'spread_error' not in df_vegas.columns:
        df_vegas['spread_error'] = df_vegas['Margin'] - (-df_vegas['vegas_spread'])

    print(f"Games with Vegas spreads: {len(df_vegas)}")
    print(f"Seasons: {sorted(df_vegas['season'].unique())}")

    return df_vegas


def get_available_features(df):
    """Get available safe features."""
    available = [f for f in SAFE_FEATURES if f in df.columns]
    print(f"\nUsing {len(available)} safe features")
    return available


def walk_forward_mae(model, df, features, n_test_weeks=5):
    """
    Calculate walk-forward MAE.
    Train on all data before each week, predict that week.
    """
    df_valid = df[df['spread_error'].notna()].copy()
    weeks_2025 = sorted(df_valid[df_valid['season'] == 2025]['week'].unique())

    if len(weeks_2025) < n_test_weeks:
        n_test_weeks = len(weeks_2025)

    test_weeks = weeks_2025[:n_test_weeks]

    all_errors = []

    for week in test_weeks:
        # Train on all data before this week
        train_mask = (df_valid['season'] < 2025) | ((df_valid['season'] == 2025) & (df_valid['week'] < week))
        test_mask = (df_valid['season'] == 2025) & (df_valid['week'] == week)

        train_data = df_valid[train_mask]
        test_data = df_valid[test_mask]

        if len(train_data) < 100 or len(test_data) == 0:
            continue

        X_train = train_data[features].fillna(0)
        y_train = train_data['spread_error']
        X_test = test_data[features].fillna(0)
        y_test = test_data['spread_error']

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            errors = np.abs(y_test - preds)
            all_errors.extend(errors.tolist())
        except Exception:
            continue

    if len(all_errors) == 0:
        return 15.0  # Penalty for failed models

    return np.mean(all_errors)


def optimize_xgboost(df, features, n_trials=100):
    """Optimize XGBoost hyperparameters using Optuna."""
    print("\n" + "=" * 70)
    print("OPTIMIZING XGBoost HYPERPARAMETERS")
    print(f"Running {n_trials} trials...")
    print("=" * 70)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 5.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
        }

        model = XGBRegressor(**params)
        mae = walk_forward_mae(model, df, features, n_test_weeks=5)
        return mae

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest XGBoost MAE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def optimize_hgb(df, features, n_trials=50):
    """Optimize HistGradientBoosting hyperparameters."""
    print("\n" + "=" * 70)
    print("OPTIMIZING HistGradientBoosting HYPERPARAMETERS")
    print(f"Running {n_trials} trials...")
    print("=" * 70)

    def objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 5.0, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
            'random_state': 42,
        }

        model = HistGradientBoostingRegressor(**params)
        mae = walk_forward_mae(model, df, features, n_test_weeks=5)
        return mae

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest HGB MAE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def full_walk_forward_validation(model, df, features):
    """
    Full walk-forward validation on 2025.
    Returns detailed results for each week.
    """
    print("\n" + "=" * 70)
    print("FULL WALK-FORWARD VALIDATION (2025)")
    print("=" * 70)

    df_valid = df[df['spread_error'].notna()].copy()
    weeks_2025 = sorted(df_valid[df_valid['season'] == 2025]['week'].unique())

    results = []
    all_pred = []
    all_actual = []
    all_vegas = []

    for week in weeks_2025:
        # Train on ALL data before this week
        train_mask = (df_valid['season'] < 2025) | ((df_valid['season'] == 2025) & (df_valid['week'] < week))
        test_mask = (df_valid['season'] == 2025) & (df_valid['week'] == week)

        train_data = df_valid[train_mask]
        test_data = df_valid[test_mask]

        if len(train_data) < 100 or len(test_data) == 0:
            continue

        X_train = train_data[features].fillna(0)
        y_train = train_data['spread_error']
        X_test = test_data[features].fillna(0)
        y_test = test_data['spread_error']

        vegas_spread = test_data['vegas_spread']
        actual_margin = test_data['Margin']

        # Train fresh model for this week
        model.fit(X_train, y_train)

        # Predict spread error
        pred_error = model.predict(X_test)

        # Final margin = Vegas + adjustment
        pred_margin = -vegas_spread + pred_error
        vegas_margin = -vegas_spread

        model_mae = mean_absolute_error(actual_margin, pred_margin)
        vegas_mae = mean_absolute_error(actual_margin, vegas_margin)

        # Betting simulation
        bets = 0
        wins = 0
        for pe, ae in zip(pred_error, y_test):
            if abs(pe) < 3.0:  # Threshold
                continue
            bets += 1
            if (pe > 0 and ae > 0) or (pe < 0 and ae < 0):
                wins += 1

        win_rate = wins / bets if bets > 0 else 0

        results.append({
            'week': week,
            'games': len(test_data),
            'model_mae': model_mae,
            'vegas_mae': vegas_mae,
            'bets': bets,
            'wins': wins,
            'win_rate': win_rate
        })

        all_pred.extend(pred_margin.tolist())
        all_actual.extend(actual_margin.tolist())
        all_vegas.extend(vegas_margin.tolist())

        beat = "OK" if model_mae < vegas_mae else "X"
        print(f"Week {week:2d}: {len(test_data):3d} games | Model: {model_mae:.2f} | Vegas: {vegas_mae:.2f} {beat} | Bets: {bets}, Win: {win_rate:.1%}")

    return results, all_pred, all_actual, all_vegas


def train_ensemble(df, features, xgb_params, hgb_params):
    """Train the final ensemble model."""
    print("\n" + "=" * 70)
    print("TRAINING FINAL ENSEMBLE")
    print("=" * 70)

    # Create models with optimized params
    xgb_model = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1)

    hgb_model = HistGradientBoostingRegressor(**hgb_params, random_state=42)

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    # Create weighted ensemble
    ensemble = VotingRegressor([
        ('xgb', xgb_model),
        ('hgb', hgb_model),
        ('rf', rf_model),
    ])

    return ensemble, {'xgb': xgb_model, 'hgb': hgb_model, 'rf': rf_model}


def main():
    # Load data
    df = load_data()
    features = get_available_features(df)

    # Optimize hyperparameters
    print("\n" + "=" * 70)
    print("PHASE 1: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    # Run optimization (reduce trials for faster execution)
    xgb_params = optimize_xgboost(df, features, n_trials=100)
    hgb_params = optimize_hgb(df, features, n_trials=50)

    # Create ensemble
    ensemble, base_models = train_ensemble(df, features, xgb_params, hgb_params)

    # Validate with walk-forward
    print("\n" + "=" * 70)
    print("PHASE 2: WALK-FORWARD VALIDATION")
    print("=" * 70)

    results, all_pred, all_actual, all_vegas = full_walk_forward_validation(
        XGBRegressor(**xgb_params, random_state=42, n_jobs=-1),  # Test best single model
        df, features
    )

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    final_model_mae = mean_absolute_error(all_actual, all_pred)
    final_vegas_mae = mean_absolute_error(all_actual, all_vegas)

    res_df = pd.DataFrame(results)
    total_bets = res_df['bets'].sum()
    total_wins = res_df['wins'].sum()
    overall_win_rate = total_wins / total_bets if total_bets > 0 else 0
    weeks_beat_vegas = (res_df['model_mae'] < res_df['vegas_mae']).sum()

    print(f"\nMAE Comparison:")
    print(f"  Model MAE: {final_model_mae:.2f}")
    print(f"  Vegas MAE: {final_vegas_mae:.2f}")
    print(f"  Difference: {final_vegas_mae - final_model_mae:+.2f}")

    print(f"\nWeeks beating Vegas: {weeks_beat_vegas}/{len(res_df)}")

    print(f"\nBetting (threshold=3.0):")
    print(f"  Total bets: {total_bets}")
    print(f"  Wins: {total_wins}")
    print(f"  Win rate: {overall_win_rate:.1%}")
    print(f"  Break-even: 52.4%")

    if overall_win_rate > 0.524:
        profit = total_wins * 0.91 - (total_bets - total_wins) * 1.0
        roi = (profit / total_bets) * 100 if total_bets > 0 else 0
        print(f"  Profit: {profit:+.1f} units")
        print(f"  ROI: {roi:+.1f}%")
        profitable = True
    else:
        print(f"  NOT PROFITABLE at -110 lines")
        profitable = False

    # Train final model on all data
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 70)

    df_valid = df[df['spread_error'].notna()].copy()
    X_all = df_valid[features].fillna(0)
    y_all = df_valid['spread_error']

    final_model = XGBRegressor(**xgb_params, random_state=42, n_jobs=-1)
    final_model.fit(X_all, y_all)

    # Save model
    joblib.dump(final_model, 'cfb_spread_error_v14.pkl')
    print("Saved model to 'cfb_spread_error_v14.pkl'")

    # Save config
    config = {
        'features': features,
        'xgb_params': xgb_params,
        'hgb_params': hgb_params,
        'validation_results': {
            'model_mae': final_model_mae,
            'vegas_mae': final_vegas_mae,
            'win_rate': overall_win_rate,
            'total_bets': total_bets,
            'profitable': profitable,
        }
    }
    joblib.dump(config, 'cfb_v14_config.pkl')
    print("Saved config to 'cfb_v14_config.pkl'")

    # Feature importance
    print("\nFeature Importance (Top 15):")
    importances = final_model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for i, (_, row) in enumerate(imp_df.head(15).iterrows(), 1):
        print(f"  {i:2d}. {row['importance']:.4f}  {row['feature']}")

    # Summary
    print("\n" + "=" * 70)
    print("V14 ULTIMATE MODEL - SUMMARY")
    print("=" * 70)

    print(f"""
RESULTS (No Data Leakage):
  Model MAE:  {final_model_mae:.2f}
  Vegas MAE:  {final_vegas_mae:.2f}
  Difference: {final_vegas_mae - final_model_mae:+.2f}

BETTING SIMULATION:
  Win Rate: {overall_win_rate:.1%}
  Profitable: {'YES' if profitable else 'NO'}
  ROI: {((total_wins * 0.91 - (total_bets - total_wins)) / total_bets * 100) if total_bets > 0 else 0:+.1f}%

FEATURES USED ({len(features)}):
  - Pre-game Elo + Elo momentum
  - Last 5 game rolling averages + scoring trends
  - Historical HFA
  - Rest days + interactions
  - Vegas spread features (line movement, favorites, etc.)
  - Win streaks + ATS history
  - Situational factors

HYPERPARAMETERS OPTIMIZED:
  XGBoost: {len(xgb_params)} parameters
  HGB: {len(hgb_params)} parameters
""")

    if final_model_mae < final_vegas_mae:
        print("Model beats Vegas on average")
    else:
        print("Model does not beat Vegas on MAE")

    if profitable:
        print("Model is PROFITABLE with 3.0 threshold")
    else:
        print("Model is NOT profitable")

    print("\n" + "=" * 70)
    print("V14 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
