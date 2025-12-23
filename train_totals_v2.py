"""
Train Totals Model V2 for CFB Betting.

APPROACH: Predict TOTALS ERROR (how wrong Vegas O/U will be)
- Target: totals_error = actual_total - vegas_ou
- Positive = went OVER, Negative = went UNDER
- Similar approach to successful spread error model

Uses Optuna for hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
import optuna
from datetime import datetime
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
OPTUNA_TRIALS = 100
RANDOM_STATE = 42

# Features that predict total points (all pre-game)
TOTALS_FEATURES = [
    # Scoring averages (direct total predictors)
    'home_last5_score_avg',
    'away_last5_score_avg',
    'home_last5_defense_avg',
    'away_last5_defense_avg',

    # Offensive efficiency (high PPA = more points)
    'home_comp_off_ppa',
    'away_comp_off_ppa',
    'home_comp_def_ppa',
    'away_comp_def_ppa',

    # Passing efficiency (pass-heavy = faster scoring)
    'home_comp_pass_ppa',
    'away_comp_pass_ppa',
    'home_comp_rush_ppa',
    'away_comp_rush_ppa',

    # Success rates (better execution = more scoring)
    'home_comp_success',
    'away_comp_success',

    # Vegas line (they know something)
    'over_under',

    # Spread magnitude (lopsided games can go high or low)
    'vegas_spread',

    # Elo (quality teams score more consistently)
    'home_pregame_elo',
    'away_pregame_elo',
    'elo_diff',
]

# Derived features we'll calculate
DERIVED_FEATURES = [
    'expected_total',        # home_last5_score + away_last5_score
    'off_ppa_sum',           # Total offensive efficiency
    'def_ppa_sum',           # Total defensive efficiency
    'net_efficiency',        # off_ppa_sum - def_ppa_sum
    'pass_heavy_index',      # Combined pass PPA
    'success_sum',           # Combined success rates
    'spread_magnitude',      # abs(vegas_spread)
    'is_blowout_expected',   # abs(spread) > 17
    'is_close_expected',     # abs(spread) < 7
]


def load_and_prepare_data():
    """Load data and prepare features for totals prediction."""
    print("Loading cfb_data_smart.csv...")
    df = pd.read_csv('cfb_data_smart.csv')
    print(f"Total games loaded: {len(df)}")

    # Filter to games with over_under data
    df = df[df['over_under'].notna()].copy()
    print(f"Games with O/U data: {len(df)}")

    # Calculate total points (target)
    df['actual_total'] = df['home_points'] + df['away_points']

    # Calculate totals error (our target)
    df['totals_error'] = df['actual_total'] - df['over_under']

    # Sort chronologically
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    return df


def engineer_features(df):
    """Create derived features for totals prediction."""
    print("\nEngineering features...")

    # Expected total from recent scoring
    df['expected_total'] = df['home_last5_score_avg'].fillna(28) + df['away_last5_score_avg'].fillna(28)

    # Offensive efficiency sum
    df['off_ppa_sum'] = df['home_comp_off_ppa'].fillna(0) + df['away_comp_off_ppa'].fillna(0)

    # Defensive efficiency sum (lower = better defense = fewer points)
    df['def_ppa_sum'] = df['home_comp_def_ppa'].fillna(0) + df['away_comp_def_ppa'].fillna(0)

    # Net efficiency
    df['net_efficiency'] = df['off_ppa_sum'] - df['def_ppa_sum']

    # Pass-heavy index (more passing = faster games typically)
    df['pass_heavy_index'] = df['home_comp_pass_ppa'].fillna(0) + df['away_comp_pass_ppa'].fillna(0)

    # Success rate sum
    df['success_sum'] = df['home_comp_success'].fillna(0.4) + df['away_comp_success'].fillna(0.4)

    # Spread magnitude
    df['spread_magnitude'] = df['vegas_spread'].abs()

    # Blowout expected (high spreads often go under - garbage time)
    df['is_blowout_expected'] = (df['spread_magnitude'] > 17).astype(int)

    # Close game expected (often go under - defensive battles or over - shootouts)
    df['is_close_expected'] = (df['spread_magnitude'] < 7).astype(int)

    # Fill any remaining NaN
    for feat in TOTALS_FEATURES + DERIVED_FEATURES:
        if feat in df.columns:
            df[feat] = df[feat].fillna(0)

    return df


def get_feature_columns():
    """Get the list of features to use."""
    features = TOTALS_FEATURES + DERIVED_FEATURES
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for f in features:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def walk_forward_validation(model_class, params, df, feature_cols, target_col='totals_error'):
    """Walk-forward validation for 2025 season."""
    errors = []

    # Train on 2022-2024, validate on 2025 weeks
    train_data = df[df['season'].isin([2022, 2023, 2024])].copy()
    test_data = df[df['season'] == 2025].copy()

    if len(test_data) == 0:
        # If no 2025 data, use 2024 as test
        train_data = df[df['season'].isin([2022, 2023])].copy()
        test_data = df[df['season'] == 2024].copy()

    if len(test_data) == 0:
        return 999.0  # No test data

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    model = model_class(**params, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    return mae


def optimize_model(df, feature_cols):
    """Use Optuna to find optimal hyperparameters."""
    print(f"\nOptimizing with Optuna ({OPTUNA_TRIALS} trials)...")

    def objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
        }

        mae = walk_forward_validation(
            HistGradientBoostingRegressor,
            params,
            df,
            feature_cols
        )
        return mae

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    print(f"\nBest trial MAE: {study.best_trial.value:.2f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def train_final_model(df, feature_cols, best_params):
    """Train final model with best parameters."""
    print("\nTraining final model...")

    # Determine train/test split
    if 2025 in df['season'].values:
        train_data = df[df['season'].isin([2022, 2023, 2024])]
        test_data = df[df['season'] == 2025]
        test_year = 2025
    else:
        train_data = df[df['season'].isin([2022, 2023])]
        test_data = df[df['season'] == 2024]
        test_year = 2024

    X_train = train_data[feature_cols]
    y_train = train_data['totals_error']
    X_test = test_data[feature_cols]
    y_test = test_data['totals_error']

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples ({test_year}): {len(X_test)}")

    # Train model
    model = HistGradientBoostingRegressor(
        **best_params,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, y_pred)

    # Vegas baseline (predicting 0 error = trusting Vegas)
    vegas_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))

    print(f"\nModel MAE: {model_mae:.2f} points")
    print(f"Vegas MAE: {vegas_mae:.2f} points")
    print(f"Improvement: {vegas_mae - model_mae:+.2f} points")

    if model_mae < vegas_mae:
        print("✅ Model beats Vegas!")
    else:
        print("⚠️ Model does not beat Vegas baseline")

    return model, X_test, y_test, y_pred, test_year


def evaluate_betting_performance(test_df, y_pred, threshold=2.0):
    """Evaluate O/U betting performance."""
    print(f"\n{'='*60}")
    print(f"BETTING PERFORMANCE (Edge > {threshold} pts)")
    print("="*60)

    # Create results dataframe
    results = test_df[['home_team', 'away_team', 'over_under', 'actual_total', 'totals_error']].copy()
    results['predicted_error'] = y_pred
    results['predicted_total'] = results['over_under'] + results['predicted_error']

    # Betting signal
    results['model_bet'] = np.where(
        results['predicted_error'] > threshold, 'OVER',
        np.where(results['predicted_error'] < -threshold, 'UNDER', 'PASS')
    )

    # Actual result
    results['actual_result'] = np.where(
        results['actual_total'] > results['over_under'], 'OVER',
        np.where(results['actual_total'] < results['over_under'], 'UNDER', 'PUSH')
    )

    # Filter to bets
    bets = results[results['model_bet'] != 'PASS']
    print(f"\nTotal bets: {len(bets)}")

    if len(bets) > 0:
        # Calculate wins
        wins = ((bets['model_bet'] == 'OVER') & (bets['actual_result'] == 'OVER')).sum()
        wins += ((bets['model_bet'] == 'UNDER') & (bets['actual_result'] == 'UNDER')).sum()

        losses = ((bets['model_bet'] == 'OVER') & (bets['actual_result'] == 'UNDER')).sum()
        losses += ((bets['model_bet'] == 'UNDER') & (bets['actual_result'] == 'OVER')).sum()

        pushes = (bets['actual_result'] == 'PUSH').sum()

        total_decided = wins + losses
        if total_decided > 0:
            win_rate = wins / total_decided * 100

            print(f"Wins: {wins}")
            print(f"Losses: {losses}")
            print(f"Pushes: {pushes}")
            print(f"Win Rate: {win_rate:.1f}%")

            # Profitability (assuming -110 odds)
            profit = wins * 100 - losses * 110
            roi = profit / (total_decided * 110) * 100

            print(f"\nProfit (units): {profit/100:+.1f}")
            print(f"ROI: {roi:+.1f}%")

            if win_rate > 52.4:
                print(f"\n✅ PROFITABLE: {win_rate:.1f}% > 52.4% breakeven")
            else:
                print(f"\n⚠️ Below breakeven: {win_rate:.1f}% < 52.4%")

    return results


def save_model(model, feature_cols, best_params, metrics):
    """Save model and configuration."""
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print("="*60)

    # Save model
    joblib.dump(model, 'cfb_totals_v2.pkl')
    print("Model saved to 'cfb_totals_v2.pkl'")

    # Save config
    config = {
        'features': feature_cols,
        'params': best_params,
        'metrics': metrics,
        'trained_at': datetime.now().isoformat(),
        'approach': 'totals_error_prediction',
    }
    joblib.dump(config, 'cfb_totals_v2_config.pkl')
    print("Config saved to 'cfb_totals_v2_config.pkl'")


def main():
    print("="*60)
    print("TRAIN TOTALS MODEL V2")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load and prepare data
    df = load_and_prepare_data()

    # Engineer features
    df = engineer_features(df)

    # Get feature columns
    feature_cols = get_feature_columns()
    print(f"\nUsing {len(feature_cols)} features:")
    for f in feature_cols:
        print(f"  - {f}")

    # Check for missing features
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"\n⚠️ Missing features: {missing}")
        feature_cols = [f for f in feature_cols if f in df.columns]

    # Optimize hyperparameters
    best_params = optimize_model(df, feature_cols)

    # Train final model
    model, X_test, y_test, y_pred, test_year = train_final_model(df, feature_cols, best_params)

    # Get test dataframe for evaluation
    if test_year == 2025:
        test_df = df[df['season'] == 2025]
    else:
        test_df = df[df['season'] == 2024]

    # Evaluate betting performance
    results = evaluate_betting_performance(test_df, y_pred, threshold=2.0)

    # Calculate metrics
    model_mae = mean_absolute_error(y_test, y_pred)
    vegas_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))

    metrics = {
        'model_mae': model_mae,
        'vegas_mae': vegas_mae,
        'improvement': vegas_mae - model_mae,
        'test_year': test_year,
        'test_samples': len(X_test),
    }

    # Save model
    save_model(model, feature_cols, best_params, metrics)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"""
Totals Model V2 Training Complete:

Approach: Predict Vegas O/U Error (not raw total)
Features: {len(feature_cols)}
Training: 2022-{test_year-1}
Testing: {test_year}

Performance:
  Model MAE: {model_mae:.2f} points
  Vegas MAE: {vegas_mae:.2f} points
  Improvement: {vegas_mae - model_mae:+.2f} points

Files:
  - cfb_totals_v2.pkl: Trained model
  - cfb_totals_v2_config.pkl: Configuration
""")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
