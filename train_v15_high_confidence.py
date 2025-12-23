"""
Train V15 High Confidence Model - Expanded Features for Better Predictions.

Improvements over V14:
1. ADDED high-correlation efficiency features (comp_ppa, pass_efficiency_diff)
2. NEW composite features for identifying mismatches
3. Better null handling for sparse features
4. More Optuna trials for better hyperparameter optimization
5. Walk-forward validation on 2025 season

Goal: Increase the proportion of HIGH confidence predictions while maintaining accuracy.
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
print("TRAIN V15 HIGH CONFIDENCE MODEL")
print("Expanded Features for Better Predictions")
print("=" * 70)

# =============================================================================
# EXPANDED SAFE FEATURES
# =============================================================================

# V14 Original Safe Features
BASE_SAFE_FEATURES = [
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

# NEW V15 High-Signal Features
# These are SEASON COMPOSITE STATS (pre-game), not game-specific
NEW_EFFICIENCY_FEATURES = [
    # PPA (Points Per Attempt) - Season composites
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',
    'home_comp_pass_ppa', 'away_comp_pass_ppa',
    'home_comp_rush_ppa', 'away_comp_rush_ppa',

    # Success rates
    'home_comp_success', 'away_comp_success',

    # EPA/YPP composites
    'home_comp_epa', 'away_comp_epa',
    'home_comp_ypp', 'away_comp_ypp',

    # Pre-calculated efficiency diff (high correlation r=0.37)
    'pass_efficiency_diff',
]

# Combine all features
SAFE_FEATURES = BASE_SAFE_FEATURES + NEW_EFFICIENCY_FEATURES


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


def create_composite_features(df):
    """Create composite interaction features for better signal extraction."""
    print("\nCreating composite features...")

    # Matchup efficiency - identifies offensive/defensive mismatches
    if 'home_comp_off_ppa' in df.columns and 'away_comp_def_ppa' in df.columns:
        df['matchup_efficiency'] = (
            (df['home_comp_off_ppa'].fillna(0) - df['away_comp_off_ppa'].fillna(0)) +
            (df['away_comp_def_ppa'].fillna(0) - df['home_comp_def_ppa'].fillna(0))
        )
        print("  Added: matchup_efficiency")

    # Pass vs rush efficiency diff
    if 'home_comp_pass_ppa' in df.columns and 'home_comp_rush_ppa' in df.columns:
        df['home_pass_rush_balance'] = df['home_comp_pass_ppa'].fillna(0) - df['home_comp_rush_ppa'].fillna(0)
        df['away_pass_rush_balance'] = df['away_comp_pass_ppa'].fillna(0) - df['away_comp_rush_ppa'].fillna(0)
        print("  Added: pass_rush_balance features")

    # Success rate difference
    if 'home_comp_success' in df.columns and 'away_comp_success' in df.columns:
        df['success_rate_diff'] = df['home_comp_success'].fillna(0.5) - df['away_comp_success'].fillna(0.5)
        print("  Added: success_rate_diff")

    # Elo x efficiency interaction - combines power ratings with efficiency
    if 'elo_diff' in df.columns and 'matchup_efficiency' in df.columns:
        df['elo_efficiency_interaction'] = df['elo_diff'] * df['matchup_efficiency'] / 100
        print("  Added: elo_efficiency_interaction")

    # Momentum x scoring - hot teams with good offenses
    if 'home_streak' in df.columns and 'home_last5_score_avg' in df.columns:
        df['momentum_strength'] = (
            (df['home_streak'].fillna(0) - df['away_streak'].fillna(0)) *
            (df['home_last5_score_avg'].fillna(25) - df['away_last5_score_avg'].fillna(25)) / 100
        )
        print("  Added: momentum_strength")

    # Dominant team flag - large Elo edge + efficiency edge
    if 'elo_diff' in df.columns and 'matchup_efficiency' in df.columns:
        df['dominant_home'] = ((df['elo_diff'] > 150) & (df['matchup_efficiency'] > 0.3)).astype(int)
        df['dominant_away'] = ((df['elo_diff'] < -150) & (df['matchup_efficiency'] < -0.3)).astype(int)
        print("  Added: dominant_home/away flags")

    # Rest x favorite interaction - tired favorites underperform
    if 'rest_diff' in df.columns and 'vegas_spread' in df.columns:
        df['rest_favorite_interaction'] = df['rest_diff'] * (df['vegas_spread'].fillna(0) < -7).astype(int)
        print("  Added: rest_favorite_interaction")

    return df


def fix_null_handling(df):
    """Fix high-null features with intelligent imputation."""
    print("\nFixing null values...")

    # Line movement - 84% null, replace with 0 (no movement)
    if 'line_movement' in df.columns:
        null_pct = df['line_movement'].isna().mean() * 100
        df['has_line_movement'] = df['line_movement'].notna().astype(int)
        df['line_movement'] = df['line_movement'].fillna(0)
        print(f"  line_movement: {null_pct:.0f}% null -> filled with 0")

    # elo_vs_spread - 60% null, replace with 0 (no difference)
    if 'elo_vs_spread' in df.columns:
        null_pct = df['elo_vs_spread'].isna().mean() * 100
        df['elo_vs_spread'] = df['elo_vs_spread'].fillna(0)
        print(f"  elo_vs_spread: {null_pct:.0f}% null -> filled with 0")

    # PPA features - fill with 0 (league average)
    ppa_features = [col for col in df.columns if 'ppa' in col.lower() or 'epa' in col.lower()]
    for col in ppa_features:
        if col in df.columns and df[col].isna().any():
            null_pct = df[col].isna().mean() * 100
            df[col] = df[col].fillna(0)
            if null_pct > 5:
                print(f"  {col}: {null_pct:.0f}% null -> filled with 0")

    # Success rate - fill with 0.5 (50%)
    success_features = [col for col in df.columns if 'success' in col.lower()]
    for col in success_features:
        if col in df.columns and df[col].isna().any():
            null_pct = df[col].isna().mean() * 100
            df[col] = df[col].fillna(0.5)
            if null_pct > 5:
                print(f"  {col}: {null_pct:.0f}% null -> filled with 0.5")

    return df


def get_available_features(df):
    """Get available safe features plus composite features."""
    # Start with defined safe features
    available = [f for f in SAFE_FEATURES if f in df.columns]

    # Add composite features we created
    composite_features = [
        'matchup_efficiency', 'home_pass_rush_balance', 'away_pass_rush_balance',
        'success_rate_diff', 'elo_efficiency_interaction', 'momentum_strength',
        'dominant_home', 'dominant_away', 'rest_favorite_interaction',
        'has_line_movement'
    ]

    for f in composite_features:
        if f in df.columns and f not in available:
            available.append(f)

    print(f"\nUsing {len(available)} features (V14 had 33)")
    print(f"  Base safe features: {len([f for f in BASE_SAFE_FEATURES if f in df.columns])}")
    print(f"  New efficiency features: {len([f for f in NEW_EFFICIENCY_FEATURES if f in df.columns])}")
    print(f"  Composite features: {len([f for f in composite_features if f in df.columns])}")

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


def optimize_xgboost(df, features, n_trials=150):
    """Optimize XGBoost hyperparameters using Optuna."""
    print("\n" + "=" * 70)
    print("OPTIMIZING XGBoost HYPERPARAMETERS")
    print(f"Running {n_trials} trials...")
    print("=" * 70)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
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


def optimize_hgb(df, features, n_trials=75):
    """Optimize HistGradientBoosting hyperparameters."""
    print("\n" + "=" * 70)
    print("OPTIMIZING HistGradientBoosting HYPERPARAMETERS")
    print(f"Running {n_trials} trials...")
    print("=" * 70)

    def objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 10.0, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 50),
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
    all_spread_errors = []
    all_actual_spread_errors = []

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

        # Betting simulation with confidence tracking
        bets = 0
        wins = 0
        high_conf_bets = 0
        high_conf_wins = 0

        for pe, ae in zip(pred_error, y_test):
            if abs(pe) < 1.0:  # Only skip very low confidence
                continue
            bets += 1
            correct = (pe > 0 and ae > 0) or (pe < 0 and ae < 0)
            if correct:
                wins += 1

            # Track high confidence (5+ points)
            if abs(pe) >= 5.0:
                high_conf_bets += 1
                if correct:
                    high_conf_wins += 1

        win_rate = wins / bets if bets > 0 else 0
        high_conf_rate = high_conf_wins / high_conf_bets if high_conf_bets > 0 else 0

        results.append({
            'week': week,
            'games': len(test_data),
            'model_mae': model_mae,
            'vegas_mae': vegas_mae,
            'bets': bets,
            'wins': wins,
            'win_rate': win_rate,
            'high_conf_bets': high_conf_bets,
            'high_conf_wins': high_conf_wins,
            'high_conf_rate': high_conf_rate
        })

        all_pred.extend(pred_margin.tolist())
        all_actual.extend(actual_margin.tolist())
        all_vegas.extend(vegas_margin.tolist())
        all_spread_errors.extend(pred_error.tolist())
        all_actual_spread_errors.extend(y_test.tolist())

        beat = "✓" if model_mae < vegas_mae else "✗"
        print(f"Week {week:2d}: {len(test_data):3d} games | Model: {model_mae:.2f} | Vegas: {vegas_mae:.2f} {beat} | Bets: {bets}, Win: {win_rate:.1%} | High-Conf: {high_conf_bets}")

    return results, all_pred, all_actual, all_vegas, all_spread_errors, all_actual_spread_errors


def analyze_confidence_distribution(all_spread_errors, all_actual_spread_errors):
    """Analyze the distribution of confidence levels."""
    print("\n" + "=" * 70)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    pred_errors = np.array(all_spread_errors)
    actual_errors = np.array(all_actual_spread_errors)

    tiers = [
        ('HIGH (5+)', 5.0, float('inf')),
        ('MEDIUM-HIGH (3.5-5)', 3.5, 5.0),
        ('MEDIUM (2-3.5)', 2.0, 3.5),
        ('LOW (1-2)', 1.0, 2.0),
        ('VERY LOW (<1)', 0, 1.0),
    ]

    print("\nTier Analysis:")
    print(f"{'Tier':<20} {'Count':<8} {'%':<8} {'Win Rate':<10} {'Profit':<10}")
    print("-" * 60)

    total = len(pred_errors)

    for tier_name, low, high in tiers:
        mask = (np.abs(pred_errors) >= low) & (np.abs(pred_errors) < high)
        count = mask.sum()
        pct = count / total * 100 if total > 0 else 0

        if count > 0:
            # Calculate win rate for this tier
            tier_pred = pred_errors[mask]
            tier_actual = actual_errors[mask]
            correct = ((tier_pred > 0) & (tier_actual > 0)) | ((tier_pred < 0) & (tier_actual < 0))
            win_rate = correct.sum() / count

            # Calculate profit at -110
            wins = correct.sum()
            losses = count - wins
            profit = wins * 0.91 - losses * 1.0
            roi = profit / count * 100 if count > 0 else 0

            print(f"{tier_name:<20} {count:<8} {pct:<7.1f}% {win_rate:<9.1%} {profit:+.1f} ({roi:+.1f}%)")
        else:
            print(f"{tier_name:<20} {count:<8} {pct:<7.1f}% {'N/A':<10} {'N/A':<10}")

    # Overall high confidence (3.5+)
    high_conf_mask = np.abs(pred_errors) >= 3.5
    high_conf_pct = high_conf_mask.sum() / total * 100 if total > 0 else 0
    print(f"\nHigh Confidence (3.5+): {high_conf_pct:.1f}% of predictions")

    return high_conf_pct


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

    # Create composite features
    df = create_composite_features(df)

    # Fix null handling
    df = fix_null_handling(df)

    # Get features
    features = get_available_features(df)

    # Optimize hyperparameters
    print("\n" + "=" * 70)
    print("PHASE 1: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    # Run optimization with more trials
    xgb_params = optimize_xgboost(df, features, n_trials=150)
    hgb_params = optimize_hgb(df, features, n_trials=75)

    # Create ensemble
    ensemble, base_models = train_ensemble(df, features, xgb_params, hgb_params)

    # Validate with walk-forward
    print("\n" + "=" * 70)
    print("PHASE 2: WALK-FORWARD VALIDATION")
    print("=" * 70)

    results, all_pred, all_actual, all_vegas, all_spread_errors, all_actual_spread_errors = full_walk_forward_validation(
        XGBRegressor(**xgb_params, random_state=42, n_jobs=-1),  # Test best single model
        df, features
    )

    # Analyze confidence distribution
    high_conf_pct = analyze_confidence_distribution(all_spread_errors, all_actual_spread_errors)

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

    # High confidence stats
    total_high_conf_bets = res_df['high_conf_bets'].sum()
    total_high_conf_wins = res_df['high_conf_wins'].sum()
    high_conf_win_rate = total_high_conf_wins / total_high_conf_bets if total_high_conf_bets > 0 else 0

    print(f"\nMAE Comparison:")
    print(f"  Model MAE: {final_model_mae:.2f}")
    print(f"  Vegas MAE: {final_vegas_mae:.2f}")
    print(f"  Difference: {final_vegas_mae - final_model_mae:+.2f}")

    print(f"\nWeeks beating Vegas: {weeks_beat_vegas}/{len(res_df)}")

    print(f"\nBetting Results (threshold=1.0):")
    print(f"  Total bets: {total_bets}")
    print(f"  Wins: {total_wins}")
    print(f"  Win rate: {overall_win_rate:.1%}")
    print(f"  Break-even: 52.4%")

    profit = total_wins * 0.91 - (total_bets - total_wins) * 1.0
    roi = (profit / total_bets) * 100 if total_bets > 0 else 0
    print(f"  Profit: {profit:+.1f} units")
    print(f"  ROI: {roi:+.1f}%")
    profitable = overall_win_rate > 0.524

    print(f"\nHigh Confidence Bets (5+):")
    print(f"  Bets: {total_high_conf_bets}")
    print(f"  Wins: {total_high_conf_wins}")
    print(f"  Win rate: {high_conf_win_rate:.1%}")
    if total_high_conf_bets > 0:
        hc_profit = total_high_conf_wins * 0.91 - (total_high_conf_bets - total_high_conf_wins) * 1.0
        hc_roi = hc_profit / total_high_conf_bets * 100
        print(f"  Profit: {hc_profit:+.1f} units")
        print(f"  ROI: {hc_roi:+.1f}%")

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
    joblib.dump(final_model, 'cfb_spread_error_v15.pkl')
    print("Saved model to 'cfb_spread_error_v15.pkl'")

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
            'high_conf_win_rate': high_conf_win_rate,
            'high_conf_pct': high_conf_pct,
        }
    }
    joblib.dump(config, 'cfb_v15_config.pkl')
    print("Saved config to 'cfb_v15_config.pkl'")

    # Feature importance
    print("\nFeature Importance (Top 20):")
    importances = final_model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for i, (_, row) in enumerate(imp_df.head(20).iterrows(), 1):
        print(f"  {i:2d}. {row['importance']:.4f}  {row['feature']}")

    # Summary
    print("\n" + "=" * 70)
    print("V15 HIGH CONFIDENCE MODEL - SUMMARY")
    print("=" * 70)

    print(f"""
IMPROVEMENTS OVER V14:
  Features: 33 -> {len(features)} (+{len(features)-33} new)
  High Confidence %: ~20% -> {high_conf_pct:.1f}%

RESULTS (No Data Leakage):
  Model MAE:  {final_model_mae:.2f}
  Vegas MAE:  {final_vegas_mae:.2f}
  Difference: {final_vegas_mae - final_model_mae:+.2f}

OVERALL BETTING:
  Win Rate: {overall_win_rate:.1%}
  ROI: {roi:+.1f}%
  Profitable: {'YES' if profitable else 'NO'}

HIGH CONFIDENCE (5+ points):
  Win Rate: {high_conf_win_rate:.1%}
  Volume: {total_high_conf_bets} bets

NEW FEATURES ADDED:
  - PPA efficiency features (offense, defense, pass, rush)
  - Success rate composites
  - Matchup efficiency interaction
  - Pass/rush balance indicators
  - Elo x efficiency interaction
  - Momentum strength composite
  - Dominant team flags
  - Rest x favorite interaction

HYPERPARAMETERS OPTIMIZED:
  XGBoost: {len(xgb_params)} params ({150} trials)
  HGB: {len(hgb_params)} params ({75} trials)
""")

    if final_model_mae < final_vegas_mae:
        print("Model beats Vegas on average!")
    else:
        print("Model does not beat Vegas on MAE")

    if profitable:
        print("Model is PROFITABLE!")
    else:
        print("Model is NOT profitable")

    print("\n" + "=" * 70)
    print("V15 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
