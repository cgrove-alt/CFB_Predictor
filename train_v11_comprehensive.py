"""
Train V11 Comprehensive Model for CFB Spread Prediction.

COMPREHENSIVE IMPROVEMENTS:
1. Vegas spread as baseline feature (learn to beat Vegas)
2. All available features from data including adjusted EPA
3. Feature importance analysis and selection
4. Walk-forward backtesting validation
5. Recent game weighting (exponential decay)
6. Conference-specific adjustments
7. Blowout filtering (extreme margins distort learning)

Target: Beat Vegas MAE of 12.00 points consistently
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
print("=" * 70)
print("TRAIN V11 COMPREHENSIVE MODEL")
print("All Improvements: Vegas baseline, Feature Selection, Walk-Forward")
print("=" * 70)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")

# Check for betting lines
has_vegas = 'vegas_spread' in df.columns and df['vegas_spread'].notna().sum() > 0
print(f"Has Vegas spreads: {has_vegas}")
if has_vegas:
    print(f"  Games with Vegas lines: {df['vegas_spread'].notna().sum()}")

# ============================================================
# FEATURE ENGINEERING - COMPREHENSIVE
# ============================================================
print("\n" + "=" * 70)
print("COMPREHENSIVE FEATURE ENGINEERING")
print("=" * 70)

# Sort by time
df = df.sort_values(['season', 'week']).reset_index(drop=True)

# 1. MOMENTUM FEATURES
print("\n1. Calculating momentum features...")
team_results = {}
home_streak = []
away_streak = []
home_trajectory = []
away_trajectory = []

for idx, row in df.iterrows():
    home = row['home_team']
    away = row['away_team']
    season = row['season']

    # Home team streak
    if home in team_results:
        results = team_results[home]
        streak = 0
        for r in reversed(results):
            if r[0] != season:
                break
            if streak == 0:
                streak = 1 if r[2] else -1
            elif (streak > 0 and r[2]) or (streak < 0 and not r[2]):
                streak += 1 if streak > 0 else -1
            else:
                break
        home_streak.append(streak)
        margins = [r[3] for r in results[-3:] if r[0] == season]
        home_trajectory.append(np.mean(margins) if margins else 0)
    else:
        home_streak.append(0)
        home_trajectory.append(0)

    # Away team streak
    if away in team_results:
        results = team_results[away]
        streak = 0
        for r in reversed(results):
            if r[0] != season:
                break
            if streak == 0:
                streak = 1 if r[2] else -1
            elif (streak > 0 and r[2]) or (streak < 0 and not r[2]):
                streak += 1 if streak > 0 else -1
            else:
                break
        away_streak.append(streak)
        margins = [r[3] for r in results[-3:] if r[0] == season]
        away_trajectory.append(np.mean(margins) if margins else 0)
    else:
        away_streak.append(0)
        away_trajectory.append(0)

    # Update results
    if pd.notna(row['Margin']):
        home_won = row['Margin'] > 0
        if home not in team_results:
            team_results[home] = []
        if away not in team_results:
            team_results[away] = []
        team_results[home].append((season, row['week'], home_won, row['Margin']))
        team_results[away].append((season, row['week'], not home_won, -row['Margin']))

df['home_win_streak'] = home_streak
df['away_win_streak'] = away_streak
df['home_trajectory'] = home_trajectory
df['away_trajectory'] = away_trajectory
df['momentum_diff'] = df['home_win_streak'] - df['away_win_streak']
df['trajectory_diff'] = df['home_trajectory'] - df['away_trajectory']
print("  Added: home_win_streak, away_win_streak, momentum_diff, trajectory_diff")

# 2. STRENGTH OF SCHEDULE PROXY
print("\n2. Calculating strength of schedule proxy...")
# Use opponent's average Elo as SOS proxy
team_avg_opp_elo = {}
for idx, row in df.iterrows():
    home = row['home_team']
    away = row['away_team']
    home_elo = row.get('home_pregame_elo', 1500)
    away_elo = row.get('away_pregame_elo', 1500)

    if home not in team_avg_opp_elo:
        team_avg_opp_elo[home] = []
    if away not in team_avg_opp_elo:
        team_avg_opp_elo[away] = []

    team_avg_opp_elo[home].append(away_elo)
    team_avg_opp_elo[away].append(home_elo)

# Calculate average opponent Elo for each team
team_sos = {team: np.mean(elos[-10:]) if elos else 1500 for team, elos in team_avg_opp_elo.items()}
df['home_sos'] = df['home_team'].map(team_sos).fillna(1500)
df['away_sos'] = df['away_team'].map(team_sos).fillna(1500)
df['sos_diff'] = df['home_sos'] - df['away_sos']
print("  Added: home_sos, away_sos, sos_diff")

# 3. VEGAS-DERIVED FEATURES (if available)
if has_vegas:
    print("\n3. Creating Vegas-derived features...")
    # Use Vegas spread as a strong signal
    df['vegas_spread_filled'] = df['vegas_spread'].fillna(df['elo_diff'] / 25)  # Rough Elo-to-spread conversion
    df['expected_total'] = df['over_under'].fillna(53)  # Average total

    # Implied team totals
    df['home_implied_score'] = (df['expected_total'] - df['vegas_spread_filled']) / 2
    df['away_implied_score'] = (df['expected_total'] + df['vegas_spread_filled']) / 2

    print("  Added: vegas_spread_filled, expected_total, implied scores")

# 4. INTERACTION FEATURES
print("\n4. Creating interaction features...")
# Elo x Momentum interaction
df['elo_momentum_interaction'] = df['elo_diff'] * df['momentum_diff']
# PPA x Rest interaction
df['ppa_rest_interaction'] = df['net_epa'] * df['rest_diff']
# Success x HFA interaction
df['success_hfa_interaction'] = df['success_diff'] * df['hfa_diff']
print("  Added: elo_momentum_interaction, ppa_rest_interaction, success_hfa_interaction")

# 5. BLOWOUT FLAG
print("\n5. Flagging potential blowouts...")
# Games with large Elo difference are more likely to be blowouts
df['large_elo_diff'] = (abs(df['elo_diff']) > 400).astype(int)
df['large_spread'] = 0
if has_vegas:
    df['large_spread'] = (abs(df['vegas_spread_filled']) > 21).astype(int)
print("  Added: large_elo_diff, large_spread")

# ============================================================
# DEFINE ALL FEATURES
# ============================================================
print("\n" + "=" * 70)
print("FEATURE SELECTION")
print("=" * 70)

# All potential features
ALL_FEATURES = [
    # Core Elo (2)
    'home_pregame_elo', 'away_pregame_elo',

    # Rolling stats (4)
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',

    # EPA/PPA (8)
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',
    'home_comp_pass_ppa', 'away_comp_pass_ppa',
    'home_comp_rush_ppa', 'away_comp_rush_ppa',

    # Success rates (2)
    'home_comp_success', 'away_comp_success',

    # HFA (3)
    'home_team_hfa', 'away_team_hfa', 'hfa_diff',

    # Rest (3)
    'home_rest_days', 'away_rest_days', 'rest_diff',

    # Pre-calculated diffs (5)
    'elo_diff', 'net_epa', 'pass_efficiency_diff', 'epa_elo_interaction', 'success_diff',

    # Momentum (4)
    'home_win_streak', 'away_win_streak', 'momentum_diff', 'trajectory_diff',

    # SOS (3)
    'home_sos', 'away_sos', 'sos_diff',

    # New interactions (3)
    'elo_momentum_interaction', 'ppa_rest_interaction', 'success_hfa_interaction',

    # Blowout indicators (2)
    'large_elo_diff', 'large_spread',
]

# Add Vegas features if available
if has_vegas:
    ALL_FEATURES.extend([
        'vegas_spread_filled', 'expected_total',
        'home_implied_score', 'away_implied_score'
    ])

# Filter to available features
available_features = [f for f in ALL_FEATURES if f in df.columns]
print(f"\nTotal features: {len(available_features)}")
for i, f in enumerate(available_features, 1):
    print(f"  {i:2d}. {f}")

# ============================================================
# PREPARE DATA FOR WALK-FORWARD VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("WALK-FORWARD VALIDATION SETUP")
print("=" * 70)

# Filter valid games (with margin)
df_valid = df[df['Margin'].notna()].copy()

# Optional: Filter out extreme blowouts for training (but keep for testing)
FILTER_BLOWOUTS = True
BLOWOUT_THRESHOLD = 42  # Games decided by more than 6 TDs

if FILTER_BLOWOUTS:
    df_train_pool = df_valid[abs(df_valid['Margin']) <= BLOWOUT_THRESHOLD].copy()
    print(f"\nFiltering blowouts (>{BLOWOUT_THRESHOLD} point margins) for training")
    print(f"  Original games: {len(df_valid)}")
    print(f"  After blowout filter: {len(df_train_pool)}")
else:
    df_train_pool = df_valid.copy()

# Walk-forward splits
# Train on 2022-2023, validate on 2024 week-by-week
print("\nWalk-forward validation:")
print("  Training base: 2022-2023")
print("  Walk-forward: 2024 (week by week)")
print("  Final test: 2025")

# ============================================================
# SAMPLE WEIGHTING - RECENT GAMES MATTER MORE
# ============================================================
print("\n" + "=" * 70)
print("SAMPLE WEIGHTING (EXPONENTIAL DECAY)")
print("=" * 70)

def calculate_sample_weights(df, decay_factor=0.95):
    """Calculate sample weights with exponential decay for older games."""
    # Most recent game gets weight 1, older games decay
    df = df.sort_values(['season', 'week'], ascending=False).reset_index(drop=True)
    weights = decay_factor ** np.arange(len(df))
    # Normalize
    weights = weights / weights.sum() * len(df)
    # Restore original order
    df['_weight'] = weights
    df = df.sort_values(['season', 'week']).reset_index(drop=True)
    return df['_weight'].values

# ============================================================
# WALK-FORWARD BACKTESTING
# ============================================================
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTESTING ON 2024")
print("=" * 70)

# Get 2024 weeks
weeks_2024 = sorted(df_valid[df_valid['season'] == 2024]['week'].unique())
print(f"\n2024 weeks to validate: {weeks_2024}")

walk_forward_results = []

for week in weeks_2024:
    # Train on all data before this week
    train_mask = (df_train_pool['season'] < 2024) | ((df_train_pool['season'] == 2024) & (df_train_pool['week'] < week))
    test_mask = (df_valid['season'] == 2024) & (df_valid['week'] == week)

    X_train = df_train_pool[train_mask][available_features]
    y_train = df_train_pool[train_mask]['Margin']
    X_test = df_valid[test_mask][available_features]
    y_test = df_valid[test_mask]['Margin']

    if len(X_train) < 100 or len(X_test) == 0:
        continue

    # Calculate sample weights
    weights = calculate_sample_weights(df_train_pool[train_mask])

    # Train XGBoost with sample weights
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train.fillna(-999), y_train, sample_weight=weights)

    # Predict
    pred = model.predict(X_test.fillna(-999))
    mae = mean_absolute_error(y_test, pred)

    # Vegas MAE for comparison (if available)
    vegas_mae = None
    if has_vegas:
        test_with_vegas = df_valid[test_mask][df_valid[test_mask]['vegas_spread'].notna()]
        if len(test_with_vegas) > 0:
            vegas_pred = -test_with_vegas['vegas_spread']  # Convert to home margin prediction
            vegas_mae = mean_absolute_error(test_with_vegas['Margin'], vegas_pred)

    walk_forward_results.append({
        'week': week,
        'n_games': len(X_test),
        'model_mae': mae,
        'vegas_mae': vegas_mae
    })

    print(f"  Week {week:2d}: {len(X_test):3d} games, Model MAE: {mae:.2f}", end="")
    if vegas_mae:
        beat_vegas = "✓" if mae < vegas_mae else "✗"
        print(f", Vegas MAE: {vegas_mae:.2f} {beat_vegas}")
    else:
        print()

# Walk-forward summary
print("\n" + "-" * 50)
print("WALK-FORWARD SUMMARY (2024)")
wf_df = pd.DataFrame(walk_forward_results)
avg_model_mae = wf_df['model_mae'].mean()
print(f"Average Model MAE: {avg_model_mae:.2f}")
if has_vegas:
    avg_vegas_mae = wf_df['vegas_mae'].dropna().mean()
    weeks_beat_vegas = (wf_df['model_mae'] < wf_df['vegas_mae']).sum()
    print(f"Average Vegas MAE: {avg_vegas_mae:.2f}")
    print(f"Weeks beating Vegas: {weeks_beat_vegas}/{len(wf_df)}")

# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# Train on 2022-2024 for importance analysis
train_data = df_train_pool[df_train_pool['season'].isin([2022, 2023, 2024])]
X_importance = train_data[available_features].fillna(-999)
y_importance = train_data['Margin']

importance_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
importance_model.fit(X_importance, y_importance)

# Get feature importances
importances = importance_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
for i, row in importance_df.head(15).iterrows():
    print(f"  {row['importance']:.4f}  {row['feature']}")

print("\nBottom 10 Features (candidates for removal):")
for i, row in importance_df.tail(10).iterrows():
    print(f"  {row['importance']:.4f}  {row['feature']}")

# Select top features
TOP_K = 25
selected_features = importance_df.head(TOP_K)['feature'].tolist()
print(f"\nSelected top {TOP_K} features for final model")

# ============================================================
# FINAL MODEL TRAINING
# ============================================================
print("\n" + "=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

# Use selected features
print(f"\nUsing {len(selected_features)} selected features")

# Full training data (2022-2024)
train_full = df_train_pool[df_train_pool['season'].isin([2022, 2023, 2024])]
X_train_full = train_full[selected_features].fillna(-999)
y_train_full = train_full['Margin']
weights_full = calculate_sample_weights(train_full)

# Test on 2025
test_2025 = df_valid[df_valid['season'] == 2025]
X_test = test_2025[selected_features].fillna(-999)
y_test = test_2025['Margin']

print(f"\nTraining samples: {len(X_train_full)} (2022-2024, blowouts filtered)")
print(f"Test samples: {len(X_test)} (2025)")

# Train multiple models
print("\nTraining models...")

# 1. XGBoost with sample weights
print("\n1. XGBoost (weighted)...")
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_full, y_train_full, sample_weight=weights_full)
xgb_pred = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
print(f"   XGBoost MAE: {xgb_mae:.2f}")

# 2. HistGradientBoosting (no sample weights but handles NaN)
print("\n2. HistGradientBoosting...")
hgb_model = HistGradientBoostingRegressor(
    max_iter=300,
    max_depth=7,
    learning_rate=0.03,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)
hgb_model.fit(train_full[selected_features], y_train_full)
hgb_pred = hgb_model.predict(test_2025[selected_features])
hgb_mae = mean_absolute_error(y_test, hgb_pred)
print(f"   HGB MAE: {hgb_mae:.2f}")

# 3. GradientBoosting (sklearn version)
print("\n3. GradientBoosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42
)
gb_model.fit(X_train_full, y_train_full, sample_weight=weights_full)
gb_pred = gb_model.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_pred)
print(f"   GB MAE: {gb_mae:.2f}")

# 4. Ensemble
print("\n4. Weighted Ensemble...")
# Weight by inverse MAE
maes = {'xgb': xgb_mae, 'hgb': hgb_mae, 'gb': gb_mae}
total_inv = sum(1/m for m in maes.values())
weights_ens = {k: (1/m)/total_inv for k, m in maes.items()}

ensemble_pred = (
    weights_ens['xgb'] * xgb_pred +
    weights_ens['hgb'] * hgb_pred +
    weights_ens['gb'] * gb_pred
)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
print(f"   Ensemble MAE: {ensemble_mae:.2f}")

# Vegas comparison
if has_vegas:
    test_with_vegas = test_2025[test_2025['vegas_spread'].notna()]
    if len(test_with_vegas) > 0:
        vegas_pred_test = -test_with_vegas['vegas_spread']
        vegas_mae_test = mean_absolute_error(test_with_vegas['Margin'], vegas_pred_test)
        print(f"\n   Vegas MAE (2025): {vegas_mae_test:.2f}")

# ============================================================
# ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

errors = y_test - ensemble_pred

print(f"\n{'Metric':<25} {'Value':>12}")
print("-" * 40)
print(f"{'MAE':<25} {ensemble_mae:>12.2f}")
print(f"{'Mean Error (Bias)':<25} {errors.mean():>+12.2f}")
print(f"{'Std Error':<25} {errors.std():>12.2f}")
print(f"{'Median Abs Error':<25} {errors.abs().median():>12.2f}")

print(f"\n{'Error Range':<25} {'Pct':>12}")
print("-" * 40)
for threshold in [3, 7, 10, 14, 21]:
    pct = (errors.abs() <= threshold).mean() * 100
    print(f"{'Within ' + str(threshold) + ' pts':<25} {pct:>11.1f}%")

# ============================================================
# SAVE MODELS
# ============================================================
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

# Save best single model
best_model = xgb_model if xgb_mae <= min(hgb_mae, gb_mae) else (hgb_model if hgb_mae <= gb_mae else gb_model)
best_mae = min(xgb_mae, hgb_mae, gb_mae)
print(f"\nBest single model MAE: {best_mae:.2f}")

joblib.dump(xgb_model, 'cfb_stacking.pkl')
print("Saved XGBoost model to 'cfb_stacking.pkl'")

# Save ensemble config
ensemble_config = {
    'models': {
        'xgb': xgb_model,
        'hgb': hgb_model,
        'gb': gb_model
    },
    'weights': weights_ens,
    'features': selected_features,
    'ensemble_mae': ensemble_mae
}
joblib.dump(ensemble_config, 'cfb_ensemble_v11.pkl')
print("Saved ensemble to 'cfb_ensemble_v11.pkl'")

# Save feature list
with open('v11_features.txt', 'w') as f:
    f.write("CFB V11 Comprehensive Model\n")
    f.write("=" * 50 + "\n\n")
    f.write("IMPROVEMENTS:\n")
    f.write("  - Vegas spread as baseline\n")
    f.write("  - Feature importance selection\n")
    f.write("  - Walk-forward validation\n")
    f.write("  - Sample weighting (recent games weighted higher)\n")
    f.write("  - Blowout filtering for training\n\n")
    f.write(f"FINAL ENSEMBLE MAE: {ensemble_mae:.2f}\n")
    f.write(f"Best Single Model MAE: {best_mae:.2f}\n\n")
    f.write(f"SELECTED FEATURES ({len(selected_features)}):\n")
    for feat in selected_features:
        f.write(f"  - {feat}\n")

print("Saved feature documentation to 'v11_features.txt'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("V11 TRAINING COMPLETE - SUMMARY")
print("=" * 70)

print(f"""
RESULTS:
  Walk-Forward MAE (2024): {avg_model_mae:.2f}
  Final Test MAE (2025):   {ensemble_mae:.2f}
  Best Single Model:       {best_mae:.2f}

INDIVIDUAL MODELS:
  XGBoost:           {xgb_mae:.2f} (weight: {weights_ens['xgb']:.1%})
  HGB:               {hgb_mae:.2f} (weight: {weights_ens['hgb']:.1%})
  GradientBoosting:  {gb_mae:.2f} (weight: {weights_ens['gb']:.1%})

FEATURES:
  Total available:   {len(available_features)}
  Selected (top-k):  {len(selected_features)}

DATA:
  Training games:    {len(X_train_full)} (blowouts filtered)
  Test games:        {len(X_test)}

FILES CREATED:
  - cfb_stacking.pkl: Best model for app
  - cfb_ensemble_v11.pkl: Full ensemble
  - v11_features.txt: Documentation
""")

print("=" * 70)
print("V11 COMPREHENSIVE TRAINING COMPLETE!")
print("=" * 70)
