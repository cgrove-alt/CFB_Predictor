"""
Train V12 Spread Error Model for CFB Betting.

KEY INSIGHT: Instead of predicting raw margin (competing with Vegas),
we predict the SPREAD ERROR - how wrong Vegas will be.

Final prediction = Vegas spread + predicted_spread_error

This approach:
1. Uses Vegas as the baseline (they're good at predicting margins)
2. Model learns to find situations where Vegas is wrong
3. Only needs to predict deviations, not absolute margins
4. Better for betting since we care about beating the spread

Target: Average spread error is 0, so we predict deviations from Vegas.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("TRAIN V12 SPREAD ERROR MODEL")
print("Predict How Wrong Vegas Will Be")
print("=" * 70)

print("\nLoading data...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games: {len(df)}")

# Filter to games with Vegas lines
df_vegas = df[df['vegas_spread'].notna()].copy()
print(f"Games with Vegas lines: {len(df_vegas)}")

# ============================================================
# CALCULATE SPREAD ERROR (TARGET)
# ============================================================
print("\n" + "=" * 70)
print("CALCULATING SPREAD ERROR TARGET")
print("=" * 70)

# Spread error = Actual margin - (-Vegas spread)
# Positive = home team beat the spread
# Negative = home team failed to cover
df_vegas['spread_error'] = df_vegas['Margin'] - (-df_vegas['vegas_spread'])

print(f"\nSpread Error Stats:")
print(f"  Mean: {df_vegas['spread_error'].mean():.2f} (should be ~0)")
print(f"  Std: {df_vegas['spread_error'].std():.2f}")
print(f"  Home cover rate: {(df_vegas['spread_error'] > 0).mean()*100:.1f}%")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING FOR SPREAD ERROR")
print("=" * 70)

# Sort by time
df_vegas = df_vegas.sort_values(['season', 'week']).reset_index(drop=True)

# 1. Line movement (indicates sharp money)
df_vegas['line_movement'] = df_vegas['vegas_spread'] - df_vegas['spread_open'].fillna(df_vegas['vegas_spread'])
print("1. Line movement (sharp money indicator)")

# 2. Large spread flag (heavy favorites often cover at lower rate)
df_vegas['large_favorite'] = (df_vegas['vegas_spread'] < -14).astype(int)
df_vegas['large_underdog'] = (df_vegas['vegas_spread'] > 14).astype(int)
print("2. Large spread flags (favorites/underdogs)")

# 3. Momentum - calculate fresh
team_streaks = {}
home_streak = []
away_streak = []

for idx, row in df_vegas.iterrows():
    home = row['home_team']
    away = row['away_team']
    season = row['season']

    # Get current streaks
    h_streak = 0
    a_streak = 0

    if home in team_streaks and team_streaks[home][0] == season:
        h_streak = team_streaks[home][1]
    if away in team_streaks and team_streaks[away][0] == season:
        a_streak = team_streaks[away][1]

    home_streak.append(h_streak)
    away_streak.append(a_streak)

    # Update streaks
    if pd.notna(row['Margin']):
        home_won = row['Margin'] > 0
        if home_won:
            new_h_streak = max(1, h_streak + 1) if h_streak >= 0 else 1
            new_a_streak = min(-1, a_streak - 1) if a_streak <= 0 else -1
        else:
            new_h_streak = min(-1, h_streak - 1) if h_streak <= 0 else -1
            new_a_streak = max(1, a_streak + 1) if a_streak >= 0 else 1

        team_streaks[home] = (season, new_h_streak)
        team_streaks[away] = (season, new_a_streak)

df_vegas['home_streak'] = home_streak
df_vegas['away_streak'] = away_streak
df_vegas['streak_diff'] = df_vegas['home_streak'] - df_vegas['away_streak']
print("3. Team momentum (win/loss streaks)")

# 4. ATS (Against The Spread) performance - historical cover rate
team_ats = {}
home_ats_rate = []
away_ats_rate = []

for idx, row in df_vegas.iterrows():
    home = row['home_team']
    away = row['away_team']

    # Get current ATS rates
    if home in team_ats and len(team_ats[home]) >= 3:
        home_ats_rate.append(np.mean(team_ats[home][-10:]))  # Last 10 games
    else:
        home_ats_rate.append(0.5)

    if away in team_ats and len(team_ats[away]) >= 3:
        away_ats_rate.append(np.mean(team_ats[away][-10:]))
    else:
        away_ats_rate.append(0.5)

    # Update ATS history
    if pd.notna(row['spread_error']):
        home_covered = row['spread_error'] > 0
        if home not in team_ats:
            team_ats[home] = []
        if away not in team_ats:
            team_ats[away] = []
        team_ats[home].append(1 if home_covered else 0)
        team_ats[away].append(0 if home_covered else 1)

df_vegas['home_ats_rate'] = home_ats_rate
df_vegas['away_ats_rate'] = away_ats_rate
df_vegas['ats_diff'] = df_vegas['home_ats_rate'] - df_vegas['away_ats_rate']
print("4. Historical ATS (against the spread) performance")

# 5. Expected close game flag (spreads near 0 are harder to predict)
df_vegas['close_game'] = (abs(df_vegas['vegas_spread']) < 7).astype(int)
print("5. Close game flag (spread < 7)")

# 6. Rest advantage vs spread (tired teams might not cover)
df_vegas['rest_spread_interaction'] = df_vegas['rest_diff'] * abs(df_vegas['vegas_spread']) / 10
print("6. Rest vs spread interaction")

# ============================================================
# DEFINE FEATURES FOR SPREAD ERROR MODEL
# ============================================================
print("\n" + "=" * 70)
print("FEATURE SELECTION FOR SPREAD ERROR MODEL")
print("=" * 70)

# Features that might predict spread error (NOT including vegas spread itself)
SPREAD_ERROR_FEATURES = [
    # Core team strength (without Elo which Vegas already accounts for)
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',
    'home_comp_pass_ppa', 'away_comp_pass_ppa',
    'home_comp_rush_ppa', 'away_comp_rush_ppa',

    # Recent form (momentum)
    'home_streak', 'away_streak', 'streak_diff',

    # ATS performance
    'home_ats_rate', 'away_ats_rate', 'ats_diff',

    # Spread characteristics
    'line_movement',
    'large_favorite', 'large_underdog',
    'close_game',

    # Rest/scheduling
    'rest_diff', 'rest_spread_interaction',

    # HFA (home teams with strong HFA might cover more)
    'home_team_hfa', 'hfa_diff',

    # Recent scoring (does offense/defense match vegas expectations?)
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',
]

# Filter to available
available_features = [f for f in SPREAD_ERROR_FEATURES if f in df_vegas.columns]
print(f"\nUsing {len(available_features)} features for spread error prediction")
for i, f in enumerate(available_features, 1):
    print(f"  {i:2d}. {f}")

# ============================================================
# PREPARE DATA
# ============================================================
print("\n" + "=" * 70)
print("PREPARING DATA")
print("=" * 70)

# Filter valid
df_valid = df_vegas[df_vegas['spread_error'].notna()].copy()
print(f"Games with spread error: {len(df_valid)}")

# Split by year
train_mask = df_valid['season'].isin([2022, 2023, 2024])
test_mask = df_valid['season'] == 2025

X_train = df_valid[train_mask][available_features]
y_train = df_valid[train_mask]['spread_error']
X_test = df_valid[test_mask][available_features]
y_test = df_valid[test_mask]['spread_error']

# Also get Vegas spread for final comparison
vegas_spread_test = df_valid[test_mask]['vegas_spread']
actual_margin_test = df_valid[test_mask]['Margin']

print(f"\nTraining: {len(X_train)} games (2022-2024)")
print(f"Testing: {len(X_test)} games (2025)")

# ============================================================
# TRAIN MODELS
# ============================================================
print("\n" + "=" * 70)
print("TRAINING SPREAD ERROR MODELS")
print("=" * 70)

models = {}

# 1. XGBoost
print("\n1. XGBoost...")
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=4,  # Shallower for spread error (less complex patterns)
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train.fillna(0), y_train)
xgb_error_pred = xgb.predict(X_test.fillna(0))
models['XGBoost'] = xgb

# 2. HGB
print("2. HistGradientBoosting...")
hgb = HistGradientBoostingRegressor(
    max_iter=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)
hgb.fit(X_train, y_train)
hgb_error_pred = hgb.predict(X_test)
models['HGB'] = hgb

# 3. RF
print("3. RandomForest...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train.fillna(0), y_train)
rf_error_pred = rf.predict(X_test.fillna(0))
models['RF'] = rf

# ============================================================
# EVALUATE SPREAD ERROR PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("SPREAD ERROR MODEL EVALUATION")
print("=" * 70)

# MAE of spread error prediction
xgb_error_mae = mean_absolute_error(y_test, xgb_error_pred)
hgb_error_mae = mean_absolute_error(y_test, hgb_error_pred)
rf_error_mae = mean_absolute_error(y_test, rf_error_pred)

print(f"\nSpread Error MAE (lower is better):")
print(f"  XGBoost: {xgb_error_mae:.2f}")
print(f"  HGB: {hgb_error_mae:.2f}")
print(f"  RF: {rf_error_mae:.2f}")

# ============================================================
# CONVERT TO MARGIN PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("FINAL MARGIN PREDICTIONS")
print("=" * 70)

# Final margin = Vegas prediction + spread error adjustment
# Vegas predicts: margin = -vegas_spread
# We adjust: margin = -vegas_spread + spread_error_pred

vegas_margin_pred = -vegas_spread_test
xgb_margin_pred = vegas_margin_pred + xgb_error_pred
hgb_margin_pred = vegas_margin_pred + hgb_error_pred
rf_margin_pred = vegas_margin_pred + rf_error_pred

# Ensemble
ensemble_error_pred = (xgb_error_pred + hgb_error_pred + rf_error_pred) / 3
ensemble_margin_pred = vegas_margin_pred + ensemble_error_pred

# Calculate final MAEs
vegas_mae = mean_absolute_error(actual_margin_test, vegas_margin_pred)
xgb_mae = mean_absolute_error(actual_margin_test, xgb_margin_pred)
hgb_mae = mean_absolute_error(actual_margin_test, hgb_margin_pred)
rf_mae = mean_absolute_error(actual_margin_test, rf_margin_pred)
ensemble_mae = mean_absolute_error(actual_margin_test, ensemble_margin_pred)

print(f"\nFinal Margin MAE (lower is better):")
print(f"  Vegas (baseline):     {vegas_mae:.2f}")
print(f"  Vegas + XGBoost:      {xgb_mae:.2f}")
print(f"  Vegas + HGB:          {hgb_mae:.2f}")
print(f"  Vegas + RF:           {rf_mae:.2f}")
print(f"  Vegas + Ensemble:     {ensemble_mae:.2f}")

# Did we beat Vegas?
best_mae = min(xgb_mae, hgb_mae, rf_mae, ensemble_mae)
if best_mae < vegas_mae:
    print(f"\n  ✓ BEAT VEGAS by {vegas_mae - best_mae:.2f} points!")
else:
    print(f"\n  ✗ Did not beat Vegas (gap: {best_mae - vegas_mae:.2f})")

# ============================================================
# BETTING SIMULATION
# ============================================================
print("\n" + "=" * 70)
print("BETTING SIMULATION")
print("=" * 70)

# Strategy: Bet when model predicts significant spread error
def simulate_betting(spread_error_pred, actual_spread_error, threshold=3.0):
    """
    Simulate betting based on spread error predictions.
    Bet home when predicted error > threshold (home will cover)
    Bet away when predicted error < -threshold (away will cover)
    """
    n_bets = 0
    wins = 0
    units = 0  # Assuming -110 lines, need to win 52.4% to break even

    for pred, actual in zip(spread_error_pred, actual_spread_error):
        if abs(pred) < threshold:
            continue  # No bet

        n_bets += 1
        bet_home = pred > 0

        if bet_home and actual > 0:  # Bet home, home covered
            wins += 1
            units += 0.91  # Win at -110
        elif not bet_home and actual < 0:  # Bet away, away covered
            wins += 1
            units += 0.91
        else:  # Lost
            units -= 1.0

    win_rate = wins / n_bets if n_bets > 0 else 0
    return n_bets, wins, win_rate, units

# Test different thresholds
print("\nBetting simulation (assuming -110 lines):")
print(f"{'Threshold':<12} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Units':<10}")
print("-" * 50)

for threshold in [1.0, 2.0, 3.0, 4.0, 5.0]:
    n_bets, wins, win_rate, units = simulate_betting(
        ensemble_error_pred, y_test, threshold
    )
    print(f"{threshold:<12.1f} {n_bets:<8d} {wins:<8d} {win_rate:<10.1%} {units:<+10.2f}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (for spread error)")
print("=" * 70)

importances = xgb.feature_importances_
importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 features for predicting spread error:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {row['importance']:.4f}  {row['feature']}")

# ============================================================
# SAVE MODELS
# ============================================================
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

# Save spread error model
best_error_model = hgb if hgb_error_mae <= min(xgb_error_mae, rf_error_mae) else (xgb if xgb_error_mae <= rf_error_mae else rf)
joblib.dump(best_error_model, 'cfb_spread_error.pkl')
print("Saved spread error model to 'cfb_spread_error.pkl'")

# Save config
config = {
    'models': models,
    'features': available_features,
    'best_model': 'hgb' if hgb_error_mae <= min(xgb_error_mae, rf_error_mae) else 'xgb',
    'final_mae': best_mae,
    'vegas_mae': vegas_mae
}
joblib.dump(config, 'cfb_spread_error_config.pkl')
print("Saved config to 'cfb_spread_error_config.pkl'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("V12 SPREAD ERROR MODEL - SUMMARY")
print("=" * 70)

print(f"""
APPROACH:
  - Instead of predicting raw margin, predict spread error
  - Final prediction = Vegas spread + spread error adjustment
  - Model learns when Vegas is likely wrong

RESULTS:
  Vegas MAE (baseline):  {vegas_mae:.2f}
  Best Model MAE:        {best_mae:.2f}
  Improvement:           {vegas_mae - best_mae:+.2f} points

SPREAD ERROR PREDICTION:
  XGBoost MAE: {xgb_error_mae:.2f}
  HGB MAE:     {hgb_error_mae:.2f}
  RF MAE:      {rf_error_mae:.2f}

KEY INSIGHT:
  The spread error model identifies situations where Vegas
  systematically over/under values teams.
""")

print("=" * 70)
print("V12 TRAINING COMPLETE!")
print("=" * 70)
