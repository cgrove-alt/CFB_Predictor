"""
Train Totals Model for CFB Betting.

Predicts game totals (combined score) using pace, PPA, and rolling stats.
Uses HistGradientBoostingRegressor with proper train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("TRAIN TOTALS MODEL")
print("=" * 60)

print("\nLoading cfb_totals_data.csv...")
df = pd.read_csv('cfb_totals_data.csv')
print(f"Total games loaded: {len(df)}")

# Check columns
print(f"\nAvailable columns: {list(df.columns)}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Sort by season and week for proper rolling calculations
df = df.sort_values(['season', 'week']).reset_index(drop=True)

# Load team pace data for lookups
try:
    team_pace = pd.read_csv('team_pace.csv')
    pace_lookup = dict(zip(team_pace['team'], team_pace['avg_pace']))
    off_ppa_lookup = dict(zip(team_pace['team'], team_pace['avg_off_ppa']))
    def_ppa_lookup = dict(zip(team_pace['team'], team_pace['avg_def_ppa']))
    print(f"Loaded team pace data for {len(team_pace)} teams")
except:
    print("team_pace.csv not found, using defaults")
    pace_lookup = {}
    off_ppa_lookup = {}
    def_ppa_lookup = {}

# Fill missing pace values with league average
avg_pace = df['combined_pace'].mean() if 'combined_pace' in df.columns else 65.0
df['home_avg_pace'] = df['home_avg_pace'].fillna(avg_pace)
df['away_avg_pace'] = df['away_avg_pace'].fillna(avg_pace)
df['combined_pace'] = df['combined_pace'].fillna(avg_pace)

# Fill missing PPA values
avg_off_ppa = df['home_off_ppa'].mean() if 'home_off_ppa' in df.columns else 0.15
avg_def_ppa = df['home_def_ppa'].mean() if 'home_def_ppa' in df.columns else 0.15

df['home_off_ppa'] = df['home_off_ppa'].fillna(avg_off_ppa)
df['away_off_ppa'] = df['away_off_ppa'].fillna(avg_off_ppa)
df['home_def_ppa'] = df['home_def_ppa'].fillna(avg_def_ppa)
df['away_def_ppa'] = df['away_def_ppa'].fillna(avg_def_ppa)

# ============================================================
# CALCULATE ROLLING STATS (Last 5 Games)
# ============================================================
print("\nCalculating rolling stats...")

def calculate_rolling_totals(df, team_col, is_home=True):
    """Calculate rolling average total points for a team's games."""
    rolling_totals = []

    for idx, row in df.iterrows():
        team = row[team_col]
        season = row['season']
        week = row['week']

        # Find previous games for this team (home or away)
        home_games = df[(df['home_team'] == team) &
                        ((df['season'] < season) |
                         ((df['season'] == season) & (df['week'] < week)))]
        away_games = df[(df['away_team'] == team) &
                        ((df['season'] < season) |
                         ((df['season'] == season) & (df['week'] < week)))]

        # Get last 5 games totals
        home_totals = home_games[['season', 'week', 'actual_total']].copy()
        away_totals = away_games[['season', 'week', 'actual_total']].copy()

        all_games = pd.concat([home_totals, away_totals])
        all_games = all_games.sort_values(['season', 'week'], ascending=False).head(5)

        if len(all_games) > 0:
            rolling_totals.append(all_games['actual_total'].mean())
        else:
            rolling_totals.append(np.nan)

    return rolling_totals

# This is slow - let's use a simpler approach with groupby
print("  Computing home team rolling totals...")
home_rolling = []
away_rolling = []

# Create a more efficient rolling calculation
for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(f"    Processing row {idx}/{len(df)}...")

    home_team = row['home_team']
    away_team = row['away_team']
    season = row['season']
    week = row['week'] if pd.notna(row['week']) else 1

    # Find previous games for home team
    mask_home = (
        ((df['home_team'] == home_team) | (df['away_team'] == home_team)) &
        ((df['season'] < season) | ((df['season'] == season) & (df['week'] < week)))
    )
    prev_home = df[mask_home].tail(5)

    if len(prev_home) > 0:
        home_rolling.append(prev_home['actual_total'].mean())
    else:
        home_rolling.append(np.nan)

    # Find previous games for away team
    mask_away = (
        ((df['home_team'] == away_team) | (df['away_team'] == away_team)) &
        ((df['season'] < season) | ((df['season'] == season) & (df['week'] < week)))
    )
    prev_away = df[mask_away].tail(5)

    if len(prev_away) > 0:
        away_rolling.append(prev_away['actual_total'].mean())
    else:
        away_rolling.append(np.nan)

df['home_rolling_total'] = home_rolling
df['away_rolling_total'] = away_rolling

print(f"  Home rolling total coverage: {df['home_rolling_total'].notna().sum()}")
print(f"  Away rolling total coverage: {df['away_rolling_total'].notna().sum()}")

# Combined rolling total expectation
df['combined_rolling_total'] = (df['home_rolling_total'] + df['away_rolling_total']) / 2

# ============================================================
# CALCULATE OFFENSIVE EFFICIENCY
# ============================================================
print("\nCalculating offensive efficiency features...")

# Expected points based on PPA (simplified)
# Higher offensive PPA + lower defensive PPA = more points
df['home_expected_pts'] = (df['home_off_ppa'] - df['away_def_ppa']) * df['home_avg_pace']
df['away_expected_pts'] = (df['away_off_ppa'] - df['home_def_ppa']) * df['away_avg_pace']
df['expected_total'] = df['home_expected_pts'] + df['away_expected_pts']

# Pace differential (faster games = more possessions = more points)
df['pace_sum'] = df['home_avg_pace'] + df['away_avg_pace']

# PPA sum (better offenses = more points)
df['off_ppa_sum'] = df['home_off_ppa'] + df['away_off_ppa']
df['def_ppa_sum'] = df['home_def_ppa'] + df['away_def_ppa']

# Net efficiency
df['net_off_efficiency'] = df['off_ppa_sum'] - df['def_ppa_sum']

# ============================================================
# TRY TO ADD WIND SPEED
# ============================================================
print("\nChecking for wind data...")

try:
    # Load main game data which might have wind
    smart_df = pd.read_csv('cfb_data_smart.csv')
    if 'wind_speed' in smart_df.columns:
        wind_lookup = dict(zip(smart_df['id'], smart_df['wind_speed']))
        df['wind_speed'] = df['game_id'].map(wind_lookup).fillna(0)
        print(f"  Wind data merged: {df['wind_speed'].notna().sum()} games")
    else:
        df['wind_speed'] = 0
        print("  No wind data found, using 0")
except:
    df['wind_speed'] = 0
    print("  Could not load wind data, using 0")

# Wind impact on totals (high wind = lower totals typically)
df['wind_factor'] = np.where(df['wind_speed'] > 15, -3.0,
                    np.where(df['wind_speed'] > 10, -1.5, 0))

# ============================================================
# DEFINE FEATURES AND TARGET
# ============================================================
print("\n" + "=" * 60)
print("PREPARING MODEL DATA")
print("=" * 60)

FEATURE_COLS = [
    'combined_pace',
    'home_avg_pace',
    'away_avg_pace',
    'home_off_ppa',
    'away_off_ppa',
    'home_def_ppa',
    'away_def_ppa',
    'home_rolling_total',
    'away_rolling_total',
    'off_ppa_sum',
    'net_off_efficiency',
    'over_under',  # Vegas line as a feature (they know something)
    'wind_speed',
]

TARGET_COL = 'actual_total'

print(f"\nFeatures: {FEATURE_COLS}")
print(f"Target: {TARGET_COL}")

# Filter to complete cases
all_cols = FEATURE_COLS + [TARGET_COL, 'season']
df_valid = df.dropna(subset=all_cols)
print(f"\nGames with complete data: {len(df_valid)}")

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT")
print("=" * 60)

# Training: 2022-2024
train_mask = df_valid['season'].isin([2022, 2023, 2024])
# Testing: 2025
test_mask = df_valid['season'] == 2025

X_train = df_valid[train_mask][FEATURE_COLS]
y_train = df_valid[train_mask][TARGET_COL]
X_test = df_valid[test_mask][FEATURE_COLS]
y_test = df_valid[test_mask][TARGET_COL]

print(f"Training samples: {len(X_train)} (2022-2024)")
print(f"Testing samples: {len(X_test)} (2025)")

if len(X_test) == 0:
    print("\nWARNING: No 2025 data available, using 2024 as test set")
    train_mask = df_valid['season'].isin([2022, 2023])
    test_mask = df_valid['season'] == 2024

    X_train = df_valid[train_mask][FEATURE_COLS]
    y_train = df_valid[train_mask][TARGET_COL]
    X_test = df_valid[test_mask][FEATURE_COLS]
    y_test = df_valid[test_mask][TARGET_COL]

    print(f"Revised Training: {len(X_train)} (2022-2023)")
    print(f"Revised Testing: {len(X_test)} (2024)")
    test_year = 2024
else:
    test_year = 2025

# ============================================================
# BASELINE: VEGAS LINE
# ============================================================
print("\n" + "=" * 60)
print("BASELINE MODEL (VEGAS LINE)")
print("=" * 60)

# Vegas O/U line is our baseline
vegas_pred = df_valid[test_mask]['over_under']
vegas_mae = mean_absolute_error(y_test, vegas_pred)
print(f"\nVegas O/U MAE: {vegas_mae:.2f} points")

# ============================================================
# TRAIN MODEL
# ============================================================
print("\n" + "=" * 60)
print("TRAINING TOTALS MODEL")
print("=" * 60)

# Use tuned parameters similar to spread model
model = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=3,
    learning_rate=0.05,
    l2_regularization=0.1,
    random_state=42
)

print("\nTraining HistGradientBoostingRegressor...")
model.fit(X_train, y_train)
print("Training complete!")

# ============================================================
# EVALUATE
# ============================================================
print("\n" + "=" * 60)
print(f"MODEL EVALUATION ({test_year})")
print("=" * 60)

# Predict
y_pred = model.predict(X_test)
model_mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel MAE:  {model_mae:.2f} points")
print(f"Vegas MAE:  {vegas_mae:.2f} points")
print(f"Difference: {model_mae - vegas_mae:+.2f} points")

if model_mae < vegas_mae:
    print(f"\n✅ Model beats Vegas by {vegas_mae - model_mae:.2f} points!")
else:
    print(f"\n⚠️ Vegas beats Model by {model_mae - vegas_mae:.2f} points")

# RMSE
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"\nRMSE: {rmse:.2f} points")

# Bias check
mean_error = np.mean(y_test - y_pred)
print(f"Mean Error (Bias): {mean_error:+.2f} points")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

# Get feature importance (using permutation importance approximation)
try:
    # Try sklearn's permutation importance
    from sklearn.inspection import permutation_importance
    perm_imp = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importance = perm_imp.importances_mean
except:
    # Fallback: use simple correlation with predictions
    importance = []
    for col in FEATURE_COLS:
        corr = abs(X_test[col].corr(pd.Series(y_pred)))
        importance.append(corr if pd.notna(corr) else 0)

feat_imp = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': importance
}).sort_values('importance', ascending=False)

print(f"\n{'Feature':<25} {'Importance':>12}")
print("-" * 40)
for _, row in feat_imp.iterrows():
    print(f"{row['feature']:<25} {row['importance']:>12.4f}")

# ============================================================
# OVER/UNDER BETTING ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("OVER/UNDER BETTING ANALYSIS")
print("=" * 60)

test_df = df_valid[test_mask].copy()
test_df['model_pred'] = y_pred
test_df['model_vs_line'] = test_df['model_pred'] - test_df['over_under']

# Model's O/U recommendation
test_df['model_bet'] = np.where(test_df['model_vs_line'] > 2, 'OVER',
                        np.where(test_df['model_vs_line'] < -2, 'UNDER', 'PASS'))

# Actual result
test_df['actual_result'] = np.where(test_df['actual_total'] > test_df['over_under'], 'OVER',
                            np.where(test_df['actual_total'] < test_df['over_under'], 'UNDER', 'PUSH'))

# Calculate betting performance
model_bets = test_df[test_df['model_bet'] != 'PASS']

if len(model_bets) > 0:
    wins = ((model_bets['model_bet'] == 'OVER') & (model_bets['actual_result'] == 'OVER') |
            (model_bets['model_bet'] == 'UNDER') & (model_bets['actual_result'] == 'UNDER')).sum()
    losses = ((model_bets['model_bet'] == 'OVER') & (model_bets['actual_result'] == 'UNDER') |
              (model_bets['model_bet'] == 'UNDER') & (model_bets['actual_result'] == 'OVER')).sum()
    pushes = (model_bets['actual_result'] == 'PUSH').sum()

    total_bets = wins + losses
    if total_bets > 0:
        win_rate = wins / total_bets * 100

        print(f"\nModel Betting Performance (Edge > 2 pts):")
        print(f"  Total Bets: {len(model_bets)}")
        print(f"  Wins: {wins}")
        print(f"  Losses: {losses}")
        print(f"  Pushes: {pushes}")
        print(f"  Win Rate: {win_rate:.1f}%")

        # Breakeven is ~52.4% at -110
        if win_rate > 52.4:
            print(f"\n✅ PROFITABLE: {win_rate:.1f}% > 52.4% breakeven")
        else:
            print(f"\n⚠️ Below breakeven: {win_rate:.1f}% < 52.4%")

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (First 10)")
print("=" * 60)

sample = test_df.head(10)
print(f"\n{'Game':<35} {'Actual':>8} {'Model':>8} {'Vegas':>8} {'Edge':>8}")
print("-" * 70)

for _, row in sample.iterrows():
    game = f"{row['away_team'][:12]} @ {row['home_team'][:12]}"
    actual = row['actual_total']
    pred = row['model_pred']
    vegas = row['over_under']
    edge = row['model_vs_line']
    print(f"{game:<35} {actual:>8.0f} {pred:>8.1f} {vegas:>8.1f} {edge:>+8.1f}")

# ============================================================
# SAVE MODEL
# ============================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(model, 'cfb_totals_model.pkl')
print(f"\nModel saved to 'cfb_totals_model.pkl'")

# Save feature list
with open('totals_features.txt', 'w') as f:
    f.write("Totals Model Features\n")
    f.write("=" * 40 + "\n\n")
    for feat in FEATURE_COLS:
        f.write(f"- {feat}\n")
    f.write(f"\nModel MAE: {model_mae:.2f} points\n")
    f.write(f"Vegas MAE: {vegas_mae:.2f} points\n")

print("Feature list saved to 'totals_features.txt'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
Totals Model Training Complete:

Data:
  - Training: 2022-2024 ({len(X_train)} games)
  - Testing: {test_year} ({len(X_test)} games)

Performance:
  - Model MAE: {model_mae:.2f} points
  - Vegas MAE: {vegas_mae:.2f} points
  - Difference: {model_mae - vegas_mae:+.2f} points

Key Features:
  1. {feat_imp.iloc[0]['feature']}: {feat_imp.iloc[0]['importance']:.3f}
  2. {feat_imp.iloc[1]['feature']}: {feat_imp.iloc[1]['importance']:.3f}
  3. {feat_imp.iloc[2]['feature']}: {feat_imp.iloc[2]['importance']:.3f}

Files:
  - cfb_totals_model.pkl: Trained model
  - totals_features.txt: Feature documentation
""")

print("=" * 60)
print("TOTALS MODEL TRAINING COMPLETE")
print("=" * 60)
