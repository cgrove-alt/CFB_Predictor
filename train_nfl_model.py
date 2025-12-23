"""
Train NFL Spread Prediction Model.

Uses HistGradientBoostingRegressor with rolling EPA stats,
rest days, and net efficiency features.
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
print("TRAIN NFL SPREAD MODEL")
print("=" * 60)

print("\nLoading nfl_data_smart.csv...")
df = pd.read_csv('nfl_data_smart.csv')
print(f"Total games loaded: {len(df)}")

# Filter to completed games with scores
df = df[df['home_score'].notna()].copy()
print(f"Completed games: {len(df)}")

# Sort by season and week for proper rolling calculations
df = df.sort_values(['season', 'week']).reset_index(drop=True)

# ============================================================
# CREATE MARGIN TARGET
# ============================================================
print("\n" + "=" * 60)
print("CREATING TARGET VARIABLE")
print("=" * 60)

df['Margin'] = df['home_margin']  # Already calculated in fetch script
print(f"Margin range: {df['Margin'].min():.0f} to {df['Margin'].max():.0f}")
print(f"Mean margin: {df['Margin'].mean():.2f}")

# ============================================================
# CREATE ROLLING FEATURES (Last 5 Games)
# ============================================================
print("\n" + "=" * 60)
print("CREATING ROLLING FEATURES (Last 5 Games)")
print("=" * 60)

# We need to calculate rolling stats per team across all their games
# First, reshape data to have one row per team per game

def calculate_team_rolling_stats(df, n_games=5):
    """Calculate rolling EPA and other stats for each team."""

    # Create team-game records for both home and away perspectives
    home_records = df[['game_id', 'season', 'week', 'home_team',
                       'home_avg_epa', 'home_def_epa_allowed',
                       'home_success_rate', 'home_score']].copy()
    home_records = home_records.rename(columns={
        'home_team': 'team',
        'home_avg_epa': 'off_epa',
        'home_def_epa_allowed': 'def_epa_allowed',
        'home_success_rate': 'success_rate',
        'home_score': 'points_scored'
    })

    away_records = df[['game_id', 'season', 'week', 'away_team',
                       'away_avg_epa', 'away_def_epa_allowed',
                       'away_success_rate', 'away_score']].copy()
    away_records = away_records.rename(columns={
        'away_team': 'team',
        'away_avg_epa': 'off_epa',
        'away_def_epa_allowed': 'def_epa_allowed',
        'away_success_rate': 'success_rate',
        'away_score': 'points_scored'
    })

    # Combine and sort
    all_games = pd.concat([home_records, away_records], ignore_index=True)
    all_games = all_games.sort_values(['team', 'season', 'week']).reset_index(drop=True)

    # Calculate rolling averages per team (shift by 1 to avoid data leakage)
    rolling_stats = all_games.groupby('team').apply(
        lambda x: pd.DataFrame({
            'game_id': x['game_id'],
            'team': x['team'],
            'rolling_off_epa': x['off_epa'].shift(1).rolling(n_games, min_periods=1).mean(),
            'rolling_def_epa': x['def_epa_allowed'].shift(1).rolling(n_games, min_periods=1).mean(),
            'rolling_success': x['success_rate'].shift(1).rolling(n_games, min_periods=1).mean(),
            'rolling_points': x['points_scored'].shift(1).rolling(n_games, min_periods=1).mean()
        })
    ).reset_index(drop=True)

    return rolling_stats

print("Calculating rolling stats...")
rolling_stats = calculate_team_rolling_stats(df, n_games=5)
print(f"Rolling stats records: {len(rolling_stats)}")

# Merge back - home team
home_rolling = rolling_stats.rename(columns={
    'team': 'home_team',
    'rolling_off_epa': 'home_rolling_off_epa',
    'rolling_def_epa': 'home_rolling_def_epa',
    'rolling_success': 'home_rolling_success',
    'rolling_points': 'home_rolling_points'
})
home_rolling = home_rolling[['game_id', 'home_team', 'home_rolling_off_epa',
                              'home_rolling_def_epa', 'home_rolling_success', 'home_rolling_points']]

# Merge back - away team
away_rolling = rolling_stats.rename(columns={
    'team': 'away_team',
    'rolling_off_epa': 'away_rolling_off_epa',
    'rolling_def_epa': 'away_rolling_def_epa',
    'rolling_success': 'away_rolling_success',
    'rolling_points': 'away_rolling_points'
})
away_rolling = away_rolling[['game_id', 'away_team', 'away_rolling_off_epa',
                              'away_rolling_def_epa', 'away_rolling_success', 'away_rolling_points']]

# Merge into main dataframe
df = df.merge(home_rolling, on=['game_id', 'home_team'], how='left')
df = df.merge(away_rolling, on=['game_id', 'away_team'], how='left')

print(f"Games with home rolling EPA: {df['home_rolling_off_epa'].notna().sum()}")
print(f"Games with away rolling EPA: {df['away_rolling_off_epa'].notna().sum()}")

# ============================================================
# CREATE DERIVED FEATURES
# ============================================================
print("\n" + "=" * 60)
print("CREATING DERIVED FEATURES")
print("=" * 60)

# Net EPA (Home Offense vs Away Defense)
df['net_off_epa'] = df['home_rolling_off_epa'] - df['away_rolling_def_epa']
df['net_def_epa'] = df['away_rolling_off_epa'] - df['home_rolling_def_epa']

# EPA differential
df['epa_diff'] = df['home_rolling_off_epa'] - df['away_rolling_off_epa']
df['def_epa_diff'] = df['home_rolling_def_epa'] - df['away_rolling_def_epa']

# Success rate differential
df['success_diff'] = df['home_rolling_success'] - df['away_rolling_success']

# Rest advantage
df['rest_advantage'] = df['home_rest'] - df['away_rest']

# Points differential
df['points_diff'] = df['home_rolling_points'] - df['away_rolling_points']

print("Features created:")
print("  - net_off_epa (Home Off vs Away Def)")
print("  - net_def_epa (Away Off vs Home Def)")
print("  - epa_diff (EPA differential)")
print("  - def_epa_diff (Defensive EPA diff)")
print("  - success_diff (Success rate diff)")
print("  - rest_advantage (Rest days diff)")
print("  - points_diff (Scoring avg diff)")

# ============================================================
# DEFINE FEATURE SET
# ============================================================
print("\n" + "=" * 60)
print("FEATURE CONFIGURATION")
print("=" * 60)

FEATURE_COLS = [
    # Rolling EPA
    'home_rolling_off_epa', 'away_rolling_off_epa',
    'home_rolling_def_epa', 'away_rolling_def_epa',

    # Derived features
    'net_off_epa', 'net_def_epa',
    'epa_diff', 'def_epa_diff',
    'success_diff',

    # Rest
    'home_rest', 'away_rest', 'rest_advantage',

    # Vegas line (for comparison)
    'spread_line'
]

# Filter to features that exist
available_features = [f for f in FEATURE_COLS if f in df.columns]
print(f"\nFeatures available: {len(available_features)}")
for f in available_features:
    print(f"  - {f}")

# ============================================================
# PREPARE DATA
# ============================================================
print("\n" + "=" * 60)
print("PREPARING DATA")
print("=" * 60)

# Filter to complete cases
df_valid = df.dropna(subset=available_features + ['Margin'])
print(f"Games with complete data: {len(df_valid)}")

# Train/Test split by season
train_mask = df_valid['season'].isin([2022, 2023, 2024])
test_mask = df_valid['season'] == 2025

X_train = df_valid[train_mask][available_features]
y_train = df_valid[train_mask]['Margin']
X_test = df_valid[test_mask][available_features]
y_test = df_valid[test_mask]['Margin']

print(f"\nTraining samples: {len(X_train)} (2022-2024)")
print(f"Testing samples: {len(X_test)} (2025)")

# If no 2025 data, use 2024 as test
if len(X_test) == 0:
    print("\nNo 2025 data yet. Using 2024 as test set...")
    train_mask = df_valid['season'].isin([2022, 2023])
    test_mask = df_valid['season'] == 2024

    X_train = df_valid[train_mask][available_features]
    y_train = df_valid[train_mask]['Margin']
    X_test = df_valid[test_mask][available_features]
    y_test = df_valid[test_mask]['Margin']

    print(f"Revised Training: {len(X_train)} (2022-2023)")
    print(f"Revised Testing: {len(X_test)} (2024)")
    test_year = 2024
else:
    test_year = 2025

# ============================================================
# TRAIN MODEL
# ============================================================
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

model = HistGradientBoostingRegressor(
    max_iter=200,
    max_depth=4,
    learning_rate=0.05,
    l2_regularization=0.1,
    min_samples_leaf=10,
    random_state=42
)

print("\nModel: HistGradientBoostingRegressor")
print("  - max_iter: 200")
print("  - max_depth: 4")
print("  - learning_rate: 0.05")
print("  - l2_regularization: 0.1")

print("\nFitting model...")
model.fit(X_train, y_train)
print("Training complete!")

# ============================================================
# EVALUATE MODEL
# ============================================================
print("\n" + "=" * 60)
print(f"MODEL EVALUATION ({test_year})")
print("=" * 60)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Metrics
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"\n{'Set':<15} {'Games':>8} {'MAE':>10}")
print("-" * 35)
print(f"{'Training':<15} {len(X_train):>8} {train_mae:>10.2f}")
print(f"{'Testing':<15} {len(X_test):>8} {test_mae:>10.2f}")
print("-" * 35)

# Compare to Vegas
if 'spread_line' in df_valid.columns:
    vegas_pred = -df_valid[test_mask]['spread_line']  # Negate because spread is from home perspective
    vegas_mae = mean_absolute_error(y_test, vegas_pred)
    print(f"{'Vegas Line':<15} {len(X_test):>8} {vegas_mae:>10.2f}")

# Target check
print("\n" + "=" * 60)
if test_mae < 10.0:
    print(f"TARGET ACHIEVED: MAE = {test_mae:.2f} (< 10.0)")
else:
    print(f"TARGET NOT MET: MAE = {test_mae:.2f} (target < 10.0)")
print("=" * 60)

# ============================================================
# ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

errors = y_test - test_pred
print(f"\nMean Error (Bias): {errors.mean():+.2f}")
print(f"Std Error: {errors.std():.2f}")
print(f"Max Error: {errors.abs().max():.2f}")
print(f"Median Abs Error: {errors.abs().median():.2f}")

# Error distribution
print(f"\nError Distribution:")
print(f"  Within 3 pts: {(errors.abs() <= 3).mean()*100:.1f}%")
print(f"  Within 7 pts: {(errors.abs() <= 7).mean()*100:.1f}%")
print(f"  Within 10 pts: {(errors.abs() <= 10).mean()*100:.1f}%")
print(f"  Within 14 pts: {(errors.abs() <= 14).mean()*100:.1f}%")

# ============================================================
# SAVE MODEL
# ============================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(model, 'nfl_predictor.pkl')
print("\nModel saved to 'nfl_predictor.pkl'")

# Save feature list
with open('nfl_model_features.txt', 'w') as f:
    f.write("NFL Spread Model Features\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Training: 2022-2023 ({len(X_train)} games)\n")
    f.write(f"Testing: {test_year} ({len(X_test)} games)\n")
    f.write(f"MAE: {test_mae:.2f} points\n\n")
    f.write("Features:\n")
    for feat in available_features:
        f.write(f"  - {feat}\n")

print("Feature list saved to 'nfl_model_features.txt'")

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (Last 10 Games)")
print("=" * 60)

test_df = df_valid[test_mask].copy()
test_df['Predicted'] = test_pred

print(f"\n{'Game':<25} {'Actual':>8} {'Pred':>8} {'Vegas':>8} {'Error':>8}")
print("-" * 60)

for _, row in test_df.tail(10).iterrows():
    game = f"{row['away_team']} @ {row['home_team']}"
    actual = row['Margin']
    pred = row['Predicted']
    vegas = -row['spread_line'] if pd.notna(row['spread_line']) else 0
    error = actual - pred
    print(f"{game:<25} {actual:>+8.0f} {pred:>+8.1f} {vegas:>+8.1f} {error:>+8.1f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
NFL Spread Model Training Complete:

Data:
  - Training: 2022-2023 ({len(X_train)} games)
  - Testing: {test_year} ({len(X_test)} games)

Model Performance:
  - Training MAE: {train_mae:.2f} points
  - Testing MAE: {test_mae:.2f} points
  - Vegas MAE: {vegas_mae:.2f} points (benchmark)

Model vs Vegas: {test_mae - vegas_mae:+.2f} points

Features Used: {len(available_features)}
  - Rolling EPA (Offense & Defense)
  - Net EPA matchups
  - Rest advantage
  - Vegas spread line

Files Created:
  - nfl_predictor.pkl: Trained model
  - nfl_model_features.txt: Feature documentation
""")

print("=" * 60)
print("NFL MODEL TRAINING COMPLETE")
print("=" * 60)
