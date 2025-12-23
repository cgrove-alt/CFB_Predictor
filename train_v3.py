"""
Train CFB Spread Model - Version 6 (The Brain)

Uses 16 features from cfb_data_smart.csv to predict game margins.
Trains HistGradientBoostingRegressor and saves to cfb_smart_model.pkl.
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
print("VERSION 6 - CFB SPREAD MODEL TRAINING")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\nEngineering Version 6 features...")

# Create net_epa from competitive EPA stats
df['offense_epa'] = df['home_comp_off_ppa']
df['defense_epa'] = df['home_comp_def_ppa']
df['net_epa'] = df['home_comp_off_ppa'] - df['away_comp_def_ppa']

# Use rest_days columns (rename for consistency)
df['home_rest'] = df['home_rest_days']
df['away_rest'] = df['away_rest_days']

print("  - net_epa calculated")
print("  - home_rest/away_rest mapped")

# ============================================================
# DEFINE 16 FEATURES
# ============================================================
FEATURE_COLS = [
    # Elo ratings (power rankings)
    'home_pregame_elo', 'away_pregame_elo',

    # Rolling stats (recent form - points)
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',

    # EPA/PPA efficiency
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',

    # Net EPA (schematic mismatch)
    'net_epa',

    # Home Field Advantage
    'home_team_hfa', 'away_team_hfa',

    # Rest advantage
    'home_rest', 'away_rest', 'rest_advantage'
]

print(f"\nVersion 6 Features ({len(FEATURE_COLS)} total):")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {f}")

# ============================================================
# PREPARE DATA
# ============================================================
print("\n" + "=" * 60)
print("PREPARING DATA")
print("=" * 60)

# Filter to complete cases
df_valid = df.dropna(subset=FEATURE_COLS + ['Margin'])
print(f"Games with complete data: {len(df_valid)}")

# Train/Test split by season (2022-2023 train, 2024 test)
train_mask = df_valid['season'].isin([2022, 2023])
test_mask = df_valid['season'] == 2024

X_train = df_valid[train_mask][FEATURE_COLS]
y_train = df_valid[train_mask]['Margin']
X_test = df_valid[test_mask][FEATURE_COLS]
y_test = df_valid[test_mask]['Margin']

print(f"\nTraining samples: {len(X_train)} (2022-2023)")
print(f"Testing samples: {len(X_test)} (2024)")

# ============================================================
# TRAIN MODEL
# ============================================================
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

model = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=6,
    learning_rate=0.1,
    l2_regularization=0.1,
    min_samples_leaf=10,
    random_state=42
)

print("\nModel: HistGradientBoostingRegressor")
print("  - max_iter: 100")
print("  - max_depth: 6")
print("  - learning_rate: 0.1")

print("\nFitting model...")
model.fit(X_train, y_train)
print("Training complete!")

# ============================================================
# EVALUATE MODEL
# ============================================================
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"\n{'Set':<15} {'Games':>8} {'MAE':>10}")
print("-" * 35)
print(f"{'Training':<15} {len(X_train):>8} {train_mae:>10.2f}")
print(f"{'Testing':<15} {len(X_test):>8} {test_mae:>10.2f}")
print("-" * 35)

# ============================================================
# SAVE MODEL
# ============================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(model, 'cfb_smart_model.pkl')
print("\nModel saved to 'cfb_smart_model.pkl'")

# Save feature list
with open('v6_features.txt', 'w') as f:
    f.write("Version 6 CFB Spread Model Features\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Training: 2022-2023 ({len(X_train)} games)\n")
    f.write(f"Testing: 2024 ({len(X_test)} games)\n")
    f.write(f"MAE: {test_mae:.2f} points\n\n")
    f.write("Features:\n")
    for feat in FEATURE_COLS:
        f.write(f"  - {feat}\n")

print("Feature list saved to 'v6_features.txt'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("VERSION 6 BRAIN RESTORED")
print("=" * 60)

print(f"""
CFB Spread Model Training Complete:

Data:
  - Training: 2022-2023 ({len(X_train)} games)
  - Testing: 2024 ({len(X_test)} games)

Model Performance:
  - Training MAE: {train_mae:.2f} points
  - Testing MAE: {test_mae:.2f} points

Features Used: {len(FEATURE_COLS)}
  - Elo Ratings (2)
  - Rolling Stats (4)
  - EPA/PPA Efficiency (4)
  - Net EPA (1)
  - Home Field Advantage (2)
  - Rest (3)

Files Created:
  - cfb_smart_model.pkl: Trained model
  - v6_features.txt: Feature documentation
""")

print("=" * 60)
print("VERSION 6 BRAIN RESTORED")
print("=" * 60)
