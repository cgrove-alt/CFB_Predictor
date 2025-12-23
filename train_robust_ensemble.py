"""
Train Robust Ensemble Model for CFB Spread Prediction.

Architecture: Pipeline with Janitor (SimpleImputer) + Senate (VotingRegressor)
- The Janitor: SimpleImputer to handle missing values automatically
- The Senate: VotingRegressor combining 3 models for consensus decisions

This model can handle NaN values at prediction time without crashing.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("TRAIN ROBUST ENSEMBLE MODEL")
print("The Janitor + The Senate Architecture")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Create derived features
df['net_epa'] = df['home_comp_off_ppa'] - df['away_comp_def_ppa']
df['home_rest'] = df['home_rest_days']
df['away_rest'] = df['away_rest_days']

# Rest advantage if not already present
if 'rest_advantage' not in df.columns:
    df['rest_advantage'] = df['home_rest'] - df['away_rest']

print("Derived features created: net_epa, home_rest, away_rest, rest_advantage")

# ============================================================
# DEFINE 16 STANDARD FEATURES
# ============================================================
print("\n" + "=" * 60)
print("STANDARD 16 FEATURES")
print("=" * 60)

FEATURE_COLS = [
    # Elo ratings (power rankings)
    'home_pregame_elo', 'away_pregame_elo',

    # Rolling stats (recent form - points)
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',

    # EPA/PPA - Points Per Attempt (efficiency)
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',

    # Net EPA - Schematic Mismatch
    'net_epa',

    # Home Field Advantage
    'home_team_hfa', 'away_team_hfa',

    # Rest days
    'home_rest', 'away_rest',

    # Rest advantage
    'rest_advantage'
]

print(f"\nFeatures ({len(FEATURE_COLS)} total):")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {f}")

# ============================================================
# PREPARE DATA (Keep NaN rows for training - imputer will handle)
# ============================================================
print("\n" + "=" * 60)
print("PREPARING DATA")
print("=" * 60)

# Check which features exist
available_features = [f for f in FEATURE_COLS if f in df.columns]
missing_features = [f for f in FEATURE_COLS if f not in df.columns]

if missing_features:
    print(f"\nWarning: Missing features: {missing_features}")
    print("Proceeding with available features only.")

print(f"Available features: {len(available_features)}")

# Only filter on target (Margin), not features - Janitor handles NaN
df_valid = df[df['Margin'].notna()].copy()
print(f"Games with Margin: {len(df_valid)}")

# Count NaN values per feature
print("\nNaN counts per feature:")
for f in available_features:
    nan_count = df_valid[f].isna().sum()
    if nan_count > 0:
        print(f"  {f}: {nan_count} NaN values")

# Train/Test split by season
train_mask = df_valid['season'].isin([2022, 2023, 2024])
test_mask = df_valid['season'] == 2025

X_train = df_valid[train_mask][available_features]
y_train = df_valid[train_mask]['Margin']
X_test = df_valid[test_mask][available_features]
y_test = df_valid[test_mask]['Margin']

print(f"\nTraining samples: {len(X_train)} (2022-2024)")
print(f"Testing samples: {len(X_test)} (2025)")

# If no 2025 data, fallback to 2024 as test
if len(X_test) == 0:
    print("\nNo 2025 data. Using 2024 as test set...")
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
# DEFINE THE JANITOR (SimpleImputer)
# ============================================================
print("\n" + "=" * 60)
print("THE JANITOR: SimpleImputer")
print("=" * 60)

janitor = SimpleImputer(strategy='mean')
print("\nThe Janitor is ready to clean missing values:")
print("  - Strategy: 'mean' (replace NaN with column mean)")
print("  - Fitted during training, applied at prediction time")

# ============================================================
# DEFINE THE SENATE (VotingRegressor)
# ============================================================
print("\n" + "=" * 60)
print("THE SENATE: VotingRegressor")
print("=" * 60)

# Senator 1: HistGradientBoosting (The Elder)
senator1 = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
print("\nSenator 1: HistGradientBoosting (The Elder)")
print("  - Weight: 0.5 (Most Trusted)")
print("  - max_iter: 100, max_depth: 6")

# Senator 2: RandomForest (The Crowd)
senator2 = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
print("\nSenator 2: RandomForest (The Crowd)")
print("  - Weight: 0.3")
print("  - n_estimators: 100, max_depth: 10")

# Senator 3: LinearRegression (The Baseline)
senator3 = LinearRegression()
print("\nSenator 3: LinearRegression (The Baseline)")
print("  - Weight: 0.2")

# The Senate: VotingRegressor
weights = [0.5, 0.3, 0.2]
senate = VotingRegressor(
    estimators=[
        ('elder', senator1),
        ('crowd', senator2),
        ('baseline', senator3)
    ],
    weights=weights
)

print(f"\nSenate Weights: Elder={weights[0]}, Crowd={weights[1]}, Baseline={weights[2]}")

# ============================================================
# BUILD THE PIPELINE: Janitor + Senate
# ============================================================
print("\n" + "=" * 60)
print("BUILDING PIPELINE: Janitor + Senate")
print("=" * 60)

model = Pipeline(steps=[
    ('janitor', janitor),   # Step 1: Clean missing values
    ('senate', senate)       # Step 2: Ensemble prediction
])

print("\nPipeline Architecture:")
print("  Input -> [Janitor: Impute NaN] -> [Senate: Vote] -> Prediction")
print("\nThis model will NEVER crash on NaN values!")

# ============================================================
# TRAIN THE PIPELINE
# ============================================================
print("\n" + "=" * 60)
print("TRAINING PIPELINE")
print("=" * 60)

print("\nFitting Pipeline (Janitor learns means, Senate learns patterns)...")
model.fit(X_train, y_train)
print("Training complete!")

# ============================================================
# EVALUATE
# ============================================================
print("\n" + "=" * 60)
print(f"EVALUATION ({test_year})")
print("=" * 60)

# Make predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"\n{'Set':<15} {'Games':>8} {'MAE':>12}")
print("-" * 40)
print(f"{'Training':<15} {len(X_train):>8} {train_mae:>12.2f}")
print(f"{'Testing':<15} {len(X_test):>8} {test_mae:>12.2f}")
print("-" * 40)

# Error analysis
errors = y_test - test_pred
print(f"\nError Analysis:")
print(f"  Mean Error (Bias): {errors.mean():+.2f}")
print(f"  Std Error: {errors.std():.2f}")
print(f"  Within 7 pts: {(errors.abs() <= 7).mean()*100:.1f}%")
print(f"  Within 10 pts: {(errors.abs() <= 10).mean()*100:.1f}%")
print(f"  Within 14 pts: {(errors.abs() <= 14).mean()*100:.1f}%")

# ============================================================
# TEST NAN HANDLING
# ============================================================
print("\n" + "=" * 60)
print("TESTING NAN HANDLING")
print("=" * 60)

# Create a test input with NaN values
test_input = X_test.iloc[0:1].copy()
test_input.iloc[0, 0] = np.nan  # Introduce NaN
test_input.iloc[0, 5] = np.nan  # Introduce another NaN

try:
    prediction = model.predict(test_input)
    print(f"\nTest with NaN values: SUCCESS!")
    print(f"  Prediction: {prediction[0]:.2f}")
    print("  The Janitor handled the missing values!")
except Exception as e:
    print(f"\nTest with NaN values: FAILED")
    print(f"  Error: {e}")

# ============================================================
# SAVE MODEL
# ============================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(model, 'cfb_ensemble.pkl')
print("\nRobust Pipeline saved to 'cfb_ensemble.pkl'")

# Save feature list
with open('ensemble_features.txt', 'w') as f:
    f.write("CFB Robust Ensemble Model\n")
    f.write("=" * 40 + "\n\n")
    f.write("Architecture: Pipeline (Janitor + Senate)\n")
    f.write("  - Janitor: SimpleImputer (strategy='mean')\n")
    f.write("  - Senate: VotingRegressor\n")
    f.write("    - Elder (HGB): weight=0.5\n")
    f.write("    - Crowd (RF): weight=0.3\n")
    f.write("    - Baseline (LR): weight=0.2\n\n")
    f.write(f"Training: 2022-2024 ({len(X_train)} games)\n")
    f.write(f"Testing: {test_year} ({len(X_test)} games)\n")
    f.write(f"MAE: {test_mae:.2f} points\n\n")
    f.write(f"Features ({len(available_features)}):\n")
    for feat in available_features:
        f.write(f"  - {feat}\n")

print("Feature list saved to 'ensemble_features.txt'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ROBUST MODEL TRAINED. THE JANITOR IS INSTALLED.")
print("=" * 60)

print(f"""
Summary:
  - Architecture: Pipeline (Janitor + Senate)
  - Training games: {len(X_train)}
  - Testing games: {len(X_test)}
  - Features: {len(available_features)}
  - Test MAE: {test_mae:.2f} points

The Janitor (SimpleImputer):
  - Handles missing values automatically
  - Strategy: Replace NaN with column mean
  - Fitted on training data

The Senate (VotingRegressor):
  - Elder (HistGradientBoosting): 50% weight
  - Crowd (RandomForest): 30% weight
  - Baseline (LinearRegression): 20% weight

Files Created:
  - cfb_ensemble.pkl: Robust Pipeline model
  - ensemble_features.txt: Documentation
""")

print("=" * 60)
print("THE JANITOR IS INSTALLED. NO MORE NaN CRASHES!")
print("=" * 60)
