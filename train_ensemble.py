"""
Train Ensemble Model for CFB Spread Prediction.

Combines 3 models using VotingRegressor:
- HistGradientBoostingRegressor (weight: 0.5)
- RandomForestRegressor (weight: 0.3)
- LinearRegression (weight: 0.2)

Uses the exact 16 features from Version 6.
"""

import pandas as pd
import numpy as np
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
print("TRAIN ENSEMBLE MODEL")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")

# ============================================================
# FEATURE ENGINEERING (Same as V6)
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
# DEFINE V7 FEATURES (21 Total - With Opponent Adjustments)
# ============================================================
print("\n" + "=" * 60)
print("VERSION 7 FEATURES (21 Total - With Opponent Adjustments)")
print("=" * 60)

FEATURE_COLS = [
    # Elo ratings (power rankings)
    'home_pregame_elo', 'away_pregame_elo',

    # Rolling stats (recent form - points)
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',

    # Raw EPA/PPA - Points Per Attempt (efficiency)
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',

    # OPPONENT-ADJUSTED EPA (New in V7!)
    'home_adj_off_epa', 'away_adj_off_epa',
    'home_adj_def_epa', 'away_adj_def_epa',
    'adj_net_epa',

    # Net EPA - Schematic Mismatch (Raw)
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
# PREPARE DATA
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

# Filter to complete cases
df_valid = df.dropna(subset=available_features + ['Margin'])
print(f"Games with complete data: {len(df_valid)}")

# Train/Test split by season
# Train on 2022-2024, Test on 2025
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
# DEFINE INDIVIDUAL MODELS
# ============================================================
print("\n" + "=" * 60)
print("DEFINING MODELS")
print("=" * 60)

# Model 1: HistGradientBoosting (The current brain)
model1 = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
print("\n1. HistGradientBoostingRegressor")
print("   - max_iter: 100")
print("   - max_depth: 6")
print("   - learning_rate: 0.1")

# Model 2: RandomForest
model2 = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
print("\n2. RandomForestRegressor")
print("   - n_estimators: 100")
print("   - max_depth: 10")

# Model 3: LinearRegression
model3 = LinearRegression()
print("\n3. LinearRegression")
print("   - (no hyperparameters)")

# ============================================================
# TRAIN SINGLE BOOSTING MODEL (BASELINE)
# ============================================================
print("\n" + "=" * 60)
print("BASELINE: SINGLE HISTGRADIENTBOOSTING MODEL")
print("=" * 60)

print("\nTraining single HistGradientBoosting model...")
single_model = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
single_model.fit(X_train, y_train)
single_pred = single_model.predict(X_test)
single_mae = mean_absolute_error(y_test, single_pred)
print(f"Single Model MAE: {single_mae:.2f} points")

# ============================================================
# CREATE ENSEMBLE (VOTING REGRESSOR)
# ============================================================
print("\n" + "=" * 60)
print("CREATING ENSEMBLE (VotingRegressor)")
print("=" * 60)

# Weights: Trust Boosting most (0.5), RandomForest second (0.3), Linear last (0.2)
weights = [0.5, 0.3, 0.2]
print(f"\nWeights: HistGradientBoosting={weights[0]}, RandomForest={weights[1]}, LinearRegression={weights[2]}")

ensemble = VotingRegressor(
    estimators=[
        ('hgb', model1),
        ('rf', model2),
        ('lr', model3)
    ],
    weights=weights
)

# ============================================================
# TRAIN ENSEMBLE
# ============================================================
print("\n" + "=" * 60)
print("TRAINING ENSEMBLE")
print("=" * 60)

print("\nFitting ensemble model (this may take a moment)...")
ensemble.fit(X_train, y_train)
print("Ensemble training complete!")

# ============================================================
# EVALUATE ENSEMBLE
# ============================================================
print("\n" + "=" * 60)
print(f"EVALUATION ({test_year})")
print("=" * 60)

ensemble_pred = ensemble.predict(X_test)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

print(f"\n{'Model':<35} {'MAE':>12}")
print("-" * 50)
print(f"{'Single HistGradientBoosting':<35} {single_mae:>12.2f}")
print(f"{'Ensemble (HGB + RF + LR)':<35} {ensemble_mae:>12.2f}")
print("-" * 50)

improvement = single_mae - ensemble_mae
if improvement > 0:
    print(f"\n✓ Ensemble IMPROVES by {improvement:.2f} points ({improvement/single_mae*100:.1f}%)")
else:
    print(f"\n✗ Single model is better by {-improvement:.2f} points")

# ============================================================
# DETAILED ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

single_errors = y_test - single_pred
ensemble_errors = y_test - ensemble_pred

print(f"\n{'Metric':<25} {'Single':>12} {'Ensemble':>12}")
print("-" * 50)
print(f"{'Mean Error (Bias)':<25} {single_errors.mean():>+12.2f} {ensemble_errors.mean():>+12.2f}")
print(f"{'Std Error':<25} {single_errors.std():>12.2f} {ensemble_errors.std():>12.2f}")
print(f"{'Median Abs Error':<25} {single_errors.abs().median():>12.2f} {ensemble_errors.abs().median():>12.2f}")
print(f"{'Max Error':<25} {single_errors.abs().max():>12.2f} {ensemble_errors.abs().max():>12.2f}")

# Error distribution
print(f"\n{'Error Range':<25} {'Single':>12} {'Ensemble':>12}")
print("-" * 50)
for threshold in [3, 7, 10, 14]:
    single_pct = (single_errors.abs() <= threshold).mean() * 100
    ensemble_pct = (ensemble_errors.abs() <= threshold).mean() * 100
    print(f"{'Within ' + str(threshold) + ' pts':<25} {single_pct:>11.1f}% {ensemble_pct:>11.1f}%")

# ============================================================
# INDIVIDUAL MODEL PERFORMANCE
# ============================================================
print("\n" + "=" * 60)
print("INDIVIDUAL MODEL PERFORMANCE")
print("=" * 60)

# Train and evaluate each model separately
for name, model in [('HistGradientBoosting', model1), ('RandomForest', model2), ('LinearRegression', model3)]:
    model_copy = model.__class__(**model.get_params())
    model_copy.fit(X_train, y_train)
    pred = model_copy.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    print(f"{name:<30} MAE: {mae:.2f} points")

# ============================================================
# SAVE ENSEMBLE MODEL
# ============================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(ensemble, 'cfb_ensemble.pkl')
print("\nEnsemble model saved to 'cfb_ensemble.pkl'")

# Also save feature list
with open('ensemble_features.txt', 'w') as f:
    f.write("CFB Ensemble Model Features\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Training: 2022-2024 ({len(X_train)} games)\n")
    f.write(f"Testing: {test_year} ({len(X_test)} games)\n")
    f.write(f"Single Model MAE: {single_mae:.2f} points\n")
    f.write(f"Ensemble MAE: {ensemble_mae:.2f} points\n\n")
    f.write("Ensemble Components:\n")
    f.write("  - HistGradientBoosting (weight: 0.5)\n")
    f.write("  - RandomForest (weight: 0.3)\n")
    f.write("  - LinearRegression (weight: 0.2)\n\n")
    f.write(f"Features ({len(available_features)}):\n")
    for feat in available_features:
        f.write(f"  - {feat}\n")

print("Feature list saved to 'ensemble_features.txt'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ENSEMBLE TRAINING COMPLETE")
print("=" * 60)

print(f"""
Summary:
  - Training games: {len(X_train)}
  - Testing games: {len(X_test)}
  - Features: {len(available_features)}

Performance:
  - Single Model MAE: {single_mae:.2f} points
  - Ensemble MAE: {ensemble_mae:.2f} points
  - Improvement: {improvement:+.2f} points

Files Created:
  - cfb_ensemble.pkl: Ensemble model
  - ensemble_features.txt: Feature documentation
""")

print("=" * 60)
