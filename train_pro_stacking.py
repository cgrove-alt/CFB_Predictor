"""
Train Pro Stacking Model for CFB Spread Prediction.

Architecture: StackingRegressor with specialized experts and a meta-learner boss.

BASE MODELS (The Experts):
- Gradient: HistGradientBoostingRegressor (handles NaN natively)
- Forest: RandomForestRegressor (ensemble of trees)
- Linear: Pipeline with Imputer/Scaler/Ridge (robust linear model)

META MODEL (The Boss):
- RidgeCV: Cross-validated Ridge that learns optimal expert weights

This architecture fixes NaN crashes by giving each model its own preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor
)
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("TRAIN PRO STACKING MODEL")
print("The Experts + The Boss Architecture")
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
# DEFINE 21 FEATURES (16 Standard + 5 Interactions)
# ============================================================
print("\n" + "=" * 60)
print("21 FEATURES (16 Standard + 5 Interactions)")
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
    'rest_advantage',

    # NEW: Interaction Features
    'rest_diff',              # Fatigue factor
    'elo_diff',               # Talent/strength differential
    'pass_efficiency_diff',   # Passing game mismatch
    'epa_elo_interaction',    # Efficiency x strength
    'success_diff'            # Play success rate differential
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

# Only filter on target (Margin), not features - models handle NaN
df_valid = df[df['Margin'].notna()].copy()
print(f"Games with Margin: {len(df_valid)}")

# Count NaN values per feature
print("\nNaN counts per feature:")
nan_found = False
for f in available_features:
    nan_count = df_valid[f].isna().sum()
    if nan_count > 0:
        print(f"  {f}: {nan_count} NaN values")
        nan_found = True
if not nan_found:
    print("  No NaN values found!")

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
# DEFINE THE EXPERTS (BASE MODELS)
# ============================================================
print("\n" + "=" * 60)
print("THE EXPERTS (BASE MODELS)")
print("=" * 60)

# Expert 1: Gradient (HistGradientBoosting - handles NaN natively)
expert_gradient = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
print("\nExpert 1: GRADIENT (HistGradientBoosting)")
print("  - Handles NaN natively")
print("  - max_iter: 100, max_depth: 6, learning_rate: 0.1")

# Expert 2: Forest (RandomForest - needs clean data)
expert_forest = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('rf', RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])
print("\nExpert 2: FOREST (RandomForest)")
print("  - Wrapped in Pipeline with Imputer")
print("  - n_estimators: 100, max_depth: 10")

# Expert 3: Linear (Ridge with full preprocessing)
expert_linear = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])
print("\nExpert 3: LINEAR (Ridge)")
print("  - Full Pipeline: Imputer -> Scaler -> Ridge")
print("  - alpha: 1.0")

# ============================================================
# DEFINE THE BOSS (META MODEL)
# ============================================================
print("\n" + "=" * 60)
print("THE BOSS (META MODEL)")
print("=" * 60)

boss = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
print("\nMeta Model: RIDGECV (The Boss)")
print("  - Cross-validates to find optimal alpha")
print("  - Learns optimal weights for combining experts")
print("  - alphas: [0.1, 1.0, 10.0, 100.0]")

# ============================================================
# BUILD THE STACKING MODEL
# ============================================================
print("\n" + "=" * 60)
print("BUILDING STACKING MODEL")
print("=" * 60)

stacking_model = StackingRegressor(
    estimators=[
        ('gradient', expert_gradient),
        ('forest', expert_forest),
        ('linear', expert_linear)
    ],
    final_estimator=boss,
    cv=5,  # 5-fold cross-validation for generating meta-features
    n_jobs=-1
)

print("\nStackingRegressor Architecture:")
print("  [Input Features]")
print("       |")
print("       +---> [Expert 1: Gradient] ---> Prediction 1")
print("       |")
print("       +---> [Expert 2: Forest]   ---> Prediction 2")
print("       |")
print("       +---> [Expert 3: Linear]   ---> Prediction 3")
print("       |")
print("       v")
print("  [Boss: RidgeCV] combines predictions")
print("       |")
print("       v")
print("  [Final Prediction]")
print("\nThis model learns from each expert's strengths!")

# ============================================================
# TRAIN SINGLE MODEL BASELINE
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
# TRAIN STACKING MODEL
# ============================================================
print("\n" + "=" * 60)
print("TRAINING STACKING MODEL")
print("=" * 60)

print("\nFitting StackingRegressor (experts training + boss learning)...")
print("This takes longer due to 5-fold CV for meta-features...")
stacking_model.fit(X_train, y_train)
print("Stacking training complete!")

# Show what the boss learned
if hasattr(boss, 'alpha_'):
    print(f"\nBoss selected alpha: {boss.alpha_}")

# ============================================================
# EVALUATE
# ============================================================
print("\n" + "=" * 60)
print(f"EVALUATION ({test_year})")
print("=" * 60)

# Make predictions
stacking_pred = stacking_model.predict(X_test)
stacking_mae = mean_absolute_error(y_test, stacking_pred)

print(f"\n{'Model':<35} {'MAE':>12}")
print("-" * 50)
print(f"{'Single HistGradientBoosting':<35} {single_mae:>12.2f}")
print(f"{'Stacking (Gradient+Forest+Linear)':<35} {stacking_mae:>12.2f}")
print("-" * 50)

improvement = single_mae - stacking_mae
if improvement > 0:
    print(f"\n>>> Stacking IMPROVES by {improvement:.2f} points ({improvement/single_mae*100:.1f}%)")
else:
    print(f"\n>>> Single model is better by {-improvement:.2f} points")

# ============================================================
# ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

single_errors = y_test - single_pred
stacking_errors = y_test - stacking_pred

print(f"\n{'Metric':<25} {'Single':>12} {'Stacking':>12}")
print("-" * 50)
print(f"{'Mean Error (Bias)':<25} {single_errors.mean():>+12.2f} {stacking_errors.mean():>+12.2f}")
print(f"{'Std Error':<25} {single_errors.std():>12.2f} {stacking_errors.std():>12.2f}")
print(f"{'Median Abs Error':<25} {single_errors.abs().median():>12.2f} {stacking_errors.abs().median():>12.2f}")
print(f"{'Max Error':<25} {single_errors.abs().max():>12.2f} {stacking_errors.abs().max():>12.2f}")

# Error distribution
print(f"\n{'Error Range':<25} {'Single':>12} {'Stacking':>12}")
print("-" * 50)
for threshold in [3, 7, 10, 14]:
    single_pct = (single_errors.abs() <= threshold).mean() * 100
    stacking_pct = (stacking_errors.abs() <= threshold).mean() * 100
    better = "+" if stacking_pct > single_pct else ""
    print(f"{'Within ' + str(threshold) + ' pts':<25} {single_pct:>11.1f}% {stacking_pct:>10.1f}%{better}")

# ============================================================
# INDIVIDUAL EXPERT PERFORMANCE
# ============================================================
print("\n" + "=" * 60)
print("INDIVIDUAL EXPERT PERFORMANCE")
print("=" * 60)

# Train and evaluate each expert separately
experts = [
    ('Gradient (HGB)', expert_gradient),
    ('Forest (RF)', expert_forest),
    ('Linear (Ridge)', expert_linear)
]

for name, expert in experts:
    # Clone the expert to avoid refitting
    if hasattr(expert, 'fit'):
        expert_clone = expert.__class__(**expert.get_params()) if not isinstance(expert, Pipeline) else Pipeline(expert.steps)
        expert_clone.fit(X_train, y_train)
        pred = expert_clone.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        print(f"{name:<30} MAE: {mae:.2f} points")

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
    prediction = stacking_model.predict(test_input)
    print(f"\nTest with NaN values: SUCCESS!")
    print(f"  Prediction: {prediction[0]:.2f}")
    print("  The Experts handled the missing values!")
except Exception as e:
    print(f"\nTest with NaN values: FAILED")
    print(f"  Error: {e}")

# ============================================================
# SAVE MODEL
# ============================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(stacking_model, 'cfb_stacking.pkl')
print("\nStacking model saved to 'cfb_stacking.pkl'")

# Save feature list and documentation
with open('stacking_features.txt', 'w') as f:
    f.write("CFB Pro Stacking Model\n")
    f.write("=" * 40 + "\n\n")
    f.write("Architecture: StackingRegressor (Experts + Boss)\n\n")
    f.write("THE EXPERTS (Base Models):\n")
    f.write("  1. Gradient (HistGradientBoostingRegressor)\n")
    f.write("     - Handles NaN natively\n")
    f.write("     - max_iter: 100, max_depth: 6\n\n")
    f.write("  2. Forest (RandomForestRegressor)\n")
    f.write("     - Pipeline: Imputer -> RF\n")
    f.write("     - n_estimators: 100, max_depth: 10\n\n")
    f.write("  3. Linear (Ridge)\n")
    f.write("     - Pipeline: Imputer -> Scaler -> Ridge\n")
    f.write("     - alpha: 1.0\n\n")
    f.write("THE BOSS (Meta Model):\n")
    f.write("  - RidgeCV with cross-validation\n")
    f.write("  - Learns optimal expert weights\n\n")
    f.write(f"Training: 2022-2024 ({len(X_train)} games)\n")
    f.write(f"Testing: {test_year} ({len(X_test)} games)\n")
    f.write(f"Single Model MAE: {single_mae:.2f} points\n")
    f.write(f"Stacking MAE: {stacking_mae:.2f} points\n")
    f.write(f"Improvement: {improvement:+.2f} points\n\n")
    f.write(f"Features ({len(available_features)}):\n")
    for feat in available_features:
        f.write(f"  - {feat}\n")

print("Documentation saved to 'stacking_features.txt'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PRO STACKING MODEL TRAINED")
print("=" * 60)

print(f"""
Summary:
  - Architecture: StackingRegressor (Experts + Boss)
  - Training games: {len(X_train)}
  - Testing games: {len(X_test)}
  - Features: {len(available_features)}

Performance:
  - Single Model MAE: {single_mae:.2f} points
  - Stacking MAE: {stacking_mae:.2f} points
  - Improvement: {improvement:+.2f} points ({improvement/single_mae*100:+.1f}%)

The Experts:
  1. Gradient: HistGradientBoosting (handles NaN)
  2. Forest: RandomForest (with Imputer)
  3. Linear: Ridge (with Imputer + Scaler)

The Boss:
  - RidgeCV learns optimal expert combination

Files Created:
  - cfb_stacking.pkl: Pro Stacking model
  - stacking_features.txt: Documentation
""")

print("=" * 60)
print("THE EXPERTS ARE ASSEMBLED. THE BOSS IS IN CHARGE.")
print("=" * 60)
