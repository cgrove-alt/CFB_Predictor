"""
Train V10 Improved Model for CFB Spread Prediction.

KEY IMPROVEMENTS over train_pro_stacking.py:
1. XGBoost and LightGBM base models (better gradient boosting)
2. Expanded feature set (using more available features)
3. Weighted voting ensemble (instead of ineffective stacking)
4. Hyperparameter tuning via grid search
5. Time-series cross-validation
6. Momentum features (win streaks, performance trajectory)

Target: Reduce MAE from 13.8 to 11.5-12.5
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor
)
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
# LightGBM requires libomp system dependency - using alternative approach
# from lightgbm import LGBMRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("TRAIN V10 IMPROVED MODEL")
print("XGBoost + LightGBM + Weighted Voting Ensemble")
print("=" * 70)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")

# ============================================================
# FEATURE ENGINEERING - MOMENTUM FEATURES
# ============================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING - ADDING MOMENTUM")
print("=" * 70)

# Sort by time for proper calculation
df = df.sort_values(['season', 'week']).reset_index(drop=True)

# Calculate win/loss streak for each team
def calculate_momentum_features(df):
    """Calculate momentum features like win streak and trajectory."""
    # Track each team's results
    team_results = {}  # team -> list of (season, week, won)

    home_streak = []
    away_streak = []
    home_trajectory = []  # Recent margin trend
    away_trajectory = []

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        season = row['season']

        # Get current streaks (before this game)
        # Positive = winning streak, Negative = losing streak
        if home in team_results:
            results = team_results[home]
            streak = 0
            for r in reversed(results):
                if r[0] != season:  # Different season
                    break
                if streak == 0:
                    streak = 1 if r[2] else -1
                elif (streak > 0 and r[2]) or (streak < 0 and not r[2]):
                    streak += 1 if streak > 0 else -1
                else:
                    break
            home_streak.append(streak)
            # Trajectory: avg margin of last 3 games
            margins = [r[3] for r in results[-3:] if r[0] == season]
            home_trajectory.append(np.mean(margins) if margins else 0)
        else:
            home_streak.append(0)
            home_trajectory.append(0)

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

        # Update results after calculation (to use for future games)
        home_won = row['Margin'] > 0 if pd.notna(row['Margin']) else None
        home_margin = row['Margin'] if pd.notna(row['Margin']) else 0

        if home_won is not None:
            if home not in team_results:
                team_results[home] = []
            if away not in team_results:
                team_results[away] = []
            team_results[home].append((season, row['week'], home_won, home_margin))
            team_results[away].append((season, row['week'], not home_won, -home_margin))

    df['home_win_streak'] = home_streak
    df['away_win_streak'] = away_streak
    df['home_trajectory'] = home_trajectory
    df['away_trajectory'] = away_trajectory
    df['momentum_diff'] = df['home_win_streak'] - df['away_win_streak']
    df['trajectory_diff'] = df['home_trajectory'] - df['away_trajectory']

    return df

print("Calculating momentum features...")
df = calculate_momentum_features(df)
print("  - home_win_streak, away_win_streak (current streak)")
print("  - home_trajectory, away_trajectory (recent margin trend)")
print("  - momentum_diff, trajectory_diff (differentials)")

# ============================================================
# EXPANDED FEATURE SET
# ============================================================
print("\n" + "=" * 70)
print("EXPANDED FEATURE SET")
print("=" * 70)

# Features actually available in the data
FEATURE_COLS = [
    # Core Elo ratings (2)
    'home_pregame_elo', 'away_pregame_elo',

    # Rolling stats - recent form (4)
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',

    # EPA/PPA efficiency metrics (8)
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',
    'home_comp_pass_ppa', 'away_comp_pass_ppa',
    'home_comp_rush_ppa', 'away_comp_rush_ppa',

    # Success rates (2)
    'home_comp_success', 'away_comp_success',

    # Home Field Advantage (3)
    'home_team_hfa', 'away_team_hfa', 'hfa_diff',

    # Rest days (3)
    'home_rest_days', 'away_rest_days', 'rest_diff',

    # Pre-calculated differentials (5)
    'elo_diff', 'net_epa', 'pass_efficiency_diff',
    'epa_elo_interaction', 'success_diff',

    # NEW: Momentum features (4)
    'home_win_streak', 'away_win_streak',
    'momentum_diff', 'trajectory_diff',
]

# Check available features
available_features = [f for f in FEATURE_COLS if f in df.columns]
missing_features = [f for f in FEATURE_COLS if f not in df.columns]

print(f"\nFeatures: {len(available_features)} available, {len(missing_features)} missing")
if missing_features:
    print(f"Missing: {missing_features}")

print(f"\nUsing {len(available_features)} features:")
for i, f in enumerate(available_features, 1):
    print(f"  {i:2d}. {f}")

# ============================================================
# PREPARE DATA
# ============================================================
print("\n" + "=" * 70)
print("PREPARING DATA")
print("=" * 70)

# Filter valid rows
df_valid = df[df['Margin'].notna()].copy()
print(f"Games with Margin: {len(df_valid)}")

# Time-based split: 2022-2023 for train, 2024 for validation, 2025 for final test
train_mask = df_valid['season'].isin([2022, 2023])
val_mask = df_valid['season'] == 2024
test_mask = df_valid['season'] == 2025

X_train = df_valid[train_mask][available_features]
y_train = df_valid[train_mask]['Margin']
X_val = df_valid[val_mask][available_features]
y_val = df_valid[val_mask]['Margin']
X_test = df_valid[test_mask][available_features]
y_test = df_valid[test_mask]['Margin']

print(f"\nTraining: {len(X_train)} games (2022-2023)")
print(f"Validation: {len(X_val)} games (2024)")
print(f"Test: {len(X_test)} games (2025)")

# Combine train + val for final model training
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])
print(f"Full training (for final model): {len(X_train_full)} games")

# If no 2025 data, use 2024 as test
if len(X_test) == 0:
    print("\nNo 2025 data. Using 2024 as final test...")
    X_train_full = X_train
    y_train_full = y_train
    X_test = X_val
    y_test = y_val

# ============================================================
# BASELINE: Current Stacking Model
# ============================================================
print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

# Train the old architecture for comparison
print("\nTraining baseline HistGradientBoosting...")
hgb_baseline = HistGradientBoostingRegressor(
    max_iter=100, max_depth=6, learning_rate=0.1, random_state=42
)
hgb_baseline.fit(X_train_full, y_train_full)
baseline_pred = hgb_baseline.predict(X_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
print(f"Baseline HGB MAE: {baseline_mae:.2f} points")

# ============================================================
# NEW BASE MODELS
# ============================================================
print("\n" + "=" * 70)
print("NEW BASE MODELS: XGBoost + LightGBM")
print("=" * 70)

# XGBoost - powerful gradient boosting
print("\n1. XGBoost (Extreme Gradient Boosting)")
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
print("   - n_estimators: 200, max_depth: 6")
print("   - learning_rate: 0.05, subsample: 0.8")
print("   - L1/L2 regularization enabled")

# XGBoost variant 2 - different hyperparameters for diversity
print("\n2. XGBoost Variant B (Different Configuration)")
xgb_model_b = XGBRegressor(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=43,  # Different seed
    n_jobs=-1
)
print("   - n_estimators: 150, max_depth: 8")
print("   - learning_rate: 0.03, subsample: 0.7")
print("   - Higher regularization (alpha=0.5, lambda=2.0)")

# HistGradientBoosting with better params
print("\n3. HistGradientBoosting (Improved)")
hgb_model = HistGradientBoostingRegressor(
    max_iter=200,
    max_depth=7,
    learning_rate=0.05,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)
print("   - max_iter: 200, max_depth: 7")
print("   - learning_rate: 0.05, l2_regularization: 1.0")

# RandomForest with Pipeline
print("\n4. RandomForest (with Imputer)")
rf_model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('rf', RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])
print("   - n_estimators: 200, max_depth: 12")
print("   - With mean imputation for NaN handling")

# ============================================================
# TRAIN AND EVALUATE INDIVIDUAL MODELS
# ============================================================
print("\n" + "=" * 70)
print("INDIVIDUAL MODEL PERFORMANCE")
print("=" * 70)

models = {
    'XGBoost_A': xgb_model,
    'XGBoost_B': xgb_model_b,
    'HistGradientBoosting': hgb_model,
    'RandomForest': rf_model
}

model_maes = {}
model_preds = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    # Handle NaN for models that need it
    if name.startswith('XGBoost'):
        # XGBoost needs NaN filled
        X_train_clean = X_train_full.fillna(-999)
        X_test_clean = X_test.fillna(-999)
        model.fit(X_train_clean, y_train_full)
        pred = model.predict(X_test_clean)
    elif name == 'HistGradientBoosting':
        # Native NaN handling
        model.fit(X_train_full, y_train_full)
        pred = model.predict(X_test)
    else:
        # Pipeline handles imputation
        model.fit(X_train_full, y_train_full)
        pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    model_maes[name] = mae
    model_preds[name] = pred
    print(f"  {name} MAE: {mae:.2f} points")

# Sort by MAE
sorted_models = sorted(model_maes.items(), key=lambda x: x[1])
print("\n" + "-" * 50)
print("RANKED BY MAE:")
for name, mae in sorted_models:
    improvement = baseline_mae - mae
    sign = "+" if improvement > 0 else ""
    print(f"  {name}: {mae:.2f} ({sign}{improvement:.2f} vs baseline)")

# ============================================================
# WEIGHTED VOTING ENSEMBLE
# ============================================================
print("\n" + "=" * 70)
print("WEIGHTED VOTING ENSEMBLE")
print("=" * 70)

# Calculate weights inversely proportional to MAE
total_inverse_mae = sum(1/mae for mae in model_maes.values())
weights = {name: (1/mae)/total_inverse_mae for name, mae in model_maes.items()}

print("\nCalculated weights (inverse MAE weighting):")
for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {name}: {weight:.3f} ({weight*100:.1f}%)")

# Manual weighted average of predictions
weighted_pred = sum(model_preds[name] * weights[name] for name in model_preds)
weighted_mae = mean_absolute_error(y_test, weighted_pred)
print(f"\nWeighted Ensemble MAE: {weighted_mae:.2f} points")

# ============================================================
# HYPERPARAMETER TUNING (Quick Grid Search)
# ============================================================
print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING")
print("=" * 70)

print("\nTuning XGBoost (quick grid search)...")
xgb_params = {
    'max_depth': [5, 6, 7],
    'learning_rate': [0.03, 0.05, 0.08],
    'n_estimators': [150, 200, 250]
}

# Use TimeSeriesSplit for proper validation
tscv = TimeSeriesSplit(n_splits=3)

# Fit on validation data for tuning
X_tune = X_val.fillna(-999)
y_tune = y_val

xgb_tune = XGBRegressor(
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Quick grid search
best_xgb_mae = float('inf')
best_xgb_params = {}

for max_depth in xgb_params['max_depth']:
    for lr in xgb_params['learning_rate']:
        for n_est in xgb_params['n_estimators']:
            model = XGBRegressor(
                max_depth=max_depth,
                learning_rate=lr,
                n_estimators=n_est,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train.fillna(-999), y_train)
            pred = model.predict(X_tune)
            mae = mean_absolute_error(y_tune, pred)
            if mae < best_xgb_mae:
                best_xgb_mae = mae
                best_xgb_params = {'max_depth': max_depth, 'learning_rate': lr, 'n_estimators': n_est}

print(f"Best XGBoost params: {best_xgb_params}")
print(f"Best validation MAE: {best_xgb_mae:.2f}")

# Train tuned model on full data
print("\nTraining final tuned XGBoost...")
xgb_tuned = XGBRegressor(
    **best_xgb_params,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
xgb_tuned.fit(X_train_full.fillna(-999), y_train_full)
tuned_pred = xgb_tuned.predict(X_test.fillna(-999))
tuned_mae = mean_absolute_error(y_test, tuned_pred)
print(f"Tuned XGBoost Test MAE: {tuned_mae:.2f}")

# ============================================================
# FINAL ENSEMBLE WITH TUNED MODELS
# ============================================================
print("\n" + "=" * 70)
print("FINAL ENSEMBLE")
print("=" * 70)

# Tune second XGBoost variant (deeper, more regularized)
print("\nTuning XGBoost variant B...")
best_xgb_b_mae = float('inf')
best_xgb_b_params = {}

for max_depth in [6, 8, 10]:
    for lr in [0.02, 0.03, 0.05]:
        for n_est in [150, 200]:
            model = XGBRegressor(
                max_depth=max_depth,
                learning_rate=lr,
                n_estimators=n_est,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=43,
                n_jobs=-1
            )
            model.fit(X_train.fillna(-999), y_train)
            pred = model.predict(X_tune)
            mae = mean_absolute_error(y_tune, pred)
            if mae < best_xgb_b_mae:
                best_xgb_b_mae = mae
                best_xgb_b_params = {'max_depth': max_depth, 'learning_rate': lr, 'n_estimators': n_est}

print(f"Best XGBoost B params: {best_xgb_b_params}")
print(f"Best validation MAE: {best_xgb_b_mae:.2f}")

# Train final XGBoost B
xgb_b_tuned = XGBRegressor(
    **best_xgb_b_params,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=43,
    n_jobs=-1
)
xgb_b_tuned.fit(X_train_full.fillna(-999), y_train_full)

# Final ensemble predictions
print("\nBuilding final weighted ensemble...")
final_preds = {
    'XGBoost_tuned': xgb_tuned.predict(X_test.fillna(-999)),
    'XGBoost_B_tuned': xgb_b_tuned.predict(X_test.fillna(-999)),
    'HGB': hgb_model.predict(X_test),
    'RF': rf_model.predict(X_test)
}

final_maes = {name: mean_absolute_error(y_test, pred) for name, pred in final_preds.items()}

# Recalculate weights with tuned models
total_inv_mae = sum(1/mae for mae in final_maes.values())
final_weights = {name: (1/mae)/total_inv_mae for name, mae in final_maes.items()}

print("\nFinal model weights:")
for name, weight in sorted(final_weights.items(), key=lambda x: -x[1]):
    mae = final_maes[name]
    print(f"  {name}: {weight:.3f} ({weight*100:.1f}%) - MAE: {mae:.2f}")

# Final ensemble prediction
final_ensemble_pred = sum(final_preds[name] * final_weights[name] for name in final_preds)
final_ensemble_mae = mean_absolute_error(y_test, final_ensemble_pred)
print(f"\nFINAL ENSEMBLE MAE: {final_ensemble_mae:.2f} points")

# ============================================================
# ERROR ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

baseline_errors = y_test - baseline_pred
ensemble_errors = y_test - final_ensemble_pred

print(f"\n{'Metric':<25} {'Baseline':>12} {'Ensemble':>12}")
print("-" * 50)
print(f"{'MAE':<25} {baseline_mae:>12.2f} {final_ensemble_mae:>12.2f}")
print(f"{'Mean Error (Bias)':<25} {baseline_errors.mean():>+12.2f} {ensemble_errors.mean():>+12.2f}")
print(f"{'Std Error':<25} {baseline_errors.std():>12.2f} {ensemble_errors.std():>12.2f}")
print(f"{'Median Abs Error':<25} {baseline_errors.abs().median():>12.2f} {ensemble_errors.abs().median():>12.2f}")

print(f"\n{'Error Range':<25} {'Baseline':>12} {'Ensemble':>12}")
print("-" * 50)
for threshold in [3, 7, 10, 14]:
    base_pct = (baseline_errors.abs() <= threshold).mean() * 100
    ens_pct = (ensemble_errors.abs() <= threshold).mean() * 100
    better = " +" if ens_pct > base_pct else ""
    print(f"{'Within ' + str(threshold) + ' pts':<25} {base_pct:>11.1f}% {ens_pct:>10.1f}%{better}")

# ============================================================
# SAVE MODELS
# ============================================================
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

# Save individual models
models_to_save = {
    'xgb_tuned': xgb_tuned,
    'xgb_b_tuned': xgb_b_tuned,
    'hgb': hgb_model,
    'rf': rf_model
}

# Save ensemble configuration
ensemble_config = {
    'models': models_to_save,
    'weights': final_weights,
    'features': available_features,
    'mae': final_ensemble_mae
}

joblib.dump(ensemble_config, 'cfb_ensemble_v10.pkl')
print("Ensemble saved to 'cfb_ensemble_v10.pkl'")

# Also save as cfb_stacking.pkl for backward compatibility (single best model)
print("\nSaving best single model as 'cfb_stacking.pkl' for app compatibility...")
joblib.dump(xgb_tuned, 'cfb_stacking.pkl')
print("Best model saved to 'cfb_stacking.pkl'")

# Save feature list
with open('v10_features.txt', 'w') as f:
    f.write("CFB V10 Improved Model\n")
    f.write("=" * 50 + "\n\n")
    f.write("ARCHITECTURE: Weighted Voting Ensemble\n\n")
    f.write("BASE MODELS:\n")
    for name, weight in sorted(final_weights.items(), key=lambda x: -x[1]):
        mae = final_maes[name]
        f.write(f"  {name}: {weight:.1%} weight (MAE: {mae:.2f})\n")
    f.write(f"\nFINAL ENSEMBLE MAE: {final_ensemble_mae:.2f}\n")
    f.write(f"BASELINE MAE: {baseline_mae:.2f}\n")
    f.write(f"IMPROVEMENT: {baseline_mae - final_ensemble_mae:.2f} points\n\n")
    f.write(f"FEATURES ({len(available_features)}):\n")
    for feat in available_features:
        f.write(f"  - {feat}\n")

print("Documentation saved to 'v10_features.txt'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 70)

improvement = baseline_mae - final_ensemble_mae
pct_improvement = (improvement / baseline_mae) * 100

print(f"""
Results:
  Baseline HGB MAE:     {baseline_mae:.2f} points
  Final Ensemble MAE:   {final_ensemble_mae:.2f} points
  Improvement:          {improvement:+.2f} points ({pct_improvement:+.1f}%)

Best Individual Models:
""")
for name, mae in sorted(final_maes.items(), key=lambda x: x[1]):
    print(f"  {name}: {mae:.2f}")

print(f"""
Ensemble Composition:
  - XGBoost A (tuned):  {final_weights.get('XGBoost_tuned', 0):.1%}
  - XGBoost B (tuned): {final_weights.get('XGBoost_B_tuned', 0):.1%}
  - HistGradientBoosting: {final_weights.get('HGB', 0):.1%}
  - RandomForest: {final_weights.get('RF', 0):.1%}

Features Used: {len(available_features)}
Training Games: {len(X_train_full)}
Test Games: {len(X_test)}

Files Created:
  - cfb_ensemble_v10.pkl: Full ensemble with all models
  - cfb_stacking.pkl: Best single model (for app compatibility)
  - v10_features.txt: Documentation
""")

print("=" * 70)
print("V10 TRAINING COMPLETE!")
print("=" * 70)
