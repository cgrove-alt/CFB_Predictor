"""
Hyperparameter Tuning for CFB Betting Model.

Uses RandomizedSearchCV to find optimal parameters for HistGradientBoostingRegressor.
Trains on 2022-2024, Tests on 2025.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
FEATURE_COLS = [
    'home_pregame_elo', 'away_pregame_elo',
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',
]

# Parameter grid for tuning (new expanded grid)
PARAM_DISTRIBUTIONS = {
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'max_iter': [100, 300, 500, 1000],
    'max_depth': [3, 5, 8, None],
    'l2_regularization': [0.0, 0.1, 1.0, 5.0],
}

N_ITER = 60  # Number of random combinations to try
CV_FOLDS = 5  # Cross-validation folds

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

print("\nLoading data...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games: {len(df)}")

# Sort by season and week
df = df.sort_values(['season', 'week']).reset_index(drop=True)

# Filter to complete cases
df_valid = df.dropna(subset=FEATURE_COLS + ['Margin'])
print(f"Games with complete data: {len(df_valid)}")

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

# Training: 2022-2024
train_mask = df_valid['season'].isin([2022, 2023, 2024])
# Validation: 2025
test_mask = df_valid['season'] == 2025

X_train = df_valid[train_mask][FEATURE_COLS]
y_train = df_valid[train_mask]['Margin']
X_test = df_valid[test_mask][FEATURE_COLS]
y_test = df_valid[test_mask]['Margin']

print(f"Training samples: {len(X_train)} (2022-2024)")
print(f"Validation samples: {len(X_test)} (2025)")

# ============================================================
# BASELINE MODEL (Current Settings)
# ============================================================
print("\n" + "=" * 60)
print("BASELINE MODEL")
print("=" * 60)

baseline_model = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print(f"\nBaseline Parameters:")
print(f"  max_iter: 100")
print(f"  max_depth: 6")
print(f"  learning_rate: 0.1")
print(f"  l2_regularization: 0.0 (default)")
print(f"\nBaseline MAE: {baseline_mae:.2f} points")

# ============================================================
# RANDOMIZED SEARCH CV
# ============================================================
print("\n" + "=" * 60)
print("RANDOMIZED SEARCH CV")
print("=" * 60)

print(f"\nParameter Grid:")
for param, values in PARAM_DISTRIBUTIONS.items():
    print(f"  {param}: {values}")

print(f"\nSearching {N_ITER} random parameter combinations...")
print(f"Using {CV_FOLDS}-fold TimeSeriesSplit cross-validation...")

# Create model
model = HistGradientBoostingRegressor(random_state=42)

# Use TimeSeriesSplit for proper temporal validation
tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

# RandomizedSearchCV
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=PARAM_DISTRIBUTIONS,
    n_iter=N_ITER,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit
search.fit(X_train, y_train)

# ============================================================
# RESULTS
# ============================================================
print("\n" + "=" * 60)
print("BEST PARAMETERS FOUND")
print("=" * 60)

print(f"\nBest parameters:")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV Score (neg MAE): {search.best_score_:.4f}")
print(f"Best CV MAE: {-search.best_score_:.2f} points")

# ============================================================
# EVALUATE ON VALIDATION SET (2025)
# ============================================================
print("\n" + "=" * 60)
print("VALIDATION SET EVALUATION (2025)")
print("=" * 60)

# Get best model
best_model = search.best_estimator_

# Predict on validation set
y_pred = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)

print(f"\nValidation Set MAE (2025 games):")
print(f"  Baseline: {baseline_mae:.2f} points")
print(f"  Tuned:    {test_mae:.2f} points")
print(f"  Change:   {test_mae - baseline_mae:+.2f} points")

if test_mae < baseline_mae:
    improvement = (baseline_mae - test_mae) / baseline_mae * 100
    print(f"\n✅ IMPROVEMENT: {improvement:.1f}% better!")
else:
    degradation = (test_mae - baseline_mae) / baseline_mae * 100
    print(f"\n⚠️ DEGRADATION: {degradation:.1f}% worse")

# ============================================================
# TOP 5 PARAMETER COMBINATIONS
# ============================================================
print("\n" + "=" * 60)
print("TOP 5 PARAMETER COMBINATIONS")
print("=" * 60)

results_df = pd.DataFrame(search.cv_results_)
results_df = results_df.sort_values('rank_test_score')

print(f"\n{'Rank':<6} {'MAE':<8} {'LR':<8} {'Depth':<8} {'L2':<8} {'Iter':<8}")
print("-" * 46)

for i, (_, row) in enumerate(results_df.head(5).iterrows()):
    mae = -row['mean_test_score']
    params = row['params']
    depth = params.get('max_depth', 'None')
    if depth is None:
        depth = 'None'
    print(f"{i+1:<6} {mae:<8.2f} {params.get('learning_rate', 'N/A'):<8} "
          f"{str(depth):<8} {params.get('l2_regularization', 'N/A'):<8} "
          f"{params.get('max_iter', 'N/A'):<8}")

# ============================================================
# SAVE TUNED MODEL
# ============================================================
print("\n" + "=" * 60)
print("SAVING TUNED MODEL")
print("=" * 60)

# Retrain on full training data with best params
final_model = HistGradientBoostingRegressor(
    **search.best_params_,
    random_state=42
)
final_model.fit(X_train, y_train)

joblib.dump(final_model, 'cfb_tuned_model.pkl')
print(f"\nTuned model saved to 'cfb_tuned_model.pkl'")

# Save best parameters
with open('best_params.txt', 'w') as f:
    f.write("Best Hyperparameters for CFB Predictor\n")
    f.write("=" * 40 + "\n")
    for param, value in search.best_params_.items():
        f.write(f"{param}: {value}\n")
    f.write(f"\nValidation MAE (2025): {test_mae:.2f} points\n")
    f.write(f"Baseline MAE: {baseline_mae:.2f} points\n")
    f.write(f"Improvement: {baseline_mae - test_mae:+.2f} points\n")

print("Best parameters saved to 'best_params.txt'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("TUNING SUMMARY")
print("=" * 60)

print(f"""
Training Data: 2022-2024 ({len(X_train)} games)
Validation Data: 2025 ({len(X_test)} games)

Baseline Model:
  - learning_rate: 0.1
  - max_depth: 6
  - max_iter: 100
  - l2_regularization: 0.0
  - MAE: {baseline_mae:.2f} points

Tuned Model:
  - learning_rate: {search.best_params_.get('learning_rate')}
  - max_depth: {search.best_params_.get('max_depth')}
  - max_iter: {search.best_params_.get('max_iter')}
  - l2_regularization: {search.best_params_.get('l2_regularization')}
  - MAE: {test_mae:.2f} points

Result: {test_mae - baseline_mae:+.2f} points ({((test_mae - baseline_mae) / baseline_mae * 100):+.1f}%)
""")

print("=" * 60)
print("TUNING COMPLETE")
print("=" * 60)
