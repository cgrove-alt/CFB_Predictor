"""
Optimize Features using Recursive Feature Elimination with Cross Validation (RFECV).

Uses RandomForestRegressor as the estimator to find the minimum subset of features
that gives the lowest MAE. RandomForest exposes feature_importances_ needed for RFECV.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# All potential features from cfb_data_smart.csv
ALL_FEATURES = [
    # Elo ratings
    'home_pregame_elo', 'away_pregame_elo',

    # Rolling stats
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',

    # Rest and situational
    'home_rest_days', 'away_rest_days', 'rest_advantage',
    'west_coast_early', 'home_lookahead', 'away_lookahead',

    # Success rates
    'home_off_rush_success', 'home_off_pass_success',
    'home_def_rush_success', 'home_def_pass_success',
    'away_off_rush_success', 'away_off_pass_success',
    'away_def_rush_success', 'away_def_pass_success',

    # Down-specific PPA
    'home_off_std_downs_ppa', 'home_def_pass_downs_ppa',
    'away_off_std_downs_ppa', 'away_def_pass_downs_ppa',

    # Composite/Competitive PPA
    'home_comp_off_ppa', 'home_comp_def_ppa',
    'home_comp_pass_ppa', 'home_comp_rush_ppa',
    'away_comp_off_ppa', 'away_comp_def_ppa',
    'away_comp_pass_ppa', 'away_comp_rush_ppa',

    # EPA and efficiency
    'home_comp_epa', 'home_comp_success', 'home_comp_ypp',
    'away_comp_epa', 'away_comp_success', 'away_comp_ypp',

    # Clean/Garbage-adjusted
    'home_clean_off_ppa', 'home_clean_def_ppa',
    'away_clean_off_ppa', 'away_clean_def_ppa',

    # HFA
    'home_team_hfa', 'away_team_hfa', 'hfa_diff',
]

TARGET = 'Margin'

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("OPTIMIZE FEATURES WITH RFECV")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")

# Sort by season and week
df = df.sort_values(['season', 'week']).reset_index(drop=True)

# ============================================================
# FILTER TO AVAILABLE FEATURES
# ============================================================
print("\n" + "=" * 60)
print("CHECKING AVAILABLE FEATURES")
print("=" * 60)

available_features = [f for f in ALL_FEATURES if f in df.columns]
missing_features = [f for f in ALL_FEATURES if f not in df.columns]

print(f"\nFeatures available: {len(available_features)}")
print(f"Features missing: {len(missing_features)}")

if missing_features:
    print(f"\nMissing: {missing_features[:10]}...")  # Show first 10

# ============================================================
# PREPARE DATA
# ============================================================
print("\n" + "=" * 60)
print("PREPARING DATA")
print("=" * 60)

# Filter to complete cases
df_valid = df.dropna(subset=available_features + [TARGET])
print(f"Games with complete data: {len(df_valid)}")

# Train/Test split
train_mask = df_valid['season'].isin([2022, 2023, 2024])
test_mask = df_valid['season'] == 2025

X_train = df_valid[train_mask][available_features]
y_train = df_valid[train_mask][TARGET]
X_test = df_valid[test_mask][available_features]
y_test = df_valid[test_mask][TARGET]

print(f"\nTraining samples: {len(X_train)} (2022-2024)")
print(f"Testing samples: {len(X_test)} (2025)")

if len(X_test) == 0:
    print("\nNo 2025 data. Using 2024 as test set...")
    train_mask = df_valid['season'].isin([2022, 2023])
    test_mask = df_valid['season'] == 2024

    X_train = df_valid[train_mask][available_features]
    y_train = df_valid[train_mask][TARGET]
    X_test = df_valid[test_mask][available_features]
    y_test = df_valid[test_mask][TARGET]

    print(f"Revised Training: {len(X_train)} (2022-2023)")
    print(f"Revised Testing: {len(X_test)} (2024)")

# ============================================================
# BASELINE MODEL (All Features)
# ============================================================
print("\n" + "=" * 60)
print("BASELINE MODEL (All Features)")
print("=" * 60)

baseline_model = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print(f"\nBaseline MAE (all {len(available_features)} features): {baseline_mae:.2f} points")

# ============================================================
# RECURSIVE FEATURE ELIMINATION WITH CV
# ============================================================
print("\n" + "=" * 60)
print("RUNNING RFECV")
print("=" * 60)

print(f"\nThis will test feature subsets from 1 to {len(available_features)} features...")
print("Using TimeSeriesSplit with 5 folds...")
print("This may take a few minutes...\n")

# RandomForest estimator (exposes feature_importances_ for RFECV)
estimator = RandomForestRegressor(
    n_estimators=50,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# Custom scorer (negative MAE because sklearn maximizes)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# TimeSeriesSplit for proper temporal validation
tscv = TimeSeriesSplit(n_splits=5)

# RFECV
rfecv = RFECV(
    estimator=estimator,
    step=1,  # Remove 1 feature at a time
    cv=tscv,
    scoring=mae_scorer,
    min_features_to_select=3,  # Keep at least 3 features
    n_jobs=-1,
    verbose=1
)

print("Fitting RFECV...")
rfecv.fit(X_train, y_train)

# ============================================================
# RESULTS
# ============================================================
print("\n" + "=" * 60)
print("RFECV RESULTS")
print("=" * 60)

optimal_n_features = rfecv.n_features_
print(f"\nOptimal number of features: {optimal_n_features}")

# Get selected and dropped features
selected_mask = rfecv.support_
selected_features = [f for f, s in zip(available_features, selected_mask) if s]
dropped_features = [f for f, s in zip(available_features, selected_mask) if not s]

print(f"\n" + "=" * 60)
print("SELECTED FEATURES (KEEP)")
print("=" * 60)

# Get feature rankings
rankings = rfecv.ranking_
feature_ranks = list(zip(available_features, rankings, selected_mask))
feature_ranks.sort(key=lambda x: x[1])

print(f"\n{'Rank':<6} {'Feature':<35} {'Status':<10}")
print("-" * 55)

for feat, rank, selected in feature_ranks:
    status = "✅ KEEP" if selected else "❌ DROP"
    print(f"{rank:<6} {feat:<35} {status:<10}")

print(f"\n" + "=" * 60)
print("USELESS FEATURES (DROP)")
print("=" * 60)

print(f"\nFeatures to DROP ({len(dropped_features)}):")
for i, feat in enumerate(dropped_features, 1):
    print(f"  {i:2d}. {feat}")

print(f"\n" + "=" * 60)
print("BEST FEATURES (KEEP)")
print("=" * 60)

print(f"\nFeatures to KEEP ({len(selected_features)}):")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feat}")

# ============================================================
# EVALUATE OPTIMIZED MODEL
# ============================================================
print("\n" + "=" * 60)
print("OPTIMIZED MODEL PERFORMANCE")
print("=" * 60)

# Train with only selected features
X_train_opt = X_train[selected_features]
X_test_opt = X_test[selected_features]

# Use HistGradientBoosting for final model (our production model)
optimized_model = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

optimized_model.fit(X_train_opt, y_train)
optimized_pred = optimized_model.predict(X_test_opt)
optimized_mae = mean_absolute_error(y_test, optimized_pred)

print(f"\n{'Model':<30} {'Features':>10} {'MAE':>12}")
print("-" * 55)
print(f"{'Baseline (All Features)':<30} {len(available_features):>10} {baseline_mae:>12.2f}")
print(f"{'Optimized (RFECV)':<30} {len(selected_features):>10} {optimized_mae:>12.2f}")
print("-" * 55)

diff = optimized_mae - baseline_mae
if optimized_mae < baseline_mae:
    print(f"\n✅ IMPROVEMENT: {abs(diff):.2f} points better with {len(available_features) - len(selected_features)} fewer features!")
elif optimized_mae > baseline_mae:
    print(f"\n⚠️ Slight degradation: {abs(diff):.2f} points worse, but using {len(available_features) - len(selected_features)} fewer features")
else:
    print(f"\n🤝 Same performance with {len(available_features) - len(selected_features)} fewer features!")

# ============================================================
# CV SCORES ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("CV SCORE BY NUMBER OF FEATURES")
print("=" * 60)

cv_scores = -rfecv.cv_results_['mean_test_score']  # Negate because we used negative MAE
cv_std = rfecv.cv_results_['std_test_score']

# Find best score and its feature count
best_idx = np.argmin(cv_scores)
best_cv_mae = cv_scores[best_idx]
best_n_features = best_idx + rfecv.min_features_to_select

print(f"\nBest CV MAE: {best_cv_mae:.2f} points at {best_n_features} features")

# Show a sample of the CV curve
print(f"\n{'Features':<12} {'CV MAE':>10} {'Std':>8}")
print("-" * 32)

step = max(1, len(cv_scores) // 10)  # Show ~10 rows
for i in range(0, len(cv_scores), step):
    n_feat = i + rfecv.min_features_to_select
    print(f"{n_feat:<12} {cv_scores[i]:>10.2f} {cv_std[i]:>8.2f}")

# Show optimal
print("-" * 32)
print(f"{best_n_features:<12} {best_cv_mae:>10.2f} {'<-- BEST':>8}")

# ============================================================
# FEATURE IMPORTANCE (Optimized Model)
# ============================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Optimized Model)")
print("=" * 60)

# Use RandomForest to get feature importances (HistGradientBoosting doesn't expose them)
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_opt, y_train)
importance = rf_model.feature_importances_
feat_imp = list(zip(selected_features, importance))
feat_imp.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Feature':<35} {'Importance':>12}")
print("-" * 55)

for i, (feat, imp) in enumerate(feat_imp, 1):
    bar = '█' * int(imp * 50)
    print(f"{i:<6} {feat:<35} {imp:>12.4f} {bar}")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save selected features list
with open('optimized_features.txt', 'w') as f:
    f.write("RFECV Optimized Feature List\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Baseline MAE: {baseline_mae:.2f} points ({len(available_features)} features)\n")
    f.write(f"Optimized MAE: {optimized_mae:.2f} points ({len(selected_features)} features)\n")
    f.write(f"Improvement: {baseline_mae - optimized_mae:+.2f} points\n\n")
    f.write("SELECTED FEATURES:\n")
    for feat in selected_features:
        f.write(f"  - {feat}\n")
    f.write("\nDROPPED FEATURES:\n")
    for feat in dropped_features:
        f.write(f"  - {feat}\n")

print("\nSaved to 'optimized_features.txt'")

# Save as Python list for easy copy-paste
with open('optimal_feature_list.py', 'w') as f:
    f.write("# Optimal features from RFECV analysis\n")
    f.write("OPTIMAL_FEATURES = [\n")
    for feat in selected_features:
        f.write(f"    '{feat}',\n")
    f.write("]\n")

print("Saved to 'optimal_feature_list.py'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
Feature Optimization Complete:

Before:
  - Features: {len(available_features)}
  - MAE: {baseline_mae:.2f} points

After (RFECV):
  - Features: {len(selected_features)} ({len(dropped_features)} dropped)
  - MAE: {optimized_mae:.2f} points
  - Change: {optimized_mae - baseline_mae:+.2f} points

Dropped Features ({len(dropped_features)}):
""")

for feat in dropped_features[:10]:
    print(f"  ❌ {feat}")
if len(dropped_features) > 10:
    print(f"  ... and {len(dropped_features) - 10} more")

print(f"""
Top 5 Most Important Features:
""")
for i, (feat, imp) in enumerate(feat_imp[:5], 1):
    print(f"  {i}. {feat}: {imp:.4f}")

print("\n" + "=" * 60)
print("FEATURE OPTIMIZATION COMPLETE")
print("=" * 60)
