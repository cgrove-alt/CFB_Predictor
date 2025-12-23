"""
Train a HistGradientBoosting model using V3 features.
Features: Elo, Talent, Rolling Stats, EPA, Wind Speed, Net EPA (Schematic Mismatch)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import pickle
import joblib

# Load data
print("Loading data...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\nEngineering V3 features...")

# Net EPA: Schematic Mismatch Feature
# How well does home offense match up vs away defense (and vice versa)
# Using comp_off_ppa (competitive offensive PPA) and comp_def_ppa (competitive defensive PPA)
df['net_epa_home'] = df['home_comp_off_ppa'] - df['away_comp_def_ppa']
df['net_epa_away'] = df['away_comp_off_ppa'] - df['home_comp_def_ppa']
df['net_epa'] = df['net_epa_home'] - df['net_epa_away']  # Positive = Home advantage

print(f"  Net EPA calculated")
print(f"  Net EPA range: {df['net_epa'].min():.2f} to {df['net_epa'].max():.2f}")

# ============================================================
# DEFINE V3 FEATURES
# ============================================================
feature_cols = [
    # Elo ratings (power rankings)
    'home_pregame_elo', 'away_pregame_elo',

    # Rolling stats (recent form - points)
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',

    # EPA/PPA - Points Per Attempt (efficiency)
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',

    # Clean EPA (garbage-time filtered)
    'home_comp_epa', 'away_comp_epa',

    # Home Field Advantage
    'home_team_hfa', 'hfa_diff',

    # Rest advantage
    'rest_advantage',

    # Net EPA - Schematic Mismatch
    'net_epa'
]

print(f"\nV3 Features ({len(feature_cols)} total):")
for i, f in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {f}")

# ============================================================
# PREPARE DATA
# ============================================================
print(f"\nTotal games before filtering: {len(df)}")

# Drop rows with missing feature values
df_filtered = df.dropna(subset=feature_cols)
print(f"Total games after filtering: {len(df_filtered)}")

# Define features and target
X = df_filtered[feature_cols]
y = df_filtered['Margin']

# Time-series split: Train on 2022-2023, Test on 2024
train_mask = df_filtered['season'].isin([2022, 2023])
test_mask = df_filtered['season'] == 2024

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTraining samples: {len(X_train)} (2022-2023)")
print(f"Testing samples: {len(X_test)} (2024)")

# ============================================================
# TRAIN MODEL
# ============================================================
print("\nTraining HistGradientBoostingRegressor (V3)...")
model = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*60)
print(f"MEAN ABSOLUTE ERROR (MAE): {mae:.2f} points")
print("="*60)

# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================
print("\nCalculating Permutation Importance...")
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

print("\n" + "="*60)
print("FEATURE IMPORTANCE (V3 Model)")
print("="*60)

# Sort by importance
sorted_idx = perm_importance.importances_mean.argsort()[::-1]
total_importance = sum(max(0, perm_importance.importances_mean[i]) for i in range(len(feature_cols)))

for rank, idx in enumerate(sorted_idx, 1):
    mean_imp = perm_importance.importances_mean[idx]
    std_imp = perm_importance.importances_std[idx]
    pct = (max(0, mean_imp) / total_importance * 100) if total_importance > 0 else 0
    bar = '█' * int(mean_imp * 10)
    print(f"{rank:2d}. {feature_cols[idx]:25} {mean_imp:7.4f} ± {std_imp:.4f} ({pct:5.1f}%) {bar}")

# ============================================================
# FEATURE CATEGORY SUMMARY
# ============================================================
print("\n" + "="*60)
print("FEATURE CATEGORY BREAKDOWN")
print("="*60)

categories = {
    'Elo Ratings': ['home_pregame_elo', 'away_pregame_elo'],
    'Rolling Stats': ['home_last5_score_avg', 'away_last5_score_avg',
                      'home_last5_defense_avg', 'away_last5_defense_avg'],
    'PPA Efficiency': ['home_comp_off_ppa', 'away_comp_off_ppa',
                       'home_comp_def_ppa', 'away_comp_def_ppa'],
    'Clean EPA': ['home_comp_epa', 'away_comp_epa'],
    'Home Field': ['home_team_hfa', 'hfa_diff'],
    'Rest': ['rest_advantage'],
    'Net EPA': ['net_epa']
}

for cat_name, cat_features in categories.items():
    cat_importance = sum(
        max(0, perm_importance.importances_mean[feature_cols.index(f)])
        for f in cat_features if f in feature_cols
    )
    cat_pct = (cat_importance / total_importance * 100) if total_importance > 0 else 0
    bar = '█' * int(cat_pct / 2)
    print(f"{cat_name:15} {cat_pct:5.1f}% {bar}")

# ============================================================
# SAVE MODELS
# ============================================================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# Save pickle format
with open('cfb_predictor_v3.model', 'wb') as f:
    pickle.dump(model, f)
print(f"Saved to 'cfb_predictor_v3.model'")

# Save joblib format (what app.py uses)
joblib.dump(model, 'cfb_smart_model.pkl')
print("Saved to 'cfb_smart_model.pkl'")

# Save feature list
with open('v3_features.txt', 'w') as f:
    f.write('\n'.join(feature_cols))
print("Feature list saved to 'v3_features.txt'")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("V3 MODEL TRAINING COMPLETE")
print("="*60)
print(f"MAE: {mae:.2f} points")
print(f"Features: {len(feature_cols)}")
print(f"Training games: {len(X_train)}")
print(f"Test games: {len(X_test)}")
print("="*60)
