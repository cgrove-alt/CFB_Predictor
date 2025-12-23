"""
Verify True MAE for CFB Betting Model.

Performs proper train/test split:
- Train: All games before 2025
- Test: 2025 games only

Includes data leakage checks to ensure valid evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
# Using available columns in cfb_data_smart.csv
FEATURE_COLS = [
    'home_pregame_elo', 'away_pregame_elo',
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg'
]

TARGET_COL = 'Margin'  # Home Points - Away Points

# Columns that would cause leakage if included
LEAKAGE_COLS = [
    'Margin', 'margin', 'home_points', 'away_points',
    'home_score', 'away_score', 'total_points', 'winner'
]

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("VERIFY TRUE MAE")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")

# Check for 'season' column
if 'season' not in df.columns:
    print("\nERROR: 'season' column not found!")
    print(f"Available columns: {list(df.columns)[:20]}...")
    exit(1)

# ============================================================
# DATA LEAKAGE CHECK
# ============================================================
print("\n" + "=" * 60)
print("DATA LEAKAGE CHECK")
print("=" * 60)

print("\nFeatures being used:")
for f in FEATURE_COLS:
    print(f"  - {f}")

print("\nChecking for leakage columns in features...")
leakage_found = []
for col in FEATURE_COLS:
    col_lower = col.lower()
    for leak in LEAKAGE_COLS:
        if leak.lower() in col_lower:
            leakage_found.append(col)
            break

if leakage_found:
    print(f"\n  LEAKAGE DETECTED!")
    print(f"  Problematic columns: {leakage_found}")
    print("  Removing these from features...")
    FEATURE_COLS = [f for f in FEATURE_COLS if f not in leakage_found]
    print(f"  Remaining features: {FEATURE_COLS}")
else:
    print("  CLEAN - No leakage columns found in features")

# Check if target exists
if TARGET_COL not in df.columns:
    # Try to calculate it
    if 'home_points' in df.columns and 'away_points' in df.columns:
        df['Margin'] = df['home_points'] - df['away_points']
        print(f"\nCalculated '{TARGET_COL}' from home_points - away_points")
    else:
        print(f"\nERROR: Target column '{TARGET_COL}' not found!")
        exit(1)

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT")
print("=" * 60)

# Check available seasons
print(f"\nSeasons in data: {sorted(df['season'].unique())}")

# Train: All games before 2025
train_mask = df['season'] < 2025
# Test: 2025 games only
test_mask = df['season'] == 2025

# Filter to rows with complete features
all_cols = FEATURE_COLS + [TARGET_COL]
df_valid = df.dropna(subset=all_cols)

train_df = df_valid[df_valid['season'] < 2025]
test_df = df_valid[df_valid['season'] == 2025]

print(f"\nTrain Set (< 2025): {len(train_df)} games")
print(f"Test Set (2025):    {len(test_df)} games")

if len(test_df) == 0:
    print("\nWARNING: No 2025 games with complete data!")
    print("Checking what 2025 data is available...")

    df_2025 = df[df['season'] == 2025]
    print(f"Total 2025 games: {len(df_2025)}")

    if len(df_2025) > 0:
        # Check which features are missing
        for col in all_cols:
            missing = df_2025[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} missing ({missing/len(df_2025)*100:.1f}%)")

        # Try with available features only
        print("\nAttempting with available features only...")
        available_features = [f for f in FEATURE_COLS if df_2025[f].notna().sum() > 0]
        print(f"Available features: {available_features}")

        if len(available_features) > 0:
            # Use subset of features
            FEATURE_COLS = available_features
            all_cols = FEATURE_COLS + [TARGET_COL]
            df_valid = df.dropna(subset=all_cols)
            train_df = df_valid[df_valid['season'] < 2025]
            test_df = df_valid[df_valid['season'] == 2025]
            print(f"\nRevised Train Set: {len(train_df)} games")
            print(f"Revised Test Set:  {len(test_df)} games")

if len(test_df) == 0:
    print("\n" + "=" * 60)
    print("NO 2025 DATA AVAILABLE - USING 2024 AS TEST SET")
    print("=" * 60)

    # Fall back to 2024 as test set
    train_df = df_valid[df_valid['season'] < 2024]
    test_df = df_valid[df_valid['season'] == 2024]

    print(f"\nFallback Train Set (< 2024): {len(train_df)} games")
    print(f"Fallback Test Set (2024):    {len(test_df)} games")
    test_year = 2024
else:
    test_year = 2025

# ============================================================
# PREPARE DATA
# ============================================================
print("\n" + "=" * 60)
print("PREPARING DATA")
print("=" * 60)

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]
X_test = test_df[FEATURE_COLS]
y_test = test_df[TARGET_COL]

print(f"\nTraining features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Final feature check
print(f"\nFinal features used:")
for f in FEATURE_COLS:
    print(f"  - {f}")

# ============================================================
# TRAIN MODEL
# ============================================================
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

print("\nTraining HistGradientBoostingRegressor...")
model = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)
print("Training complete!")

# ============================================================
# PREDICT & EVALUATE
# ============================================================
print("\n" + "=" * 60)
print("PREDICTION & EVALUATION")
print("=" * 60)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

print(f"\n" + "=" * 60)
print(f"TRUE MAE for {test_year} Season: {mae:.2f} points")
print(f"Total Games Tested: {len(y_test)}")
print("=" * 60)

# ============================================================
# ADDITIONAL METRICS
# ============================================================
print("\n" + "=" * 60)
print("ADDITIONAL METRICS")
print("=" * 60)

# Root Mean Square Error
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"\nRMSE: {rmse:.2f} points")

# Mean Error (bias check)
mean_error = np.mean(y_test - y_pred)
print(f"Mean Error (Bias): {mean_error:+.2f} points")

# Median Absolute Error
median_ae = np.median(np.abs(y_test - y_pred))
print(f"Median Absolute Error: {median_ae:.2f} points")

# Percentage within certain thresholds
within_3 = (np.abs(y_test - y_pred) <= 3).mean() * 100
within_7 = (np.abs(y_test - y_pred) <= 7).mean() * 100
within_14 = (np.abs(y_test - y_pred) <= 14).mean() * 100

print(f"\nPredictions within 3 pts: {within_3:.1f}%")
print(f"Predictions within 7 pts: {within_7:.1f}%")
print(f"Predictions within 14 pts: {within_14:.1f}%")

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (First 10)")
print("=" * 60)

sample = test_df.head(10).copy()
sample['Predicted'] = y_pred[:10]
sample['Error'] = sample[TARGET_COL] - sample['Predicted']

print(f"\n{'Game':<40} {'Actual':>8} {'Pred':>8} {'Error':>8}")
print("-" * 66)

for _, row in sample.iterrows():
    game = f"{row.get('away_team', 'Away')[:15]} @ {row.get('home_team', 'Home')[:15]}"
    actual = row[TARGET_COL]
    pred = row['Predicted']
    error = row['Error']
    print(f"{game:<40} {actual:>8.0f} {pred:>8.1f} {error:>+8.1f}")

# ============================================================
# VEGAS COMPARISON (if available)
# ============================================================
if 'spread_line' in test_df.columns:
    print("\n" + "=" * 60)
    print("COMPARISON VS VEGAS")
    print("=" * 60)

    # Only compare games with spread data
    vegas_df = test_df[test_df['spread_line'].notna()].copy()

    if len(vegas_df) > 0:
        vegas_df['Predicted'] = model.predict(vegas_df[FEATURE_COLS])

        # Model MAE
        model_mae = mean_absolute_error(vegas_df[TARGET_COL], vegas_df['Predicted'])

        # Vegas MAE (spread is from home perspective, negative = home favored)
        # Predicted margin from Vegas = -spread_line
        vegas_pred = -vegas_df['spread_line']
        vegas_mae = mean_absolute_error(vegas_df[TARGET_COL], vegas_pred)

        print(f"\nGames with Vegas lines: {len(vegas_df)}")
        print(f"\nModel MAE:  {model_mae:.2f} points")
        print(f"Vegas MAE:  {vegas_mae:.2f} points")
        print(f"Difference: {model_mae - vegas_mae:+.2f} points")

        if model_mae < vegas_mae:
            print(f"\nModel beats Vegas by {vegas_mae - model_mae:.2f} points!")
        else:
            print(f"\nVegas beats Model by {vegas_mae - model_mae:.2f} points")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
