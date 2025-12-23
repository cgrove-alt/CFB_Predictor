#!/usr/bin/env python3
"""
Diagnostic Training Script for Sharp Sports Predictor.

This script establishes a rigorous baseline with detailed metrics:
- MAE by season, week, conference
- Error distribution statistics
- Feature importance rankings
- Prediction vs actual analysis
- Data quality assessment
"""

import os
import sys
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def print_header(text, char="="):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f" {text}")
    print(f"{char * 70}")


def print_subheader(text):
    """Print a formatted subheader."""
    print(f"\n--- {text} ---")


def analyze_data_quality(df, feature_cols):
    """Analyze data quality and missing values."""
    print_header("DATA QUALITY ANALYSIS")

    # Overall stats
    total_cells = len(df) * len(feature_cols)
    missing_cells = df[feature_cols].isna().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100

    print(f"\nTotal games: {len(df)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Total cells: {total_cells:,}")
    print(f"Missing cells: {missing_cells:,} ({missing_pct:.2f}%)")

    # Missing by feature
    print_subheader("Missing Values by Feature")
    missing_by_feat = df[feature_cols].isna().sum().sort_values(ascending=False)
    for feat, count in missing_by_feat.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"  {feat}: {count} ({pct:.1f}%)")

    # Coverage by season
    print_subheader("Data Coverage by Season")
    for season in sorted(df['season'].unique()):
        season_df = df[df['season'] == season]
        season_missing = season_df[feature_cols].isna().sum().sum()
        season_total = len(season_df) * len(feature_cols)
        coverage = (1 - season_missing / season_total) * 100
        print(f"  {season}: {len(season_df)} games, {coverage:.1f}% feature coverage")

    # Target variable stats
    print_subheader("Target Variable (Margin) Statistics")
    margin = df['Margin'].dropna()
    print(f"  Count: {len(margin)}")
    print(f"  Mean: {margin.mean():.2f}")
    print(f"  Std: {margin.std():.2f}")
    print(f"  Min: {margin.min():.0f}")
    print(f"  Max: {margin.max():.0f}")
    print(f"  Median: {margin.median():.0f}")

    # Outliers (blowouts)
    outlier_threshold = margin.std() * 3
    outliers = df[df['Margin'].abs() > outlier_threshold]
    print(f"\n  Outliers (|margin| > {outlier_threshold:.1f}): {len(outliers)} games")

    return {
        'total_games': len(df),
        'missing_pct': missing_pct,
        'outlier_count': len(outliers),
        'margin_std': margin.std()
    }


def train_baseline_model(X_train, y_train, X_test, y_test, params):
    """Train baseline HGB model and return detailed results."""
    model = HistGradientBoostingRegressor(
        max_iter=params.get('max_iter', 100),
        max_depth=params.get('max_depth', 3),
        learning_rate=params.get('learning_rate', 0.05),
        l2_regularization=params.get('l2_regularization', 0.1),
        random_state=42,
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return model, train_pred, test_pred


def analyze_errors(y_true, y_pred, df_test, model_name="Model"):
    """Detailed error analysis."""
    print_header(f"ERROR ANALYSIS: {model_name}")

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    # Basic stats
    print_subheader("Error Distribution")
    print(f"  MAE: {abs_errors.mean():.2f}")
    print(f"  RMSE: {np.sqrt((errors**2).mean()):.2f}")
    print(f"  Mean Error (Bias): {errors.mean():+.2f}")
    print(f"  Std Error: {errors.std():.2f}")
    print(f"  Median Abs Error: {np.median(abs_errors):.2f}")
    print(f"  Max Error: {abs_errors.max():.2f}")

    # Percentiles
    print_subheader("Error Percentiles")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(abs_errors, p):.2f}")

    # Accuracy within thresholds
    print_subheader("Accuracy Within Thresholds")
    for threshold in [3, 7, 10, 14, 21]:
        pct = (abs_errors <= threshold).mean() * 100
        print(f"  Within {threshold} pts: {pct:.1f}%")

    # Error by predicted margin range
    print_subheader("Error by Predicted Margin Range")
    pred_bins = [(-100, -21), (-21, -14), (-14, -7), (-7, 0), (0, 7), (7, 14), (14, 21), (21, 100)]
    for low, high in pred_bins:
        mask = (y_pred >= low) & (y_pred < high)
        if mask.sum() > 0:
            bin_mae = abs_errors[mask].mean()
            print(f"  [{low:+3d}, {high:+3d}): n={mask.sum():4d}, MAE={bin_mae:.2f}")

    # Error by actual margin range (how hard are blowouts to predict?)
    print_subheader("Error by Actual Margin Range")
    actual_bins = [(-100, -28), (-28, -14), (-14, -7), (-7, 7), (7, 14), (14, 28), (28, 100)]
    for low, high in actual_bins:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            bin_mae = abs_errors[mask].mean()
            print(f"  [{low:+3d}, {high:+3d}): n={mask.sum():4d}, MAE={bin_mae:.2f}")

    # Error by season
    if 'season' in df_test.columns:
        print_subheader("Error by Season")
        for season in sorted(df_test['season'].unique()):
            mask = df_test['season'] == season
            if mask.sum() > 0:
                season_mae = abs_errors[mask.values].mean()
                print(f"  {season}: n={mask.sum():4d}, MAE={season_mae:.2f}")

    # Error by week
    if 'week' in df_test.columns:
        print_subheader("Error by Week")
        weeks = sorted(df_test['week'].unique())
        for week in weeks[:15]:  # First 15 weeks
            mask = df_test['week'] == week
            if mask.sum() > 0:
                week_mae = abs_errors[mask.values].mean()
                print(f"  Week {week:2d}: n={mask.sum():4d}, MAE={week_mae:.2f}")

    return {
        'mae': abs_errors.mean(),
        'rmse': np.sqrt((errors**2).mean()),
        'bias': errors.mean(),
        'std': errors.std(),
        'median': np.median(abs_errors),
    }


def analyze_feature_importance(model, feature_names, X_train, y_train):
    """Analyze feature importance using permutation importance."""
    print_header("FEATURE IMPORTANCE ANALYSIS")

    # For HGB, we can use feature_importances_ if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print_subheader("Feature Importance (Impurity-based)")
        for i, idx in enumerate(indices):
            print(f"  {i+1:2d}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Also do permutation importance for more robust estimate
    print_subheader("Permutation Importance (5 permutations)")
    from sklearn.inspection import permutation_importance

    perm_result = permutation_importance(model, X_train, y_train,
                                          n_repeats=5, random_state=42, n_jobs=-1)
    perm_importances = perm_result.importances_mean
    perm_indices = np.argsort(perm_importances)[::-1]

    for i, idx in enumerate(perm_indices):
        print(f"  {i+1:2d}. {feature_names[idx]}: {perm_importances[idx]:.4f} (+/- {perm_result.importances_std[idx]:.4f})")

    return dict(zip(feature_names, perm_importances))


def analyze_feature_correlations(X, feature_names):
    """Analyze feature correlations."""
    print_header("FEATURE CORRELATION ANALYSIS")

    corr_matrix = X.corr()

    # Find highly correlated pairs
    print_subheader("Highly Correlated Feature Pairs (|r| > 0.7)")
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for f1, f2, corr in high_corr_pairs[:15]:  # Top 15
        print(f"  {f1} <-> {f2}: {corr:.3f}")

    if not high_corr_pairs:
        print("  No pairs with |r| > 0.7 found")

    return corr_matrix


def cross_validate_model(model, X, y, n_splits=5):
    """Perform time-series cross-validation."""
    print_header("CROSS-VALIDATION RESULTS")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    mae_scores = -scores

    print(f"\nTime-Series CV ({n_splits} folds):")
    for i, score in enumerate(mae_scores):
        print(f"  Fold {i+1}: MAE = {score:.2f}")

    print(f"\n  Mean MAE: {mae_scores.mean():.2f}")
    print(f"  Std MAE: {mae_scores.std():.2f}")
    print(f"  Min MAE: {mae_scores.min():.2f}")
    print(f"  Max MAE: {mae_scores.max():.2f}")

    return mae_scores


def save_diagnostic_report(results, filename='diagnostic_report.txt'):
    """Save diagnostic results to file."""
    with open(filename, 'w') as f:
        f.write("SHARP SPORTS PREDICTOR - DIAGNOSTIC REPORT\n")
        f.write("=" * 60 + "\n\n")

        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print(f"\nDiagnostic report saved to '{filename}'")


def main():
    """Main diagnostic function."""
    print_header("SHARP SPORTS PREDICTOR - DIAGNOSTIC ANALYSIS", "=")

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    print_header("LOADING DATA")

    print("Loading cfb_data_smart.csv...")
    df = pd.read_csv('cfb_data_smart.csv')
    print(f"Total games loaded: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # ==========================================================================
    # DEFINE ALL AVAILABLE FEATURES
    # ==========================================================================

    # Current V8 features
    CURRENT_FEATURES = [
        'home_pregame_elo', 'away_pregame_elo',
        'home_last5_score_avg', 'away_last5_score_avg',
        'home_last5_defense_avg', 'away_last5_defense_avg',
        'home_comp_off_ppa', 'away_comp_off_ppa',
        'home_comp_def_ppa', 'away_comp_def_ppa',
        'net_epa',
        'home_team_hfa', 'away_team_hfa',
        'home_rest_days', 'away_rest_days',
        'rest_advantage',
        'rest_diff',
        'elo_diff',
    ]

    # UNUSED features that exist in data but aren't used
    UNUSED_FEATURES = [
        # Adjusted EPA
        'home_adj_off_epa', 'away_adj_off_epa',
        'home_adj_def_epa', 'away_adj_def_epa',
        'adj_net_epa',
        # Interaction features
        'epa_elo_interaction',
        'pass_efficiency_diff',
        'matchup_advantage',
        'success_diff',
        # Advanced stats
        'home_comp_pass_ppa', 'away_comp_pass_ppa',
        'home_comp_rush_ppa', 'away_comp_rush_ppa',
        'home_comp_epa', 'away_comp_epa',
        'home_comp_success', 'away_comp_success',
        'home_comp_ypp', 'away_comp_ypp',
        # Situational
        'west_coast_early',
        'home_lookahead', 'away_lookahead',
    ]

    # Filter to available features
    current_available = [f for f in CURRENT_FEATURES if f in df.columns]
    unused_available = [f for f in UNUSED_FEATURES if f in df.columns]
    all_available = current_available + unused_available

    print(f"\nCurrent features available: {len(current_available)}/{len(CURRENT_FEATURES)}")
    print(f"Unused features available: {len(unused_available)}/{len(UNUSED_FEATURES)}")
    print(f"Total available: {len(all_available)}")

    # ==========================================================================
    # DATA QUALITY ANALYSIS
    # ==========================================================================
    quality_results = analyze_data_quality(df, all_available)

    # ==========================================================================
    # PREPARE DATA
    # ==========================================================================
    print_header("PREPARING DATA")

    # Filter to games with valid margins
    df_valid = df[df['Margin'].notna()].copy()
    print(f"Games with valid Margin: {len(df_valid)}")

    # Train/Test split by season
    TRAIN_SEASONS = [2022, 2023, 2024]
    TEST_SEASON = 2025

    train_mask = df_valid['season'].isin(TRAIN_SEASONS)
    test_mask = df_valid['season'] == TEST_SEASON

    # Fallback if no 2025 data
    if test_mask.sum() == 0:
        print("\nNo 2025 data available. Using 2024 as test set.")
        TRAIN_SEASONS = [2022, 2023]
        TEST_SEASON = 2024
        train_mask = df_valid['season'].isin(TRAIN_SEASONS)
        test_mask = df_valid['season'] == TEST_SEASON

    print(f"Training seasons: {TRAIN_SEASONS}")
    print(f"Test season: {TEST_SEASON}")
    print(f"Training samples: {train_mask.sum()}")
    print(f"Test samples: {test_mask.sum()}")

    # Create feature matrices
    X_train_current = df_valid[train_mask][current_available]
    X_test_current = df_valid[test_mask][current_available]
    y_train = df_valid[train_mask]['Margin']
    y_test = df_valid[test_mask]['Margin']

    X_train_all = df_valid[train_mask][all_available]
    X_test_all = df_valid[test_mask][all_available]

    df_test = df_valid[test_mask].copy()

    # ==========================================================================
    # FEATURE CORRELATION ANALYSIS
    # ==========================================================================
    corr_matrix = analyze_feature_correlations(X_train_current, current_available)

    # ==========================================================================
    # BASELINE MODEL (CURRENT FEATURES)
    # ==========================================================================
    print_header("BASELINE MODEL: CURRENT FEATURES")

    params = {
        'max_iter': 100,
        'max_depth': 3,
        'learning_rate': 0.05,
        'l2_regularization': 0.1,
    }

    print(f"\nParameters: {params}")
    print(f"Features: {len(current_available)}")

    model_current, train_pred_current, test_pred_current = train_baseline_model(
        X_train_current, y_train, X_test_current, y_test, params
    )

    error_results_current = analyze_errors(
        y_test.values, test_pred_current, df_test, "Baseline (Current Features)"
    )

    # Feature importance
    importance_current = analyze_feature_importance(
        model_current, current_available, X_train_current, y_train
    )

    # Cross-validation
    cv_scores_current = cross_validate_model(
        HistGradientBoostingRegressor(**params, random_state=42),
        X_train_current, y_train
    )

    # ==========================================================================
    # EXPANDED MODEL (ALL AVAILABLE FEATURES)
    # ==========================================================================
    print_header("EXPANDED MODEL: ALL AVAILABLE FEATURES")

    print(f"\nFeatures: {len(all_available)}")
    print("New features being tested:")
    for f in unused_available:
        print(f"  + {f}")

    model_all, train_pred_all, test_pred_all = train_baseline_model(
        X_train_all, y_train, X_test_all, y_test, params
    )

    error_results_all = analyze_errors(
        y_test.values, test_pred_all, df_test, "Expanded (All Features)"
    )

    # Feature importance for expanded model
    importance_all = analyze_feature_importance(
        model_all, all_available, X_train_all, y_train
    )

    # Cross-validation for expanded model
    cv_scores_all = cross_validate_model(
        HistGradientBoostingRegressor(**params, random_state=42),
        X_train_all, y_train
    )

    # ==========================================================================
    # COMPARISON
    # ==========================================================================
    print_header("MODEL COMPARISON")

    print(f"\n{'Metric':<25} {'Current':<15} {'Expanded':<15} {'Diff':<15}")
    print("-" * 70)

    metrics = [
        ('Test MAE', error_results_current['mae'], error_results_all['mae']),
        ('Test RMSE', error_results_current['rmse'], error_results_all['rmse']),
        ('Bias', error_results_current['bias'], error_results_all['bias']),
        ('CV MAE (mean)', cv_scores_current.mean(), cv_scores_all.mean()),
        ('CV MAE (std)', cv_scores_current.std(), cv_scores_all.std()),
    ]

    for name, current, expanded in metrics:
        diff = expanded - current
        sign = "+" if diff > 0 else ""
        print(f"{name:<25} {current:<15.2f} {expanded:<15.2f} {sign}{diff:<15.2f}")

    improvement = error_results_current['mae'] - error_results_all['mae']
    print(f"\nExpanded model {'IMPROVES' if improvement > 0 else 'DEGRADES'} MAE by {abs(improvement):.2f} points")

    # ==========================================================================
    # HYPERPARAMETER SENSITIVITY
    # ==========================================================================
    print_header("HYPERPARAMETER SENSITIVITY ANALYSIS")

    best_mae = float('inf')
    best_params = params.copy()

    # Test different max_depth values
    print_subheader("Testing max_depth")
    for depth in [3, 4, 5, 6, 7, 8]:
        test_params = params.copy()
        test_params['max_depth'] = depth
        model, _, pred = train_baseline_model(X_train_all, y_train, X_test_all, y_test, test_params)
        mae = mean_absolute_error(y_test, pred)
        marker = " <-- BEST" if mae < best_mae else ""
        print(f"  depth={depth}: MAE={mae:.2f}{marker}")
        if mae < best_mae:
            best_mae = mae
            best_params = test_params.copy()

    # Test different learning_rate values
    print_subheader("Testing learning_rate")
    for lr in [0.03, 0.05, 0.08, 0.1, 0.12, 0.15]:
        test_params = best_params.copy()
        test_params['learning_rate'] = lr
        model, _, pred = train_baseline_model(X_train_all, y_train, X_test_all, y_test, test_params)
        mae = mean_absolute_error(y_test, pred)
        marker = " <-- BEST" if mae < best_mae else ""
        print(f"  lr={lr}: MAE={mae:.2f}{marker}")
        if mae < best_mae:
            best_mae = mae
            best_params = test_params.copy()

    # Test different max_iter values
    print_subheader("Testing max_iter")
    for iters in [100, 150, 200, 300, 500]:
        test_params = best_params.copy()
        test_params['max_iter'] = iters
        model, _, pred = train_baseline_model(X_train_all, y_train, X_test_all, y_test, test_params)
        mae = mean_absolute_error(y_test, pred)
        marker = " <-- BEST" if mae < best_mae else ""
        print(f"  max_iter={iters}: MAE={mae:.2f}{marker}")
        if mae < best_mae:
            best_mae = mae
            best_params = test_params.copy()

    print_subheader("Best Parameters Found")
    print(f"  {best_params}")
    print(f"  Best MAE: {best_mae:.2f}")
    print(f"  Improvement over baseline: {error_results_current['mae'] - best_mae:.2f} points")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_header("DIAGNOSTIC SUMMARY")

    results = {
        'baseline_mae': error_results_current['mae'],
        'expanded_mae': error_results_all['mae'],
        'best_tuned_mae': best_mae,
        'best_params': best_params,
        'cv_mae_mean_current': cv_scores_current.mean(),
        'cv_mae_std_current': cv_scores_current.std(),
        'cv_mae_mean_expanded': cv_scores_all.mean(),
        'cv_mae_std_expanded': cv_scores_all.std(),
        'data_quality_missing_pct': quality_results['missing_pct'],
        'training_samples': train_mask.sum(),
        'test_samples': test_mask.sum(),
        'current_features': len(current_available),
        'expanded_features': len(all_available),
    }

    print(f"""
BASELINE (Current {len(current_available)} features):
  - Test MAE: {error_results_current['mae']:.2f} points
  - CV MAE: {cv_scores_current.mean():.2f} (+/- {cv_scores_current.std():.2f})

EXPANDED (All {len(all_available)} features):
  - Test MAE: {error_results_all['mae']:.2f} points
  - CV MAE: {cv_scores_all.mean():.2f} (+/- {cv_scores_all.std():.2f})
  - Improvement: {error_results_current['mae'] - error_results_all['mae']:.2f} points

BEST TUNED:
  - Test MAE: {best_mae:.2f} points
  - Parameters: depth={best_params['max_depth']}, lr={best_params['learning_rate']}, iters={best_params['max_iter']}
  - Total improvement: {error_results_current['mae'] - best_mae:.2f} points

KEY INSIGHTS:
  - Data missing: {quality_results['missing_pct']:.1f}% of feature values
  - Training samples: {train_mask.sum()}
  - Test samples: {test_mask.sum()}
""")

    # Save report
    save_diagnostic_report(results)

    print_header("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()
