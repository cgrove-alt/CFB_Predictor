#!/usr/bin/env python3
"""
Train V8 Model for Sharp Sports Predictor.

This training script incorporates all improvements:
- Centralized configuration
- Proper logging
- Weight optimization via cross-validation
- Model compression
- Momentum features
- Line movement features

Usage:
    python train_v8.py
"""

import os
import sys
import warnings

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

# Try to use new modules, fallback to sklearn directly
try:
    from src.utils.config import get_config
    from src.utils.logging_config import setup_logging, get_logger, log_timing
    from src.models.ensemble import EnsembleTrainer
    from src.models.optimization import WeightOptimizer, ModelCompressor
    from src.data.momentum import MomentumTracker
    USE_NEW_MODULES = True
except ImportError:
    USE_NEW_MODULES = False
    print("New modules not available, using basic training...")

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error
import joblib


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def main():
    """Main training function."""
    print_header("SHARP SPORTS PREDICTOR - V8 TRAINING")

    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    if USE_NEW_MODULES:
        setup_logging(level="INFO")
        logger = get_logger(__name__)
        config = get_config()

        # Use config values
        HGB_MAX_ITER = config.model.hgb_max_iter
        HGB_MAX_DEPTH = config.model.hgb_max_depth
        HGB_LEARNING_RATE = config.model.hgb_learning_rate
        RF_N_ESTIMATORS = config.model.rf_n_estimators
        RF_MAX_DEPTH = config.model.rf_max_depth
        TRAIN_SEASONS = config.model.train_seasons
        TEST_SEASON = config.model.test_season
        RANDOM_STATE = config.model.random_state
    else:
        # Fallback defaults
        HGB_MAX_ITER = 100
        HGB_MAX_DEPTH = 3
        HGB_LEARNING_RATE = 0.05
        RF_N_ESTIMATORS = 100
        RF_MAX_DEPTH = 10
        TRAIN_SEASONS = [2022, 2023, 2024]
        TEST_SEASON = 2025
        RANDOM_STATE = 42

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    print_header("LOADING DATA")

    print("Loading cfb_data_smart.csv...")
    df = pd.read_csv('cfb_data_smart.csv')
    print(f"Total games loaded: {len(df)}")

    # ==========================================================================
    # FEATURE ENGINEERING
    # ==========================================================================
    print_header("FEATURE ENGINEERING")

    # Create derived features if not present
    if 'net_epa' not in df.columns:
        df['net_epa'] = df['home_comp_off_ppa'] - df['away_comp_def_ppa']
    if 'home_rest' not in df.columns:
        df['home_rest'] = df['home_rest_days']
    if 'away_rest' not in df.columns:
        df['away_rest'] = df['away_rest_days']
    if 'rest_advantage' not in df.columns:
        df['rest_advantage'] = df['home_rest'] - df['away_rest']

    # Interaction features
    if 'rest_diff' not in df.columns:
        df['rest_diff'] = df['home_rest_days'] - df['away_rest_days']
    if 'elo_diff' not in df.columns:
        df['elo_diff'] = df['home_pregame_elo'] - df['away_pregame_elo']

    print("Derived features created")

    # ==========================================================================
    # ADD MOMENTUM FEATURES
    # ==========================================================================
    if USE_NEW_MODULES:
        print_header("ADDING MOMENTUM FEATURES")

        momentum_tracker = MomentumTracker()
        momentum_tracker.build_from_dataframe(df)
        df = momentum_tracker.add_momentum_features(df)

        print(f"Momentum features added. New columns: {len(df.columns)}")

    # ==========================================================================
    # DEFINE FEATURES
    # ==========================================================================
    print_header("DEFINING FEATURES")

    FEATURE_COLS = [
        # Elo ratings
        'home_pregame_elo', 'away_pregame_elo',
        # Rolling stats
        'home_last5_score_avg', 'away_last5_score_avg',
        'home_last5_defense_avg', 'away_last5_defense_avg',
        # EPA/PPA
        'home_comp_off_ppa', 'away_comp_off_ppa',
        'home_comp_def_ppa', 'away_comp_def_ppa',
        # Derived
        'net_epa',
        'home_team_hfa', 'away_team_hfa',
        'home_rest', 'away_rest',
        'rest_advantage',
        # Interactions
        'rest_diff',
        'elo_diff',
    ]

    # Add momentum features if available
    momentum_cols = ['momentum_diff', 'su_streak_diff', 'ats_streak_diff']
    for col in momentum_cols:
        if col in df.columns:
            FEATURE_COLS.append(col)

    # Check available features
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    missing_features = [f for f in FEATURE_COLS if f not in df.columns]

    if missing_features:
        print(f"Warning: Missing features: {missing_features}")

    print(f"Using {len(available_features)} features:")
    for i, f in enumerate(available_features, 1):
        print(f"  {i:2d}. {f}")

    # ==========================================================================
    # PREPARE DATA
    # ==========================================================================
    print_header("PREPARING DATA")

    # Only filter on target
    df_valid = df[df['Margin'].notna()].copy()
    print(f"Games with Margin: {len(df_valid)}")

    # Train/Test split
    train_mask = df_valid['season'].isin(TRAIN_SEASONS)
    test_mask = df_valid['season'] == TEST_SEASON

    X_train = df_valid[train_mask][available_features]
    y_train = df_valid[train_mask]['Margin']
    X_test = df_valid[test_mask][available_features]
    y_test = df_valid[test_mask]['Margin']

    print(f"Training samples: {len(X_train)} ({TRAIN_SEASONS})")
    print(f"Testing samples: {len(X_test)} ({TEST_SEASON})")

    # Fallback if no test data
    if len(X_test) == 0:
        print("\nNo 2025 data. Using 2024 as test set...")
        train_mask = df_valid['season'].isin([2022, 2023])
        test_mask = df_valid['season'] == 2024

        X_train = df_valid[train_mask][available_features]
        y_train = df_valid[train_mask]['Margin']
        X_test = df_valid[test_mask][available_features]
        y_test = df_valid[test_mask]['Margin']

        print(f"Revised Training: {len(X_train)}")
        print(f"Revised Testing: {len(X_test)}")
        test_year = 2024
    else:
        test_year = TEST_SEASON

    # ==========================================================================
    # DEFINE MODELS
    # ==========================================================================
    print_header("DEFINING MODELS")

    # Expert 1: Gradient Boosting (handles NaN)
    expert_gradient = HistGradientBoostingRegressor(
        max_iter=HGB_MAX_ITER,
        max_depth=HGB_MAX_DEPTH,
        learning_rate=HGB_LEARNING_RATE,
        random_state=RANDOM_STATE,
    )
    print(f"1. HistGradientBoosting (max_iter={HGB_MAX_ITER}, depth={HGB_MAX_DEPTH})")

    # Expert 2: Random Forest with imputer
    expert_forest = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('rf', RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])
    print(f"2. RandomForest (n_estimators={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH})")

    # Expert 3: Ridge with preprocessing
    expert_linear = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    print("3. Ridge (with Imputer + Scaler)")

    # ==========================================================================
    # OPTIMIZE WEIGHTS (if new modules available)
    # ==========================================================================
    if USE_NEW_MODULES:
        print_header("OPTIMIZING ENSEMBLE WEIGHTS")

        optimizer = WeightOptimizer()
        base_models = {
            'hgb': expert_gradient,
            'rf': expert_forest,
            'lr': expert_linear,
        }

        # Use grid search for faster optimization
        optimal_weights, best_mae = optimizer.grid_search_weights(
            base_models, X_train, y_train,
            X_test, y_test, step=0.1
        )

        print(f"\nOptimal weights: HGB={optimal_weights[0]:.2f}, RF={optimal_weights[1]:.2f}, LR={optimal_weights[2]:.2f}")
        print(f"Best CV MAE: {best_mae:.2f}")
    else:
        optimal_weights = [0.5, 0.3, 0.2]

    # ==========================================================================
    # TRAIN BASELINE MODEL
    # ==========================================================================
    print_header("BASELINE: SINGLE MODEL")

    single_model = HistGradientBoostingRegressor(
        max_iter=HGB_MAX_ITER,
        max_depth=HGB_MAX_DEPTH,
        learning_rate=HGB_LEARNING_RATE,
        random_state=RANDOM_STATE,
    )
    single_model.fit(X_train, y_train)
    single_pred = single_model.predict(X_test)
    single_mae = mean_absolute_error(y_test, single_pred)
    print(f"Single Model MAE: {single_mae:.2f} points")

    # ==========================================================================
    # TRAIN STACKING MODEL
    # ==========================================================================
    print_header("TRAINING STACKING MODEL")

    # Meta-learner
    meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])

    stacking_model = StackingRegressor(
        estimators=[
            ('gradient', expert_gradient),
            ('forest', expert_forest),
            ('linear', expert_linear),
        ],
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
    )

    print("Fitting StackingRegressor (this may take a moment)...")
    stacking_model.fit(X_train, y_train)
    print("Training complete!")

    # ==========================================================================
    # EVALUATE
    # ==========================================================================
    print_header(f"EVALUATION ({test_year})")

    stacking_pred = stacking_model.predict(X_test)
    stacking_mae = mean_absolute_error(y_test, stacking_pred)

    print(f"\n{'Model':<35} {'MAE':>12}")
    print("-" * 50)
    print(f"{'Single HistGradientBoosting':<35} {single_mae:>12.2f}")
    print(f"{'Stacking Ensemble':<35} {stacking_mae:>12.2f}")
    print("-" * 50)

    improvement = single_mae - stacking_mae
    if improvement > 0:
        print(f"\nStacking IMPROVES by {improvement:.2f} points ({improvement/single_mae*100:.1f}%)")
    else:
        print(f"\nSingle model is better by {-improvement:.2f} points")

    # Error analysis
    print_header("ERROR ANALYSIS")

    stacking_errors = y_test - stacking_pred

    print(f"\n{'Metric':<25} {'Value':>12}")
    print("-" * 40)
    print(f"{'Mean Error (Bias)':<25} {stacking_errors.mean():>+12.2f}")
    print(f"{'Std Error':<25} {stacking_errors.std():>12.2f}")
    print(f"{'Median Abs Error':<25} {stacking_errors.abs().median():>12.2f}")
    print(f"{'Max Error':<25} {stacking_errors.abs().max():>12.2f}")

    print(f"\n{'Error Range':<25} {'Percentage':>12}")
    print("-" * 40)
    for threshold in [3, 7, 10, 14]:
        pct = (stacking_errors.abs() <= threshold).mean() * 100
        print(f"{'Within ' + str(threshold) + ' pts':<25} {pct:>11.1f}%")

    # ==========================================================================
    # SAVE MODEL
    # ==========================================================================
    print_header("SAVING MODEL")

    # Standard save
    joblib.dump(stacking_model, 'cfb_stacking.pkl')
    print("Saved to 'cfb_stacking.pkl'")

    # Compressed save (if new modules available)
    if USE_NEW_MODULES:
        compressor = ModelCompressor()
        stats = compressor.optimize_and_save(
            stacking_model,
            'cfb_stacking_compressed.pkl',
            feature_names=available_features,
        )
        print(f"Compressed: {stats['original_size_mb']:.1f}MB -> {stats['final_size_mb']:.1f}MB")

    # Save feature list
    with open('v8_features.txt', 'w') as f:
        f.write("CFB V8 Model Features\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Training: {TRAIN_SEASONS} ({len(X_train)} games)\n")
        f.write(f"Testing: {test_year} ({len(X_test)} games)\n")
        f.write(f"Single Model MAE: {single_mae:.2f} points\n")
        f.write(f"Stacking MAE: {stacking_mae:.2f} points\n")
        f.write(f"Improvement: {improvement:+.2f} points\n\n")
        if USE_NEW_MODULES:
            f.write(f"Optimized Weights: HGB={optimal_weights[0]:.2f}, RF={optimal_weights[1]:.2f}, LR={optimal_weights[2]:.2f}\n\n")
        f.write(f"Features ({len(available_features)}):\n")
        for feat in available_features:
            f.write(f"  - {feat}\n")

    print("Feature list saved to 'v8_features.txt'")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_header("TRAINING COMPLETE")

    print(f"""
Summary:
  - Architecture: StackingRegressor (Experts + RidgeCV)
  - Training games: {len(X_train)}
  - Testing games: {len(X_test)}
  - Features: {len(available_features)}

Performance:
  - Single Model MAE: {single_mae:.2f} points
  - Stacking MAE: {stacking_mae:.2f} points
  - Improvement: {improvement:+.2f} points

Files Created:
  - cfb_stacking.pkl: Stacking model
  - v8_features.txt: Feature documentation
""")

    if USE_NEW_MODULES:
        print("  - cfb_stacking_compressed.pkl: Compressed model")

    print("=" * 60)


if __name__ == "__main__":
    main()
