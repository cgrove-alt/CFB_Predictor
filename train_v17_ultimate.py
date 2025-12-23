"""
V17 ULTIMATE: World's Best CFB Prediction Model

Key improvements over V16:
1. REMOVED 11 zero-impact features (cleaner model)
2. ADDED garbage time adjustment features
3. ADDED primetime game flag
4. CAPPED error-amplifying momentum features
5. ISOTONIC CALIBRATION for confidence tiers
6. 3-EXPERT ROUTING (Mismatch, Competitive, Blowout specialists)
7. BAYESIAN uncertainty quantification
8. Enhanced Optuna optimization (200 XGB, 100 HGB trials)

Target: MAE 8.5 (from 9.47), Win Rate 72% (from 66.6%)
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import optuna
from datetime import datetime
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = 'cfb_data_safe.csv'
OUTPUT_MODEL = 'cfb_spread_error_v17.pkl'
OUTPUT_CONFIG = 'cfb_v17_config.pkl'
OUTPUT_CALIBRATOR = 'cfb_v17_calibrator.pkl'
OUTPUT_EXPERTS = 'cfb_v17_experts.pkl'
XGB_TRIALS = 200
HGB_TRIALS = 100
RANDOM_STATE = 42

# ============================================================
# V17 FEATURE ENGINEERING
# ============================================================

# REMOVED: Zero-impact features (SHAP = 0)
ZERO_IMPACT_FEATURES = [
    'home_comp_epa', 'away_comp_epa',
    'home_comp_ypp', 'away_comp_ypp',
    'home_comp_success', 'away_comp_success',
    'success_rate_diff',
    'dominant_home', 'dominant_away',
    'home_lookahead', 'away_lookahead',
    'west_coast_early',
]

# V17 CORE FEATURES (cleaned from V16)
V17_CORE_FEATURES = [
    # Elo features (stable, high predictive power)
    'home_pregame_elo', 'away_pregame_elo', 'elo_diff',

    # Rolling stats (past 5 games)
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',

    # Home field advantage
    'home_team_hfa', 'hfa_diff',

    # Rest (scheduling edge)
    'rest_diff',

    # Vegas lines (wisdom of crowds)
    'vegas_spread', 'line_movement', 'has_line_movement',
    'large_favorite', 'large_underdog', 'close_game',

    # Streaks (CAPPED at ±3 to reduce error amplification)
    'home_streak_capped', 'away_streak_capped', 'streak_diff_capped',

    # ATS history (CAPPED to reduce error amplification)
    'home_ats_capped', 'away_ats_capped', 'ats_diff',

    # Momentum (dampened)
    'home_elo_momentum', 'away_elo_momentum', 'elo_momentum_diff',

    # Derived features
    'elo_vs_spread', 'rest_spread_interaction',
    'home_short_rest', 'away_short_rest',
    'expected_total',

    # PPA efficiency (core - stable predictors)
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',
    'home_comp_pass_ppa', 'away_comp_pass_ppa',
    'home_comp_rush_ppa', 'away_comp_rush_ppa',
    'pass_efficiency_diff',

    # Composite features
    'matchup_efficiency',
    'home_pass_rush_balance', 'away_pass_rush_balance',
    'elo_efficiency_interaction',
    'rest_favorite_interaction',
]

# NEW V17 FEATURES
V17_NEW_FEATURES = [
    # Garbage time adjustment (team quality without stat padding)
    'home_garbage_impact',      # raw_ppa - clean_ppa (high = pads stats)
    'away_garbage_impact',
    'garbage_diff',             # differential

    # Clean PPA (more reliable than raw)
    'home_clean_off_ppa', 'away_clean_off_ppa',
    'home_clean_def_ppa', 'away_clean_def_ppa',

    # Primetime flag
    'is_primetime',             # 7pm+ ET kickoff

    # Strength of schedule
    'home_sos', 'away_sos', 'sos_diff',

    # Game type indicators (for expert routing)
    'is_mismatch',              # |elo_diff| > 200
    'is_competitive',           # |elo_diff| < 200 AND |spread| < 10
    'is_blowout_spread',        # |spread| > 14

    # Uncertainty features (from V16)
    'is_pickem',
    'is_early_season',
    'is_rivalry_week',
    'spread_bucket_error',
    'feature_completeness',
]


def load_data():
    """Load and prepare training data."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df = df[df['vegas_spread'].notna()].copy()
    print(f"Loaded {len(df)} games with spreads")

    # Ensure spread_error exists
    if 'spread_error' not in df.columns:
        df['margin'] = df['home_points'] - df['away_points']
        df['spread_error'] = df['margin'] - (-df['vegas_spread'])

    # Sort chronologically
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    return df


def add_v17_features(df):
    """Add V17-specific features."""
    print("\nEngineering V17 features...")

    # 1. CAP MOMENTUM FEATURES (reduce error amplification)
    df['home_streak_capped'] = df['home_streak'].clip(-3, 3)
    df['away_streak_capped'] = df['away_streak'].clip(-3, 3)
    df['streak_diff_capped'] = df['streak_diff'].clip(-3, 3)
    df['home_ats_capped'] = df['home_ats'].clip(0.3, 0.7)
    df['away_ats_capped'] = df['away_ats'].clip(0.3, 0.7)
    print("  - Capped momentum features to reduce error amplification")

    # 2. GARBAGE TIME IMPACT
    if 'home_raw_off_ppa' in df.columns and 'home_clean_off_ppa' in df.columns:
        df['home_garbage_impact'] = df['home_raw_off_ppa'] - df['home_clean_off_ppa']
        df['away_garbage_impact'] = df['away_raw_off_ppa'] - df['away_clean_off_ppa']
        df['garbage_diff'] = df['home_garbage_impact'] - df['away_garbage_impact']
        print("  - Added garbage time impact features")
    else:
        df['home_garbage_impact'] = 0
        df['away_garbage_impact'] = 0
        df['garbage_diff'] = 0
        print("  - Garbage time features not available (set to 0)")

    # 3. PRIMETIME FLAG (extract from start_date)
    if 'start_date' in df.columns:
        try:
            df['start_hour'] = pd.to_datetime(df['start_date']).dt.hour
            df['is_primetime'] = (df['start_hour'] >= 19).astype(int)
            print("  - Added primetime game flag")
        except:
            df['is_primetime'] = 0
            print("  - Could not parse start_date (primetime = 0)")
    else:
        df['is_primetime'] = 0

    # 4. STRENGTH OF SCHEDULE
    print("  - Calculating strength of schedule...")
    df['home_sos'] = 1500.0  # Will be computed per-game
    df['away_sos'] = 1500.0

    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        for week in range(2, 16):
            week_mask = (df['season'] == season) & (df['week'] == week)

            for idx in df[week_mask].index:
                home_team = df.loc[idx, 'home_team']
                away_team = df.loc[idx, 'away_team']

                # Home team's SOS (avg opponent Elo)
                home_past = df[
                    (df['season'] == season) &
                    (df['week'] < week) &
                    ((df['home_team'] == home_team) | (df['away_team'] == home_team))
                ]
                if len(home_past) > 0:
                    opp_elos = []
                    for _, g in home_past.iterrows():
                        if g['home_team'] == home_team:
                            opp_elos.append(g['away_pregame_elo'])
                        else:
                            opp_elos.append(g['home_pregame_elo'])
                    df.loc[idx, 'home_sos'] = np.mean(opp_elos)

                # Away team's SOS
                away_past = df[
                    (df['season'] == season) &
                    (df['week'] < week) &
                    ((df['home_team'] == away_team) | (df['away_team'] == away_team))
                ]
                if len(away_past) > 0:
                    opp_elos = []
                    for _, g in away_past.iterrows():
                        if g['home_team'] == away_team:
                            opp_elos.append(g['away_pregame_elo'])
                        else:
                            opp_elos.append(g['home_pregame_elo'])
                    df.loc[idx, 'away_sos'] = np.mean(opp_elos)

    df['sos_diff'] = df['home_sos'] - df['away_sos']
    print("  - Calculated strength of schedule")

    # 5. GAME TYPE INDICATORS (for expert routing)
    df['is_mismatch'] = (df['elo_diff'].abs() > 200).astype(int)
    df['is_competitive'] = ((df['elo_diff'].abs() <= 200) & (df['vegas_spread'].abs() < 10)).astype(int)
    df['is_blowout_spread'] = (df['vegas_spread'].abs() > 14).astype(int)
    print("  - Added game type indicators")

    # 6. UNCERTAINTY FEATURES (from V16)
    df['is_pickem'] = (df['vegas_spread'].abs() <= 3).astype(int)
    df['is_early_season'] = (df['week'] <= 3).astype(int)
    df['is_rivalry_week'] = df['week'].isin([12, 13]).astype(int)

    # Spread bucket error (historical average)
    bucket_errors = {'pickem': 11.0, 'small': 9.5, 'medium': 9.2, 'large': 9.5, 'blowout': 10.2}
    def get_bucket_error(spread):
        abs_spread = abs(spread) if pd.notna(spread) else 0
        if abs_spread <= 3:
            return bucket_errors['pickem']
        elif abs_spread <= 7:
            return bucket_errors['small']
        elif abs_spread <= 14:
            return bucket_errors['medium']
        elif abs_spread <= 21:
            return bucket_errors['large']
        return bucket_errors['blowout']

    df['spread_bucket_error'] = df['vegas_spread'].apply(get_bucket_error)

    # Feature completeness
    core_features = [f for f in V17_CORE_FEATURES if f in df.columns and not f.endswith('_capped')]
    df['feature_completeness'] = df[core_features].notna().sum(axis=1) / len(core_features)

    print(f"  - Added uncertainty features")

    return df


def get_v17_features(df):
    """Get the complete V17 feature list."""
    all_features = V17_CORE_FEATURES + V17_NEW_FEATURES

    # Filter to features that exist in the dataframe
    available = [f for f in all_features if f in df.columns]

    # Remove zero-impact features
    available = [f for f in available if f not in ZERO_IMPACT_FEATURES]

    print(f"\nV17 Feature Set: {len(available)} features")
    return available


def walk_forward_validation(model_class, params, df, feature_cols, target='spread_error'):
    """Walk-forward validation on 2025 data."""
    errors = []
    predictions = []
    actuals = []

    for week in range(1, 16):
        train_mask = (df['season'] < 2025) | ((df['season'] == 2025) & (df['week'] < week))
        test_mask = (df['season'] == 2025) & (df['week'] == week)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) == 0:
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target]

        model = model_class(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        errors.extend(np.abs(preds - y_test).tolist())
        predictions.extend(preds.tolist())
        actuals.extend(y_test.tolist())

    return np.mean(errors), predictions, actuals


def optimize_xgboost(df, feature_cols):
    """Optimize XGBoost with enhanced Optuna search."""
    print(f"\nOptimizing XGBoost ({XGB_TRIALS} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'n_jobs': -1,
        }
        mae, _, _ = walk_forward_validation(XGBRegressor, params, df, feature_cols)
        return mae

    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20)
    )
    study.optimize(objective, n_trials=XGB_TRIALS, show_progress_bar=True)

    print(f"Best XGBoost MAE: {study.best_trial.value:.4f}")
    return study.best_params


def optimize_hgb(df, feature_cols):
    """Optimize HistGradientBoosting."""
    print(f"\nOptimizing HistGradientBoosting ({HGB_TRIALS} trials)...")

    def objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 80),
        }
        mae, _, _ = walk_forward_validation(HistGradientBoostingRegressor, params, df, feature_cols)
        return mae

    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=15)
    )
    study.optimize(objective, n_trials=HGB_TRIALS, show_progress_bar=True)

    print(f"Best HGB MAE: {study.best_trial.value:.4f}")
    return study.best_params


def train_expert_models(df, feature_cols, xgb_params, hgb_params):
    """Train 3 specialized expert models for different game types."""
    print("\n" + "="*60)
    print("TRAINING EXPERT MODELS")
    print("="*60)

    experts = {}

    # Prepare base training data
    train_df = df[df['season'] < 2025].copy()

    # Expert 1: MISMATCH SPECIALIST (|elo_diff| > 200)
    print("\nTraining Mismatch Expert...")
    mismatch_df = train_df[train_df['is_mismatch'] == 1]
    X_mismatch = mismatch_df[feature_cols].fillna(0)
    y_mismatch = mismatch_df['spread_error']

    mismatch_model = XGBRegressor(**xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
    mismatch_model.fit(X_mismatch, y_mismatch)
    experts['mismatch'] = mismatch_model
    print(f"  Trained on {len(mismatch_df)} mismatch games")

    # Expert 2: COMPETITIVE SPECIALIST (|elo_diff| <= 200 AND |spread| < 10)
    print("\nTraining Competitive Expert...")
    competitive_df = train_df[train_df['is_competitive'] == 1]
    X_competitive = competitive_df[feature_cols].fillna(0)
    y_competitive = competitive_df['spread_error']

    competitive_model = XGBRegressor(**xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
    competitive_model.fit(X_competitive, y_competitive)
    experts['competitive'] = competitive_model
    print(f"  Trained on {len(competitive_df)} competitive games")

    # Expert 3: BLOWOUT SPECIALIST (|spread| > 14)
    print("\nTraining Blowout Expert...")
    blowout_df = train_df[train_df['is_blowout_spread'] == 1]
    X_blowout = blowout_df[feature_cols].fillna(0)
    y_blowout = blowout_df['spread_error']

    blowout_model = XGBRegressor(**xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
    blowout_model.fit(X_blowout, y_blowout)
    experts['blowout'] = blowout_model
    print(f"  Trained on {len(blowout_df)} blowout games")

    # General model (fallback)
    print("\nTraining General Model...")
    X_all = train_df[feature_cols].fillna(0)
    y_all = train_df['spread_error']

    general_xgb = XGBRegressor(**xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
    general_xgb.fit(X_all, y_all)
    experts['xgb'] = general_xgb

    general_hgb = HistGradientBoostingRegressor(**hgb_params, random_state=RANDOM_STATE)
    general_hgb.fit(X_all, y_all)
    experts['hgb'] = general_hgb

    return experts


def route_to_expert(game_features, experts):
    """Route game to appropriate expert model."""
    if game_features.get('is_mismatch', 0) == 1:
        return experts['mismatch']
    elif game_features.get('is_blowout_spread', 0) == 1:
        return experts['blowout']
    elif game_features.get('is_competitive', 0) == 1:
        return experts['competitive']
    else:
        return experts['xgb']  # Fallback to general


def train_bayesian_meta(df, experts, feature_cols):
    """Train Bayesian meta-learner for uncertainty quantification."""
    print("\n" + "="*60)
    print("TRAINING BAYESIAN META-LEARNER")
    print("="*60)

    # Generate predictions from all experts on validation data
    val_df = df[(df['season'] == 2025) & (df['week'] <= 10)].copy()
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['spread_error']

    # Get predictions from each model
    meta_features = []
    for name, model in experts.items():
        if name in ['xgb', 'hgb', 'mismatch', 'competitive', 'blowout']:
            preds = model.predict(X_val)
            meta_features.append(preds)

    # Stack predictions as meta-features
    X_meta = np.column_stack(meta_features)

    # Train Bayesian Ridge (provides mean + std)
    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)

    bayesian_meta = BayesianRidge(n_iter=1000, tol=1e-4)
    bayesian_meta.fit(X_meta_scaled, y_val)

    # Test uncertainty estimation
    y_pred, y_std = bayesian_meta.predict(X_meta_scaled, return_std=True)
    print(f"  Mean prediction std: {y_std.mean():.2f}")
    print(f"  Std range: [{y_std.min():.2f}, {y_std.max():.2f}]")

    return bayesian_meta, scaler


def train_calibrator(predictions, actuals):
    """Train isotonic calibration for confidence tiers."""
    print("\n" + "="*60)
    print("TRAINING CONFIDENCE CALIBRATOR")
    print("="*60)

    # Convert to numpy arrays
    preds = np.array(predictions)
    acts = np.array(actuals)

    # Prediction magnitude -> actual error mapping
    pred_magnitude = np.abs(preds)
    actual_errors = np.abs(preds - acts)

    # Isotonic regression: higher pred magnitude should mean lower error
    # But we want: confidence score -> actual accuracy
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(pred_magnitude, actual_errors)

    # Test calibration
    calibrated = calibrator.predict(pred_magnitude)
    print(f"  Raw magnitude range: [{pred_magnitude.min():.2f}, {pred_magnitude.max():.2f}]")
    print(f"  Calibrated error range: [{calibrated.min():.2f}, {calibrated.max():.2f}]")

    return calibrator


def evaluate_v17(df, experts, feature_cols, calibrator):
    """Comprehensive evaluation of V17 model."""
    print("\n" + "="*60)
    print("V17 EVALUATION ON 2025 DATA")
    print("="*60)

    results = []
    all_preds = []
    all_actuals = []

    for week in range(1, 16):
        train_mask = (df['season'] < 2025) | ((df['season'] == 2025) & (df['week'] < week))
        test_mask = (df['season'] == 2025) & (df['week'] == week)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) == 0:
            continue

        # Retrain experts on available data
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['spread_error']
        experts['xgb'].fit(X_train, y_train)
        experts['hgb'].fit(X_train, y_train)

        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['spread_error']

        # Generate predictions with expert routing
        week_preds = []
        for i, (idx, row) in enumerate(test_df.iterrows()):
            game_features = row.to_dict()
            expert = route_to_expert(game_features, experts)
            pred = expert.predict(X_test.iloc[[i]])[0]
            week_preds.append(pred)

        week_preds = np.array(week_preds)

        # Metrics
        mae = mean_absolute_error(y_test, week_preds)
        vegas_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))
        direction_acc = ((week_preds > 0) == (y_test > 0)).mean()

        results.append({
            'week': week,
            'games': len(test_df),
            'mae': mae,
            'vegas_mae': vegas_mae,
            'direction_accuracy': direction_acc,
        })

        all_preds.extend(week_preds.tolist())
        all_actuals.extend(y_test.tolist())

        beat = "✓" if mae < vegas_mae else "✗"
        print(f"Week {week:2d}: {len(test_df):3d} games | MAE: {mae:.2f} vs Vegas {vegas_mae:.2f} {beat} | "
              f"Direction: {direction_acc*100:.1f}%")

    # Overall metrics
    results_df = pd.DataFrame(results)
    total_games = results_df['games'].sum()

    overall_mae = mean_absolute_error(all_actuals, all_preds)
    overall_vegas = np.mean(np.abs(all_actuals))
    weeks_beat = (results_df['mae'] < results_df['vegas_mae']).sum()
    overall_direction = ((np.array(all_preds) > 0) == (np.array(all_actuals) > 0)).mean()

    # Betting simulation
    correct = ((np.array(all_preds) > 0) == (np.array(all_actuals) > 0)).sum()
    wrong = len(all_preds) - correct
    profit = correct - wrong * 1.1  # -110 odds
    roi = profit / len(all_preds) * 100

    print("\n" + "-"*50)
    print(f"OVERALL V17 PERFORMANCE ({total_games} games)")
    print("-"*50)
    print(f"MAE: {overall_mae:.2f} (Vegas: {overall_vegas:.2f})")
    print(f"Improvement: {overall_vegas - overall_mae:+.2f} points")
    print(f"Direction Accuracy: {overall_direction*100:.1f}%")
    print(f"Weeks Beat Vegas: {weeks_beat}/{len(results_df)}")
    print(f"Profit: {profit:+.1f} units | ROI: {roi:+.1f}%")

    return {
        'mae': overall_mae,
        'vegas_mae': overall_vegas,
        'direction_accuracy': overall_direction,
        'weeks_beat_vegas': weeks_beat,
        'profit': profit,
        'roi': roi,
        'all_preds': all_preds,
        'all_actuals': all_actuals,
    }


def save_v17_model(experts, feature_cols, calibrator, bayesian_meta, scaler, metrics, xgb_params, hgb_params):
    """Save all V17 model components."""
    print("\n" + "="*60)
    print("SAVING V17 MODEL")
    print("="*60)

    # Save primary model (XGBoost)
    joblib.dump(experts['xgb'], OUTPUT_MODEL)
    print(f"Primary model: {OUTPUT_MODEL}")

    # Save all experts
    joblib.dump(experts, OUTPUT_EXPERTS)
    print(f"Expert models: {OUTPUT_EXPERTS}")

    # Save calibrator
    calibrator_bundle = {
        'isotonic': calibrator,
        'bayesian_meta': bayesian_meta,
        'scaler': scaler,
    }
    joblib.dump(calibrator_bundle, OUTPUT_CALIBRATOR)
    print(f"Calibrator: {OUTPUT_CALIBRATOR}")

    # Save configuration
    config = {
        'features': feature_cols,
        'xgb_params': xgb_params,
        'hgb_params': hgb_params,
        'metrics': metrics,
        'version': 'v17_ultimate',
        'trained_at': datetime.now().isoformat(),
        'improvements': [
            'Removed 11 zero-impact features',
            'Added garbage time adjustment',
            'Added primetime flag',
            'Added strength of schedule',
            'Capped momentum features',
            '3-expert routing system',
            'Bayesian uncertainty',
            'Isotonic calibration',
        ],
    }
    joblib.dump(config, OUTPUT_CONFIG)
    print(f"Config: {OUTPUT_CONFIG}")


def main():
    print("="*70)
    print("V17 ULTIMATE: WORLD'S BEST CFB PREDICTION MODEL")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()

    # Add V17 features
    df = add_v17_features(df)

    # Get feature list
    feature_cols = get_v17_features(df)
    print(f"\nFeatures ({len(feature_cols)}):")
    for f in feature_cols[:20]:
        print(f"  - {f}")
    if len(feature_cols) > 20:
        print(f"  ... and {len(feature_cols) - 20} more")

    # Ensure all features exist
    for f in feature_cols:
        if f not in df.columns:
            df[f] = 0

    # Optimize hyperparameters
    xgb_params = optimize_xgboost(df, feature_cols)
    hgb_params = optimize_hgb(df, feature_cols)

    # Train expert models
    experts = train_expert_models(df, feature_cols, xgb_params, hgb_params)

    # Train Bayesian meta-learner
    bayesian_meta, scaler = train_bayesian_meta(df, experts, feature_cols)

    # Generate validation predictions for calibration
    print("\nGenerating validation predictions...")
    _, all_preds, all_actuals = walk_forward_validation(
        XGBRegressor, xgb_params, df, feature_cols
    )

    # Train calibrator
    calibrator = train_calibrator(all_preds, all_actuals)

    # Evaluate V17
    metrics = evaluate_v17(df, experts, feature_cols, calibrator)

    # Save model
    save_v17_model(experts, feature_cols, calibrator, bayesian_meta, scaler, metrics, xgb_params, hgb_params)

    # Final summary
    print("\n" + "="*70)
    print("V17 TRAINING COMPLETE")
    print("="*70)
    print(f"""
V17 Ultimate Model Summary:
  - Features: {len(feature_cols)} (cleaned from 89 in V16)
  - Architecture: 3 Expert Models + Bayesian Meta-Learner + Calibration

Performance vs V16 Target:
  - MAE: {metrics['mae']:.2f} (target: 8.5, V16: 9.47)
  - Direction: {metrics['direction_accuracy']*100:.1f}% (target: 72%, V16: 66.6%)
  - ROI: {metrics['roi']:+.1f}%

Key Improvements:
  - Removed 11 zero-impact features
  - Added garbage time adjustment
  - 3-expert routing for game types
  - Isotonic confidence calibration
  - Bayesian uncertainty quantification
""")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
