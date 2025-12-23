"""
V18 Game-Type Router - Route games to specialized expert models.

Based on analysis showing different game types have vastly different predictability:
- Elite vs Elite: 7.50 MAE, 75.6% accuracy (BEST)
- Mismatch: 7.93 MAE, 76.4% accuracy
- Average vs Average: 11.35 MAE, 55.4% accuracy (WORST - 51% of games!)
- Pick-em: Hardest to predict

This router classifies games and routes them to appropriate expert models.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = 'cfb_data_safe.csv'
CONFIG_FILE = 'cfb_v16_config.pkl'
OUTPUT_MODEL = 'cfb_v18_experts.pkl'
OUTPUT_CONFIG = 'cfb_v18_config.pkl'
OPTUNA_TRIALS = 50  # Per expert model
RANDOM_STATE = 42

# Game type definitions
GAME_TYPES = {
    'elite': 'Both teams Elo > 1550',
    'mismatch': 'Elo diff > 250',
    'competitive': 'Elo diff < 150 AND spread < 10',
    'blowout': 'Spread > 17',
    'early_season': 'Week <= 3',
    'general': 'Everything else'
}


def classify_game_type(row):
    """
    Classify a game into one of the expert categories.
    Priority order matters - first match wins.
    """
    home_elo = row.get('home_pregame_elo', 1500)
    away_elo = row.get('away_pregame_elo', 1500)
    elo_diff = abs(home_elo - away_elo)
    spread = abs(row.get('vegas_spread', 0))
    week = row.get('week', 10)

    # Priority 1: Early season (cold start problem)
    if week <= 3:
        return 'early_season'

    # Priority 2: Elite matchups (both strong teams)
    if home_elo > 1550 and away_elo > 1550:
        return 'elite'

    # Priority 3: Mismatches (large Elo gap)
    if elo_diff > 250:
        return 'mismatch'

    # Priority 4: Blowout spreads
    if spread > 17:
        return 'blowout'

    # Priority 5: Competitive games (close matchups)
    if elo_diff < 150 and spread < 10:
        return 'competitive'

    # Default: General model
    return 'general'


def load_data():
    """Load and prepare data."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    # Only use games with spreads
    df = df[df['vegas_spread'].notna()].copy()

    # Calculate spread_error if missing
    if 'spread_error' not in df.columns:
        df['spread_error'] = df['Margin'] - (-df['vegas_spread'])

    # Classify each game
    df['game_type'] = df.apply(classify_game_type, axis=1)

    # Sort chronologically
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    print(f"Total games: {len(df)}")
    print("\nGame type distribution:")
    for gt, count in df['game_type'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {gt}: {count} ({pct:.1f}%)")

    return df


def get_features():
    """Get V18 feature list (based on V16 with improvements)."""
    try:
        config = joblib.load(CONFIG_FILE)
        features = config.get('features', [])
        print(f"Loaded {len(features)} features from {CONFIG_FILE}")
        return features
    except:
        # Fallback features
        return [
            'home_pregame_elo', 'away_pregame_elo', 'elo_diff',
            'home_last5_score_avg', 'away_last5_score_avg',
            'home_last5_defense_avg', 'away_last5_defense_avg',
            'home_team_hfa', 'hfa_diff', 'rest_diff',
            'vegas_spread', 'line_movement',
            'home_streak', 'away_streak', 'streak_diff',
            'home_ats', 'away_ats', 'ats_diff',
            'home_elo_momentum', 'away_elo_momentum',
            'home_scoring_trend', 'away_scoring_trend',
            'large_favorite', 'large_underdog', 'close_game',
            'elo_vs_spread', 'expected_total',
            'home_comp_off_ppa', 'away_comp_off_ppa',
            'home_comp_def_ppa', 'away_comp_def_ppa',
            'matchup_efficiency',
        ]


def optimize_expert_model(X_train, y_train, X_val, y_val, game_type, n_trials=OPTUNA_TRIALS):
    """Optimize hyperparameters for a single expert model."""
    print(f"\n  Optimizing {game_type} expert ({n_trials} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        }

        model = XGBRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return mean_absolute_error(y_val, y_pred)

    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"    Best MAE: {study.best_value:.4f}")
    return study.best_params


def train_expert_models(df, feature_cols):
    """Train specialized expert models for each game type."""
    print("\n" + "=" * 70)
    print("TRAINING V18 EXPERT MODELS")
    print("=" * 70)

    # Split data
    train_mask = df['season'] < 2025
    test_mask = df['season'] == 2025

    experts = {}
    expert_params = {}

    # Train general model first (will be used as fallback)
    print("\n[1] Training GENERAL model (fallback)...")
    train_df = df[train_mask]
    val_mask = train_df['season'] == 2024
    train_inner = train_df[~val_mask]
    val_inner = train_df[val_mask]

    X_train = train_inner[feature_cols].fillna(0)
    y_train = train_inner['spread_error']
    X_val = val_inner[feature_cols].fillna(0)
    y_val = val_inner['spread_error']

    general_params = optimize_expert_model(X_train, y_train, X_val, y_val, 'general')

    general_model = XGBRegressor(
        **general_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    # Train on all pre-2025 data
    X_full = train_df[feature_cols].fillna(0)
    y_full = train_df['spread_error']
    general_model.fit(X_full, y_full)

    experts['general'] = general_model
    expert_params['general'] = general_params

    # Train specialized experts
    game_types_to_train = ['elite', 'mismatch', 'competitive', 'blowout', 'early_season']

    for i, game_type in enumerate(game_types_to_train, 2):
        print(f"\n[{i}] Training {game_type.upper()} expert...")

        # Get games of this type
        type_mask = train_df['game_type'] == game_type
        type_games = train_df[type_mask]

        if len(type_games) < 50:
            print(f"  Insufficient data ({len(type_games)} games), using general model")
            experts[game_type] = general_model
            expert_params[game_type] = general_params
            continue

        # Split for validation
        val_mask = type_games['season'] == 2024
        train_inner = type_games[~val_mask]
        val_inner = type_games[val_mask]

        if len(val_inner) < 10:
            print(f"  Insufficient validation data ({len(val_inner)} games), using all for training")
            # Use a portion for validation
            split_idx = int(len(type_games) * 0.8)
            train_inner = type_games.iloc[:split_idx]
            val_inner = type_games.iloc[split_idx:]

        X_train = train_inner[feature_cols].fillna(0)
        y_train = train_inner['spread_error']
        X_val = val_inner[feature_cols].fillna(0)
        y_val = val_inner['spread_error']

        print(f"  Training: {len(train_inner)} games, Validation: {len(val_inner)} games")

        params = optimize_expert_model(X_train, y_train, X_val, y_val, game_type)

        model = XGBRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )

        # Train on all data of this type
        X_full = type_games[feature_cols].fillna(0)
        y_full = type_games['spread_error']
        model.fit(X_full, y_full)

        experts[game_type] = model
        expert_params[game_type] = params

    return experts, expert_params


def train_meta_learner(df, experts, feature_cols):
    """Train a Bayesian meta-learner to combine expert predictions."""
    print("\n" + "=" * 70)
    print("TRAINING META-LEARNER")
    print("=" * 70)

    train_mask = df['season'] < 2025
    train_df = df[train_mask]

    # Get predictions from each expert
    meta_features = []
    for game_type, model in experts.items():
        X = train_df[feature_cols].fillna(0)
        preds = model.predict(X)
        meta_features.append(preds)

    # Stack predictions
    X_meta = np.column_stack(meta_features)
    y_meta = train_df['spread_error'].values

    # Also include game type as features
    game_type_dummies = pd.get_dummies(train_df['game_type'], prefix='gt')
    X_meta = np.hstack([X_meta, game_type_dummies.values])

    # Train Bayesian meta-learner
    meta_learner = BayesianRidge(
        n_iter=1000,
        compute_score=True
    )
    meta_learner.fit(X_meta, y_meta)

    # Evaluate on training data
    y_pred = meta_learner.predict(X_meta)
    mae = mean_absolute_error(y_meta, y_pred)
    print(f"Meta-learner training MAE: {mae:.4f}")

    return meta_learner, list(game_type_dummies.columns)


def evaluate_experts(df, experts, meta_learner, feature_cols, meta_columns):
    """Evaluate expert models on 2025 data."""
    print("\n" + "=" * 70)
    print("EVALUATION ON 2025 DATA")
    print("=" * 70)

    test_mask = df['season'] == 2025
    test_df = df[test_mask].copy()

    if len(test_df) == 0:
        print("No 2025 data available for evaluation")
        return

    print(f"\nTotal test games: {len(test_df)}")

    # Method 1: Route to appropriate expert
    routed_preds = []
    for idx, row in test_df.iterrows():
        game_type = row['game_type']
        X = row[feature_cols].fillna(0).values.reshape(1, -1)
        pred = experts[game_type].predict(X)[0]
        routed_preds.append(pred)

    test_df['routed_pred'] = routed_preds

    # Method 2: Meta-learner ensemble
    meta_features = []
    for game_type, model in experts.items():
        X = test_df[feature_cols].fillna(0)
        preds = model.predict(X)
        meta_features.append(preds)

    X_meta = np.column_stack(meta_features)
    game_type_dummies = pd.get_dummies(test_df['game_type'], prefix='gt')

    # Ensure all columns are present
    for col in meta_columns:
        if col not in game_type_dummies.columns:
            game_type_dummies[col] = 0
    game_type_dummies = game_type_dummies[meta_columns]

    X_meta = np.hstack([X_meta, game_type_dummies.values])

    test_df['meta_pred'] = meta_learner.predict(X_meta)

    # Evaluate
    actual = test_df['spread_error']

    print("\n--- Overall Results ---")

    # Routed expert
    routed_mae = mean_absolute_error(actual, test_df['routed_pred'])
    routed_dir = (np.sign(test_df['routed_pred']) == np.sign(actual)).mean()
    print(f"Routed Expert:   MAE={routed_mae:.3f}, Direction={routed_dir:.1%}")

    # Meta-learner
    meta_mae = mean_absolute_error(actual, test_df['meta_pred'])
    meta_dir = (np.sign(test_df['meta_pred']) == np.sign(actual)).mean()
    print(f"Meta-Learner:    MAE={meta_mae:.3f}, Direction={meta_dir:.1%}")

    # Vegas baseline
    vegas_mae = mean_absolute_error(actual, np.zeros(len(actual)))
    print(f"Vegas Baseline:  MAE={vegas_mae:.3f}")

    # By game type
    print("\n--- Results by Game Type ---")
    for game_type in test_df['game_type'].unique():
        type_df = test_df[test_df['game_type'] == game_type]
        if len(type_df) > 0:
            type_mae = mean_absolute_error(type_df['spread_error'], type_df['routed_pred'])
            type_dir = (np.sign(type_df['routed_pred']) == np.sign(type_df['spread_error'])).mean()
            print(f"  {game_type:15s}: {len(type_df):4d} games, MAE={type_mae:.2f}, Dir={type_dir:.1%}")


def main():
    print("=" * 70)
    print("V18 GAME-TYPE EXPERT TRAINING")
    print("=" * 70)

    # Load data
    df = load_data()

    # Get features
    feature_cols = get_features()
    feature_cols = [f for f in feature_cols if f in df.columns]
    print(f"\nUsing {len(feature_cols)} features")

    # Train expert models
    experts, expert_params = train_expert_models(df, feature_cols)

    # Train meta-learner
    meta_learner, meta_columns = train_meta_learner(df, experts, feature_cols)

    # Evaluate
    evaluate_experts(df, experts, meta_learner, feature_cols, meta_columns)

    # Save models
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    model_bundle = {
        'experts': experts,
        'meta_learner': meta_learner,
        'meta_columns': meta_columns,
        'classify_fn': classify_game_type,
        'game_types': list(experts.keys()),
    }

    joblib.dump(model_bundle, OUTPUT_MODEL)
    print(f"Saved expert models to {OUTPUT_MODEL}")

    config = {
        'version': 'V18',
        'features': feature_cols,
        'expert_params': expert_params,
        'game_types': GAME_TYPES,
    }
    joblib.dump(config, OUTPUT_CONFIG)
    print(f"Saved config to {OUTPUT_CONFIG}")

    print("\n" + "=" * 70)
    print("V18 EXPERT TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
