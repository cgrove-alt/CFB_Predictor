"""
V19 Dual-Target Model: Margin Prediction + Cover Probability.

KEY INSIGHT FROM ANALYSIS:
- Predicting spread_error (model error) is trying to beat Vegas at their own game
- Better approach: Predict game margin + classify cover probability separately

This model trains TWO targets:
1. MARGIN MODEL: Predict actual home team margin (regression)
2. COVER MODEL: Predict probability of covering spread (classification)

Final prediction combines both:
- Edge = predicted_margin - (-vegas_spread)
- Confidence = cover_probability (calibrated)

Usage:
    python train_v19_dual.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import BayesianRidge, LogisticRegression
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# V19 FEATURE SET (Cleaned - no constant features)
# =============================================================================
V19_FEATURES = [
    # Core power ratings
    'home_pregame_elo', 'away_pregame_elo', 'elo_diff',

    # Rolling performance
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',

    # Home field advantage
    'home_team_hfa', 'hfa_diff',

    # Scheduling factors
    'rest_diff', 'home_rest_days', 'away_rest_days',
    'home_short_rest', 'away_short_rest',

    # Vegas features (critical for cover model)
    'vegas_spread', 'line_movement', 'spread_open',
    'large_favorite', 'large_underdog', 'close_game',
    'elo_vs_spread', 'rest_spread_interaction',

    # Momentum features (dampened in V18+)
    'home_streak', 'away_streak', 'streak_diff',
    'home_ats', 'away_ats', 'ats_diff',
    'home_elo_momentum', 'away_elo_momentum', 'elo_momentum_diff',
    'home_scoring_trend', 'away_scoring_trend',

    # PPA efficiency (non-constant ones)
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',
    'home_comp_pass_ppa', 'away_comp_pass_ppa',
    'home_comp_rush_ppa', 'away_comp_rush_ppa',
    'pass_efficiency_diff',

    # Composite features
    'matchup_efficiency',
    'home_pass_rush_balance', 'away_pass_rush_balance',
    'elo_efficiency_interaction', 'momentum_strength',
    'dominant_home', 'dominant_away',
    'rest_favorite_interaction', 'has_line_movement',

    # Expected total (for game flow)
    'expected_total',
]

# Game type classification for filtering
def classify_game_type(row):
    """
    Classify game type for smart filtering.

    Returns dict with game type info.
    """
    elo_diff = abs(row.get('elo_diff', 0))
    spread = abs(row.get('vegas_spread', 0))
    week = row.get('week', 10)

    game_type = {
        'is_pick_em': spread < 3,  # Pick-em games are hard
        'is_avg_vs_avg': elo_diff < 100,  # No clear favorite
        'is_large_mismatch': elo_diff > 300,  # Blowout potential
        'is_early_season': week <= 2,  # Data staleness
        'is_elite_matchup': row.get('home_pregame_elo', 0) > 1700 and row.get('away_pregame_elo', 0) > 1700,
    }

    # Calculate a "bet quality" score
    bet_quality = 0
    if game_type['is_pick_em']:
        bet_quality -= 2  # Hard to predict
    if game_type['is_avg_vs_avg']:
        bet_quality -= 3  # Main source of losses
    if game_type['is_early_season']:
        bet_quality -= 1  # Data staleness
    if game_type['is_large_mismatch']:
        bet_quality += 1  # Easier to predict direction
    if game_type['is_elite_matchup']:
        bet_quality += 2  # Good data, predictable

    game_type['bet_quality_score'] = bet_quality
    return game_type


class V19DualTargetModel:
    """
    Dual-target model combining margin prediction and cover classification.
    """

    def __init__(self):
        self.margin_model = None
        self.cover_model = None
        self.calibrated_cover_model = None
        self.feature_names = None
        self.config = None

    def prepare_data(self, df):
        """Prepare data for training."""
        # Filter to games with Vegas spread
        df_valid = df[df['vegas_spread'].notna()].copy()

        # Create targets
        df_valid['margin'] = df_valid['Margin']  # Actual margin
        df_valid['covered'] = (df_valid['Margin'] > -df_valid['vegas_spread']).astype(int)

        # Get features
        available_features = [f for f in V19_FEATURES if f in df_valid.columns]
        self.feature_names = available_features

        X = df_valid[available_features].copy()
        y_margin = df_valid['margin']
        y_cover = df_valid['covered']

        # Fill missing values
        X = X.fillna(X.median())

        return X, y_margin, y_cover, df_valid

    def optimize_margin_model(self, X_train, y_train, n_trials=50):
        """Optimize margin prediction model with Optuna."""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            }

            model = XGBRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model.fit(X_t, y_t)
                preds = model.predict(X_v)
                scores.append(mean_absolute_error(y_v, preds))

            return np.mean(scores)

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return study.best_params

    def optimize_cover_model(self, X_train, y_train, n_trials=50):
        """Optimize cover probability model with Optuna."""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            }

            model = XGBClassifier(
                **params,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model.fit(X_t, y_t)
                probs = model.predict_proba(X_v)[:, 1]
                scores.append(brier_score_loss(y_v, probs))

            return np.mean(scores)

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return study.best_params

    def train(self, X, y_margin, y_cover, n_trials=30):
        """Train both models with optimization."""

        print("=" * 70)
        print("V19 DUAL-TARGET MODEL TRAINING")
        print("=" * 70)
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.feature_names)}")

        # Split data (use last 20% as holdout)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_margin_train, y_margin_test = y_margin.iloc[:split_idx], y_margin.iloc[split_idx:]
        y_cover_train, y_cover_test = y_cover.iloc[:split_idx], y_cover.iloc[split_idx:]

        # Optimize margin model
        print("\n" + "=" * 70)
        print("OPTIMIZING MARGIN MODEL")
        print("=" * 70)
        margin_params = self.optimize_margin_model(X_train, y_margin_train, n_trials=n_trials)
        print(f"Best margin params: {margin_params}")

        # Train final margin model
        self.margin_model = XGBRegressor(
            **margin_params,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        self.margin_model.fit(X_train, y_margin_train)

        # Evaluate margin model
        margin_preds_train = self.margin_model.predict(X_train)
        margin_preds_test = self.margin_model.predict(X_test)

        print(f"\nMargin Model Performance:")
        print(f"  Train MAE: {mean_absolute_error(y_margin_train, margin_preds_train):.2f}")
        print(f"  Test MAE: {mean_absolute_error(y_margin_test, margin_preds_test):.2f}")

        # Optimize cover model
        print("\n" + "=" * 70)
        print("OPTIMIZING COVER MODEL")
        print("=" * 70)
        cover_params = self.optimize_cover_model(X_train, y_cover_train, n_trials=n_trials)
        print(f"Best cover params: {cover_params}")

        # Train final cover model
        self.cover_model = XGBClassifier(
            **cover_params,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.cover_model.fit(X_train, y_cover_train)

        # Calibrate cover model for well-calibrated probabilities
        print("\nCalibrating cover model...")
        self.calibrated_cover_model = CalibratedClassifierCV(
            self.cover_model,
            method='isotonic',
            cv='prefit'
        )
        self.calibrated_cover_model.fit(X_test, y_cover_test)

        # Evaluate cover model
        cover_preds_train = self.cover_model.predict(X_train)
        cover_preds_test = self.cover_model.predict(X_test)
        cover_probs_test = self.calibrated_cover_model.predict_proba(X_test)[:, 1]

        print(f"\nCover Model Performance:")
        print(f"  Train Accuracy: {accuracy_score(y_cover_train, cover_preds_train)*100:.1f}%")
        print(f"  Test Accuracy: {accuracy_score(y_cover_test, cover_preds_test)*100:.1f}%")
        print(f"  Test Brier Score: {brier_score_loss(y_cover_test, cover_probs_test):.4f}")

        # Save config
        self.config = {
            'version': 'V19_DUAL',
            'trained_at': datetime.now().isoformat(),
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'margin_params': margin_params,
            'cover_params': cover_params,
            'train_margin_mae': mean_absolute_error(y_margin_train, margin_preds_train),
            'test_margin_mae': mean_absolute_error(y_margin_test, margin_preds_test),
            'train_cover_acc': accuracy_score(y_cover_train, cover_preds_train),
            'test_cover_acc': accuracy_score(y_cover_test, cover_preds_test),
        }

        return self

    def predict(self, X):
        """
        Make predictions with both models.

        Returns dict with:
        - predicted_margin: Predicted home team margin
        - predicted_edge: Edge over Vegas (margin - (-spread))
        - cover_probability: Calibrated probability of covering
        - confidence_tier: HIGH/MEDIUM-HIGH/MEDIUM/LOW/VERY LOW
        - bet_recommendation: BET/LEAN/PASS
        """
        # Fill missing values
        X_filled = X.fillna(X.median())

        # Get predictions
        predicted_margin = self.margin_model.predict(X_filled)
        cover_prob = self.calibrated_cover_model.predict_proba(X_filled)[:, 1]

        results = []
        for i in range(len(X)):
            margin = predicted_margin[i]
            prob = cover_prob[i]
            spread = X.iloc[i].get('vegas_spread', 0)

            # Calculate edge
            edge = margin - (-spread)  # Positive = model says home beats spread

            # Classify game type
            game_type = classify_game_type(X.iloc[i])

            # Determine confidence tier based on calibrated probability
            if prob >= 0.65 or prob <= 0.35:
                conf_tier = 'HIGH'
            elif prob >= 0.60 or prob <= 0.40:
                conf_tier = 'MEDIUM-HIGH'
            elif prob >= 0.55 or prob <= 0.45:
                conf_tier = 'MEDIUM'
            elif prob >= 0.52 or prob <= 0.48:
                conf_tier = 'LOW'
            else:
                conf_tier = 'VERY LOW'

            # Bet recommendation with stricter thresholds
            # Need both: good edge AND good probability AND good game type
            bet_home = edge > 0
            edge_abs = abs(edge)
            prob_confidence = abs(prob - 0.5)

            # PASS if game type is bad
            if game_type['bet_quality_score'] <= -3:
                recommendation = 'PASS'
            # BET only if strong confidence AND good edge
            elif edge_abs >= 4.5 and prob_confidence >= 0.15:
                recommendation = 'BET'
            # LEAN if moderate confidence
            elif edge_abs >= 3.0 and prob_confidence >= 0.10:
                recommendation = 'LEAN'
            else:
                recommendation = 'PASS'

            results.append({
                'predicted_margin': margin,
                'predicted_edge': edge,
                'cover_probability': prob,
                'confidence_tier': conf_tier,
                'bet_recommendation': recommendation,
                'pick_side': 'HOME' if bet_home else 'AWAY',
                'game_quality_score': game_type['bet_quality_score'],
            })

        return results

    def save(self, path_prefix='cfb_v19'):
        """Save model and config."""
        # Save models
        with open(f'{path_prefix}_dual.pkl', 'wb') as f:
            pickle.dump({
                'margin_model': self.margin_model,
                'cover_model': self.cover_model,
                'calibrated_cover_model': self.calibrated_cover_model,
                'feature_names': self.feature_names,
            }, f)

        # Save config
        with open(f'{path_prefix}_dual_config.pkl', 'wb') as f:
            pickle.dump(self.config, f)

        print(f"\nSaved model to {path_prefix}_dual.pkl")
        print(f"Saved config to {path_prefix}_dual_config.pkl")

    @classmethod
    def load(cls, path_prefix='cfb_v19'):
        """Load model from file."""
        model = cls()

        with open(f'{path_prefix}_dual.pkl', 'rb') as f:
            data = pickle.load(f)
            model.margin_model = data['margin_model']
            model.cover_model = data['cover_model']
            model.calibrated_cover_model = data['calibrated_cover_model']
            model.feature_names = data['feature_names']

        with open(f'{path_prefix}_dual_config.pkl', 'rb') as f:
            model.config = pickle.load(f)

        return model


def main():
    """Train and save the V19 dual-target model."""
    print("=" * 70)
    print("V19 DUAL-TARGET MODEL - MARGIN + COVER PROBABILITY")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('cfb_data_safe.csv')
    print(f"Loaded {len(df)} games")

    # Initialize model
    model = V19DualTargetModel()

    # Prepare data
    print("\nPreparing data...")
    X, y_margin, y_cover, df_valid = model.prepare_data(df)
    print(f"Valid games (with Vegas spread): {len(X)}")
    print(f"Features available: {len(model.feature_names)}")

    # Train
    model.train(X, y_margin, y_cover, n_trials=30)

    # Save
    model.save('cfb_v19')

    # Test predictions on recent data
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (Recent Games)")
    print("=" * 70)

    recent = X.tail(20)
    predictions = model.predict(recent)

    for i, (idx, row) in enumerate(recent.iterrows()):
        pred = predictions[i]
        print(f"\n{row.get('home_team', 'Home')} vs {row.get('away_team', 'Away')}")
        print(f"  Vegas Spread: {row.get('vegas_spread', 0):+.1f}")
        print(f"  Predicted Margin: {pred['predicted_margin']:+.1f}")
        print(f"  Edge: {pred['predicted_edge']:+.1f}")
        print(f"  Cover Prob: {pred['cover_probability']:.1%}")
        print(f"  Tier: {pred['confidence_tier']}")
        print(f"  Recommendation: {pred['bet_recommendation']} {pred['pick_side']}")

    print("\n" + "=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
