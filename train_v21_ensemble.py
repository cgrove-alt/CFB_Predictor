"""
V21 Ensemble Model - Combining All Improvements.

This ensemble combines:
1. V19 XGBoost dual-target model (margin + cover probability)
2. NGBoost for uncertainty quantification
3. Graph-based transitive margin for cross-conference games

Ensemble strategy:
- Use XGBoost as primary prediction
- Use NGBoost uncertainty for Kelly adjustment
- Weight transitive margin more heavily for cross-conference games

Usage:
    python train_v21_ensemble.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor, XGBClassifier
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# V21 FULL FEATURE SET (62 base + 6 graph = 68 features)
# =============================================================================
V21_FEATURES = [
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
    # Vegas features
    'vegas_spread', 'line_movement', 'spread_open',
    'large_favorite', 'large_underdog', 'close_game',
    'elo_vs_spread', 'rest_spread_interaction',
    # Momentum features
    'home_streak', 'away_streak', 'streak_diff',
    'home_ats', 'away_ats', 'ats_diff',
    'home_elo_momentum', 'away_elo_momentum', 'elo_momentum_diff',
    'home_scoring_trend', 'away_scoring_trend',
    # PPA efficiency
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
    # Expected total
    'expected_total',
    # Weather features
    'wind_speed', 'temperature', 'is_dome', 'high_wind',
    'cold_game', 'wind_pass_impact',
    # QB availability features
    'home_qb_status', 'away_qb_status', 'qb_advantage', 'qb_uncertainty',
    # Graph features (transitive margin from common opponents)
    'transitive_margin', 'transitive_confidence', 'num_common_opponents',
    'is_cross_conference', 'graph_margin_variance', 'transitive_weighted',
]


class V21EnsembleModel:
    """
    V21 Ensemble combining XGBoost, NGBoost, and graph features.

    Predictions:
    - predicted_margin: Weighted combination of XGBoost and transitive margin
    - cover_probability: Calibrated probability from classifier
    - uncertainty: From NGBoost for Kelly adjustment
    - confidence_tier: Based on cover prob and uncertainty
    """

    def __init__(self):
        self.margin_model = None
        self.cover_model = None
        self.ngb_model = None
        self.calibrator = None
        self.feature_names = None
        self.config = None

    def prepare_data(self, df):
        """Prepare data for training."""
        # Filter to games with Vegas spread and margin
        df_valid = df[(df['vegas_spread'].notna()) & (df['Margin'].notna())].copy()

        # Create targets
        df_valid['margin'] = df_valid['Margin']
        df_valid['covered'] = (df_valid['Margin'] > -df_valid['vegas_spread']).astype(int)

        # Get available features
        available_features = [f for f in V21_FEATURES if f in df_valid.columns]
        self.feature_names = available_features

        X = df_valid[available_features].copy()
        y_margin = df_valid['margin']
        y_cover = df_valid['covered']

        # Fill missing values
        X = X.fillna(X.median())

        return X, y_margin, y_cover, df_valid

    def optimize_margin_model(self, X_train, y_train, n_trials=30):
        """Optimize XGBoost margin model."""
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

            model = XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)

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

    def optimize_cover_model(self, X_train, y_train, n_trials=30):
        """Optimize XGBoost cover model."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }

            model = XGBClassifier(
                **params, random_state=42, n_jobs=-1, verbosity=0,
                use_label_encoder=False, eval_metric='logloss'
            )

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

    def train(self, X, y_margin, y_cover, n_trials=25):
        """Train all ensemble components."""
        print("=" * 70)
        print("V21 ENSEMBLE MODEL TRAINING")
        print("=" * 70)
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.feature_names)}")

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_margin_train, y_margin_test = y_margin.iloc[:split_idx], y_margin.iloc[split_idx:]
        y_cover_train, y_cover_test = y_cover.iloc[:split_idx], y_cover.iloc[split_idx:]

        # 1. Train XGBoost margin model
        print("\n" + "=" * 70)
        print("TRAINING XGBOOST MARGIN MODEL")
        print("=" * 70)
        margin_params = self.optimize_margin_model(X_train, y_margin_train, n_trials=n_trials)
        print(f"Best params: {margin_params}")

        self.margin_model = XGBRegressor(
            **margin_params, random_state=42, n_jobs=-1, verbosity=0
        )
        self.margin_model.fit(X_train, y_margin_train)

        margin_preds_test = self.margin_model.predict(X_test)
        margin_mae = mean_absolute_error(y_margin_test, margin_preds_test)
        print(f"Test MAE: {margin_mae:.2f}")

        # 2. Train XGBoost cover model
        print("\n" + "=" * 70)
        print("TRAINING XGBOOST COVER MODEL")
        print("=" * 70)
        cover_params = self.optimize_cover_model(X_train, y_cover_train, n_trials=n_trials)
        print(f"Best params: {cover_params}")

        self.cover_model = XGBClassifier(
            **cover_params, random_state=42, n_jobs=-1, verbosity=0,
            use_label_encoder=False, eval_metric='logloss'
        )
        self.cover_model.fit(X_train, y_cover_train)

        # Calibrate
        uncal_probs = self.cover_model.predict_proba(X_test)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(uncal_probs, y_cover_test)

        cover_probs_test = self.calibrator.transform(uncal_probs)
        cover_brier = brier_score_loss(y_cover_test, cover_probs_test)
        cover_acc = accuracy_score(y_cover_test, (cover_probs_test > 0.5).astype(int))
        print(f"Test Accuracy: {cover_acc*100:.1f}%")
        print(f"Test Brier Score: {cover_brier:.4f}")

        # 3. Train NGBoost for uncertainty
        print("\n" + "=" * 70)
        print("TRAINING NGBOOST FOR UNCERTAINTY")
        print("=" * 70)

        base_learner = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20)
        self.ngb_model = NGBRegressor(
            Dist=Normal,
            n_estimators=200,
            learning_rate=0.05,
            minibatch_frac=0.8,
            Base=base_learner,
            random_state=42,
            verbose=False
        )
        self.ngb_model.fit(X_train.values, y_margin_train.values)

        ngb_dist = self.ngb_model.pred_dist(X_test.values)
        ngb_mean = ngb_dist.mean()
        ngb_std = ngb_dist.std()

        ngb_mae = mean_absolute_error(y_margin_test, ngb_mean)
        print(f"NGBoost Test MAE: {ngb_mae:.2f}")
        print(f"Average uncertainty: {np.mean(ngb_std):.2f}")

        # Check ensemble improvement
        print("\n" + "=" * 70)
        print("ENSEMBLE EVALUATION")
        print("=" * 70)

        # Blend XGBoost and NGBoost predictions (simple average)
        ensemble_preds = (margin_preds_test + ngb_mean) / 2
        ensemble_mae = mean_absolute_error(y_margin_test, ensemble_preds)
        print(f"Ensemble (XGB+NGB avg) MAE: {ensemble_mae:.2f}")

        # Check if transitive margin helps for cross-conference
        if 'is_cross_conference' in X.columns and 'transitive_margin' in X.columns:
            cross_mask = (X_test['is_cross_conference'] == 1).values
            if cross_mask.sum() > 0:
                cross_margin_preds = margin_preds_test[cross_mask]
                cross_actual = y_margin_test.values[cross_mask]
                cross_mae = mean_absolute_error(cross_actual, cross_margin_preds)

                # Use transitive margin for cross-conference
                trans_margin = X_test['transitive_margin'].values[cross_mask]

                # Blend: more weight to transitive for cross-conference
                blended = cross_margin_preds * 0.6 + trans_margin * 0.4
                blended_mae = mean_absolute_error(cross_actual, blended)

                print(f"\nCross-conference games ({cross_mask.sum()}):")
                print(f"  XGBoost only MAE: {cross_mae:.2f}")
                print(f"  Blended (60/40) MAE: {blended_mae:.2f}")

        # Save config
        self.config = {
            'version': 'V21_ENSEMBLE',
            'trained_at': datetime.now().isoformat(),
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'margin_params': margin_params,
            'cover_params': cover_params,
            'xgb_margin_mae': margin_mae,
            'ngb_margin_mae': ngb_mae,
            'ensemble_mae': ensemble_mae,
            'cover_brier': cover_brier,
            'cover_accuracy': cover_acc,
        }

        return self

    def predict(self, X, vegas_spread=None):
        """Make ensemble predictions."""
        if isinstance(X, pd.DataFrame):
            X_filled = X.fillna(X.median())
        else:
            X_filled = np.nan_to_num(X, nan=0.0)

        # XGBoost predictions
        xgb_margin = self.margin_model.predict(X_filled)
        xgb_cover_uncal = self.cover_model.predict_proba(X_filled)[:, 1]
        xgb_cover = self.calibrator.transform(xgb_cover_uncal)

        # NGBoost predictions (for uncertainty)
        X_arr = X_filled.values if isinstance(X_filled, pd.DataFrame) else X_filled
        ngb_dist = self.ngb_model.pred_dist(X_arr)
        ngb_mean = ngb_dist.mean()
        ngb_std = ngb_dist.std()

        results = []
        for i in range(len(X_filled)):
            # Get spread
            if vegas_spread is not None:
                spread = vegas_spread
            elif isinstance(X_filled, pd.DataFrame):
                spread = X_filled.iloc[i].get('vegas_spread', 0)
            else:
                spread = 0

            # Ensemble margin: average XGB and NGB
            margin = (xgb_margin[i] + ngb_mean[i]) / 2

            # Note: Transitive margin is already a feature in the XGBoost model,
            # so we don't need to explicitly blend it here. The model learned
            # the optimal weighting during training.

            # Calculate edge
            edge = margin - (-spread)

            # Combine uncertainty and cover probability for confidence
            cover_prob = np.clip(xgb_cover[i], 0.05, 0.95)
            uncertainty = ngb_std[i]

            # Confidence tier
            prob_edge = abs(cover_prob - 0.5)
            if prob_edge >= 0.15 and uncertainty < 10:
                conf_tier = 'HIGH'
            elif prob_edge >= 0.10 and uncertainty < 14:
                conf_tier = 'MEDIUM-HIGH'
            elif prob_edge >= 0.05:
                conf_tier = 'MEDIUM'
            else:
                conf_tier = 'LOW'

            # Kelly adjustment based on uncertainty
            kelly_adj = max(0.5, min(1.0, 12.0 / uncertainty))

            # Bet recommendation
            if conf_tier in ['HIGH', 'MEDIUM-HIGH'] and abs(edge) >= 4.0:
                recommendation = 'BET'
            elif conf_tier in ['MEDIUM-HIGH', 'MEDIUM'] and abs(edge) >= 3.0:
                recommendation = 'LEAN'
            else:
                recommendation = 'PASS'

            results.append({
                'predicted_margin': margin,
                'predicted_edge': edge,
                'cover_probability': cover_prob,
                'uncertainty': uncertainty,
                'confidence_tier': conf_tier,
                'kelly_adjustment': kelly_adj,
                'bet_recommendation': recommendation,
                'pick_side': 'HOME' if edge > 0 else 'AWAY',
            })

        return results

    def save(self, path_prefix='cfb_v21_ensemble'):
        """Save ensemble model."""
        with open(f'{path_prefix}.pkl', 'wb') as f:
            pickle.dump({
                'margin_model': self.margin_model,
                'cover_model': self.cover_model,
                'ngb_model': self.ngb_model,
                'calibrator': self.calibrator,
                'feature_names': self.feature_names,
            }, f)

        with open(f'{path_prefix}_config.pkl', 'wb') as f:
            pickle.dump(self.config, f)

        print(f"Saved ensemble to {path_prefix}.pkl")

    @classmethod
    def load(cls, path_prefix='cfb_v21_ensemble'):
        """Load ensemble model."""
        model = cls()

        with open(f'{path_prefix}.pkl', 'rb') as f:
            data = pickle.load(f)
            model.margin_model = data['margin_model']
            model.cover_model = data['cover_model']
            model.ngb_model = data['ngb_model']
            model.calibrator = data['calibrator']
            model.feature_names = data['feature_names']

        with open(f'{path_prefix}_config.pkl', 'rb') as f:
            model.config = pickle.load(f)

        return model


def main():
    print("=" * 70)
    print("V21 ENSEMBLE TRAINING")
    print("=" * 70)

    # Load data with graph features
    print("\nLoading data...")
    try:
        df = pd.read_csv('cfb_data_graph.csv')
        print("Loaded data with graph features")
    except FileNotFoundError:
        df = pd.read_csv('cfb_data_safe.csv')
        print("Loaded base data (no graph features)")

    print(f"Total games: {len(df)}")

    # Train ensemble
    model = V21EnsembleModel()
    X, y_margin, y_cover, df_valid = model.prepare_data(df)

    print(f"Games with Vegas spread: {len(X)}")
    print(f"Features: {len(model.feature_names)}")

    model.train(X, y_margin, y_cover, n_trials=25)

    # Save
    model.save('cfb_v21_ensemble')

    # Summary
    print("\n" + "=" * 70)
    print("V21 ENSEMBLE TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel saved to cfb_v21_ensemble.pkl")
    print(f"\nKey metrics:")
    print(f"  XGBoost Margin MAE: {model.config['xgb_margin_mae']:.2f}")
    print(f"  NGBoost Margin MAE: {model.config['ngb_margin_mae']:.2f}")
    print(f"  Ensemble MAE: {model.config['ensemble_mae']:.2f}")
    print(f"  Cover Accuracy: {model.config['cover_accuracy']*100:.1f}%")
    print(f"  Cover Brier Score: {model.config['cover_brier']:.4f}")


if __name__ == "__main__":
    main()
