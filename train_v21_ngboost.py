"""
V21 NGBoost Training - Native Uncertainty Quantification.

NGBoost provides probabilistic predictions with native uncertainty estimates,
allowing us to adjust bet sizing based on prediction confidence.

Key advantages over V19:
1. Native uncertainty (std dev) without post-hoc calibration
2. Better tail probability estimates for extreme spreads
3. Uncertainty-aware Kelly fraction calculation

Usage:
    python train_v21_ngboost.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# V21 FEATURE SET (Same as V19/V20 + QB features)
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

    # V20: Weather features
    'wind_speed', 'temperature', 'is_dome', 'high_wind',
    'cold_game', 'wind_pass_impact',

    # V21: QB availability features
    'home_qb_status', 'away_qb_status', 'qb_advantage', 'qb_uncertainty',
]


class V21NGBoostModel:
    """
    NGBoost model for probabilistic margin prediction with uncertainty.

    This model predicts:
    - Mean margin (regression)
    - Standard deviation (uncertainty)

    The uncertainty can be used for:
    - Bet sizing (lower uncertainty = more confident bets)
    - Filtering games (skip high uncertainty games)
    - Kelly fraction adjustment
    """

    def __init__(self):
        self.ngb_model = None
        self.feature_names = None
        self.config = None

    def prepare_data(self, df):
        """Prepare data for training."""
        # Filter to games with Vegas spread
        df_valid = df[df['vegas_spread'].notna()].copy()

        # Target: actual margin
        df_valid['margin'] = df_valid['Margin']

        # Get features
        available_features = [f for f in V21_FEATURES if f in df_valid.columns]
        self.feature_names = available_features

        X = df_valid[available_features].copy()
        y = df_valid['margin']

        # Fill missing values
        X = X.fillna(X.median())

        return X, y, df_valid

    def optimize_ngboost(self, X_train, y_train, n_trials=30):
        """Optimize NGBoost hyperparameters with Optuna."""

        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.15)
            minibatch_frac = trial.suggest_float('minibatch_frac', 0.5, 1.0)
            max_depth = trial.suggest_int('max_depth', 2, 5)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 50)

            # Use decision tree as base learner
            base_learner = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf
            )

            model = NGBRegressor(
                Dist=Normal,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                minibatch_frac=minibatch_frac,
                Base=base_learner,
                random_state=42,
                verbose=False
            )

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model.fit(X_t.values, y_t.values)
                preds = model.predict(X_v.values)
                scores.append(mean_absolute_error(y_v, preds))

            return np.mean(scores)

        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return study.best_params

    def train(self, X, y, n_trials=30):
        """Train the NGBoost model."""

        print("=" * 70)
        print("V21 NGBOOST MODEL TRAINING")
        print("=" * 70)
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(self.feature_names)}")

        # Split data (use last 20% as holdout)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Optimize
        print("\nOptimizing NGBoost hyperparameters...")
        best_params = self.optimize_ngboost(X_train, y_train, n_trials=n_trials)
        print(f"Best params: {best_params}")

        # Train final model
        print("\nTraining final model...")
        base_learner = DecisionTreeRegressor(
            max_depth=best_params['max_depth'],
            min_samples_leaf=best_params['min_samples_leaf']
        )

        self.ngb_model = NGBRegressor(
            Dist=Normal,
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            minibatch_frac=best_params['minibatch_frac'],
            Base=base_learner,
            random_state=42,
            verbose=False
        )

        self.ngb_model.fit(X_train.values, y_train.values)

        # Evaluate
        pred_dist_train = self.ngb_model.pred_dist(X_train.values)
        pred_dist_test = self.ngb_model.pred_dist(X_test.values)

        train_mean = pred_dist_train.mean()
        test_mean = pred_dist_test.mean()
        test_std = pred_dist_test.std()

        print("\nModel Performance:")
        print(f"  Train MAE: {mean_absolute_error(y_train, train_mean):.2f}")
        print(f"  Test MAE: {mean_absolute_error(y_test, test_mean):.2f}")
        print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, test_mean)):.2f}")
        print(f"  Average predicted std: {np.mean(test_std):.2f}")
        print(f"  Std range: [{np.min(test_std):.2f}, {np.max(test_std):.2f}]")

        # Calibration check: Does predicted std correlate with actual error?
        actual_errors = np.abs(y_test.values - test_mean)
        corr = np.corrcoef(test_std, actual_errors)[0, 1]
        print(f"  Uncertainty-error correlation: {corr:.3f}")

        # Coverage check: How often is true value within 1, 2 std?
        within_1_std = np.mean(actual_errors <= test_std) * 100
        within_2_std = np.mean(actual_errors <= 2 * test_std) * 100
        print(f"  Coverage (1σ): {within_1_std:.1f}% (target: 68.3%)")
        print(f"  Coverage (2σ): {within_2_std:.1f}% (target: 95.4%)")

        # Save config
        self.config = {
            'version': 'V21_NGBOOST',
            'trained_at': datetime.now().isoformat(),
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'test_mae': mean_absolute_error(y_test, test_mean),
            'test_std_mean': np.mean(test_std),
            'uncertainty_correlation': corr,
            'coverage_1std': within_1_std,
            'coverage_2std': within_2_std,
            'params': best_params,
        }

        return self

    def predict(self, X, vegas_spread=None):
        """
        Make probabilistic predictions.

        Returns list of dicts with:
        - predicted_margin: Mean predicted margin
        - uncertainty: Standard deviation of prediction
        - lower_bound: Mean - 2*std
        - upper_bound: Mean + 2*std
        - predicted_edge: Edge over Vegas
        - confidence_from_uncertainty: High if low uncertainty
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        # Get predictive distribution
        pred_dist = self.ngb_model.pred_dist(X_arr)
        means = pred_dist.mean()
        stds = pred_dist.std()

        results = []
        for i in range(len(X_arr)):
            margin = means[i]
            std = stds[i]

            # Get spread
            if vegas_spread is not None:
                spread = vegas_spread
            elif isinstance(X, pd.DataFrame):
                spread = X.iloc[i].get('vegas_spread', 0)
            else:
                spread = 0

            # Calculate edge
            edge = margin - (-spread)

            # Uncertainty-based confidence
            # Lower std = higher confidence
            if std < 10:
                uncertainty_tier = 'HIGH'
            elif std < 14:
                uncertainty_tier = 'MEDIUM'
            else:
                uncertainty_tier = 'LOW'

            # Kelly adjustment factor based on uncertainty
            # Higher uncertainty = smaller bets
            kelly_adjustment = max(0.5, min(1.0, 12.0 / std))

            results.append({
                'predicted_margin': margin,
                'uncertainty': std,
                'lower_bound': margin - 2 * std,
                'upper_bound': margin + 2 * std,
                'predicted_edge': edge,
                'uncertainty_tier': uncertainty_tier,
                'kelly_adjustment': kelly_adjustment,
            })

        return results

    def save(self, path_prefix='cfb_v21_ngb'):
        """Save model to file."""
        with open(f'{path_prefix}.pkl', 'wb') as f:
            pickle.dump({
                'ngb_model': self.ngb_model,
                'feature_names': self.feature_names,
            }, f)

        with open(f'{path_prefix}_config.pkl', 'wb') as f:
            pickle.dump(self.config, f)

        print(f"Saved model to {path_prefix}.pkl")

    @classmethod
    def load(cls, path_prefix='cfb_v21_ngb'):
        """Load model from file."""
        model = cls()

        with open(f'{path_prefix}.pkl', 'rb') as f:
            data = pickle.load(f)
            model.ngb_model = data['ngb_model']
            model.feature_names = data['feature_names']

        with open(f'{path_prefix}_config.pkl', 'rb') as f:
            model.config = pickle.load(f)

        return model


def main():
    print("=" * 70)
    print("V21 NGBOOST TRAINING")
    print("=" * 70)

    # Load data
    print("\nLoading training data...")
    df = pd.read_csv('cfb_data_safe.csv')
    print(f"Loaded {len(df)} games")

    # Initialize and train model
    model = V21NGBoostModel()
    X, y, df_valid = model.prepare_data(df)
    print(f"Games with Vegas spread: {len(X)}")
    print(f"Available features: {len(model.feature_names)}")

    # Train
    model.train(X, y, n_trials=30)

    # Save
    model.save('cfb_v21_ngb')

    # Summary
    print("\n" + "=" * 70)
    print("V21 NGBOOST TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to cfb_v21_ngb.pkl")
    print(f"Config: {model.config}")


if __name__ == "__main__":
    main()
