"""
V17 Conformal Prediction Model for CFB Betting.

Conformal Prediction provides:
1. Statistically valid prediction intervals with guaranteed coverage
2. Distribution-free uncertainty quantification
3. Works with any underlying model (wraps XGBoost)

Research: "Conformal prediction produces statistically valid prediction regions
with explicit, non-asymptotic guarantees even without distributional assumptions."
- Berkeley Statistics (Angelopoulos & Bates 2021)

Key advantage: Unlike quantile regression, conformal prediction guarantees
that the prediction intervals will contain the true value X% of the time.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import optuna
from datetime import datetime
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# Try to import MAPIE for conformal prediction
try:
    from mapie.regression import MapieRegressor
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    print("WARNING: MAPIE not installed. Install with: pip install mapie")
    print("Falling back to manual conformal prediction implementation.")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = 'cfb_data_safe.csv'
CONFIG_FILE = 'cfb_v16_config.pkl'
OUTPUT_MODEL = 'cfb_spread_error_v17_conformal.pkl'
OUTPUT_CONFIG = 'cfb_v17_conformal_config.pkl'
OPTUNA_TRIALS = 100
RANDOM_STATE = 42

# Coverage levels
COVERAGE_LEVELS = [0.8, 0.9, 0.95]  # 80%, 90%, 95% prediction intervals

# Load features
try:
    V16_CONFIG = joblib.load(CONFIG_FILE)
    FEATURES = V16_CONFIG['features']
except FileNotFoundError:
    V15_CONFIG = joblib.load('cfb_v15_config.pkl')
    FEATURES = V15_CONFIG['features']

print(f"Using {len(FEATURES)} features")


def load_data():
    """Load training data."""
    print("Loading data...")

    df = pd.read_csv(DATA_FILE)
    df = df[df['vegas_spread'].notna()].copy()
    print(f"Loaded {len(df)} games with spreads")

    if 'spread_error' not in df.columns:
        df['margin'] = df['home_points'] - df['away_points']
        df['spread_error'] = df['margin'] - (-df['vegas_spread'])

    df = df.sort_values(['season', 'week']).reset_index(drop=True)
    return df


class ManualConformalRegressor:
    """
    Manual implementation of split conformal prediction.

    Uses the residuals from a calibration set to determine prediction intervals.
    """

    def __init__(self, base_model, coverage=0.9):
        self.base_model = base_model
        self.coverage = coverage
        self.calibration_scores = None

    def fit(self, X, y, calibration_fraction=0.2):
        """
        Fit the model and calculate calibration scores.

        Split data into training and calibration sets.
        """
        # Split into train and calibration
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=calibration_fraction, random_state=RANDOM_STATE
        )

        # Fit base model on training data
        self.base_model.fit(X_train, y_train)

        # Get predictions on calibration set
        y_pred_cal = self.base_model.predict(X_cal)

        # Calculate nonconformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_cal - y_pred_cal)

        return self

    def predict(self, X, coverage=None):
        """
        Make predictions with conformal intervals.

        Returns:
            y_pred: point predictions
            intervals: (lower, upper) bounds at specified coverage
        """
        if coverage is None:
            coverage = self.coverage

        y_pred = self.base_model.predict(X)

        # Get the quantile of calibration scores for desired coverage
        # Add 1/n correction for finite sample coverage guarantee
        n_cal = len(self.calibration_scores)
        adjusted_coverage = min(1.0, (1 + 1/n_cal) * coverage)

        quantile_value = np.quantile(self.calibration_scores, adjusted_coverage)

        # Prediction intervals
        lower = y_pred - quantile_value
        upper = y_pred + quantile_value

        return y_pred, np.column_stack([lower, upper])

    def get_interval_width(self, coverage=None):
        """Get the width of prediction intervals at given coverage."""
        if coverage is None:
            coverage = self.coverage

        n_cal = len(self.calibration_scores)
        adjusted_coverage = min(1.0, (1 + 1/n_cal) * coverage)

        return 2 * np.quantile(self.calibration_scores, adjusted_coverage)


def walk_forward_mae(model_class, params, df, feature_cols, target_col='spread_error'):
    """Standard walk-forward MAE calculation."""
    errors = []

    for test_week in range(1, 16):
        train_mask = (df['season'] < 2025) | ((df['season'] == 2025) & (df['week'] < test_week))
        test_mask = (df['season'] == 2025) & (df['week'] == test_week)

        train_data = df[train_mask]
        test_data = df[test_mask]

        if len(test_data) == 0:
            continue

        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data[target_col]
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data[target_col]

        model = model_class(**params, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        errors.extend(np.abs(y_pred - y_test).tolist())

    return np.mean(errors) if errors else 999.0


def optimize_base_model(df, feature_cols):
    """Optimize XGBoost base model."""
    print(f"\nOptimizing base XGBoost model ({OPTUNA_TRIALS} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'n_jobs': -1,
        }

        return walk_forward_mae(XGBRegressor, params, df, feature_cols)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    print(f"  Best MAE: {study.best_value:.4f}")
    return study.best_params


def train_conformal_model(df, feature_cols, xgb_params, coverage_levels=COVERAGE_LEVELS):
    """Train conformal prediction model."""
    print("\nTraining conformal prediction model...")

    train_mask = df['season'] < 2025
    train_data = df[train_mask]

    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['spread_error']

    if MAPIE_AVAILABLE:
        # Use MAPIE for conformal prediction
        print("  Using MAPIE for conformal prediction")

        base_model = XGBRegressor(**xgb_params, random_state=RANDOM_STATE, n_jobs=-1)

        # MAPIE with cross-validation for better calibration
        mapie_model = MapieRegressor(
            estimator=base_model,
            method="plus",  # Jackknife+ method
            cv=5
        )
        mapie_model.fit(X_train, y_train)

        return {'mapie': mapie_model, 'type': 'mapie'}

    else:
        # Use manual conformal prediction
        print("  Using manual conformal prediction")

        conformal_models = {}
        for coverage in coverage_levels:
            base_model = XGBRegressor(**xgb_params, random_state=RANDOM_STATE, n_jobs=-1)
            conformal_model = ManualConformalRegressor(base_model, coverage=coverage)
            conformal_model.fit(X_train, y_train, calibration_fraction=0.2)
            conformal_models[coverage] = conformal_model
            print(f"    Coverage {coverage:.0%}: interval width = {conformal_model.get_interval_width():.2f}")

        return {'models': conformal_models, 'type': 'manual'}


def evaluate_conformal(conformal_result, df, feature_cols):
    """Evaluate conformal prediction on 2025 data."""
    print("\n" + "=" * 60)
    print("CONFORMAL PREDICTION EVALUATION ON 2025 DATA")
    print("=" * 60)

    test_mask = df['season'] == 2025
    test_data = df[test_mask]

    if len(test_data) == 0:
        print("No 2025 data available")
        return

    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['spread_error'].values

    if conformal_result['type'] == 'mapie':
        mapie_model = conformal_result['mapie']

        print("\nCoverage Analysis (MAPIE):")
        for alpha in [0.2, 0.1, 0.05]:  # 80%, 90%, 95% coverage
            y_pred, y_pis = mapie_model.predict(X_test, alpha=alpha)

            # Check coverage
            covered = (y_test >= y_pis[:, 0, 0]) & (y_test <= y_pis[:, 1, 0])
            actual_coverage = covered.mean()
            expected_coverage = 1 - alpha

            # Interval width
            interval_widths = y_pis[:, 1, 0] - y_pis[:, 0, 0]
            avg_width = interval_widths.mean()

            print(f"  {expected_coverage:.0%} interval: actual coverage = {actual_coverage:.1%}, avg width = {avg_width:.2f}")

        # Point prediction metrics
        y_pred, _ = mapie_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        direction_acc = ((y_pred > 0) == (y_test > 0)).mean()

        print(f"\nPoint Prediction Metrics:")
        print(f"  MAE: {mae:.3f}")
        print(f"  Direction Accuracy: {direction_acc:.1%}")

    else:
        # Manual conformal
        print("\nCoverage Analysis (Manual Conformal):")

        for coverage, model in conformal_result['models'].items():
            y_pred, intervals = model.predict(X_test, coverage=coverage)

            # Check coverage
            covered = (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
            actual_coverage = covered.mean()

            # Interval width
            interval_widths = intervals[:, 1] - intervals[:, 0]
            avg_width = interval_widths.mean()

            print(f"  {coverage:.0%} interval: actual coverage = {actual_coverage:.1%}, avg width = {avg_width:.2f}")

        # Point prediction metrics using 90% model
        model_90 = conformal_result['models'][0.9]
        y_pred, _ = model_90.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        direction_acc = ((y_pred > 0) == (y_test > 0)).mean()

        print(f"\nPoint Prediction Metrics:")
        print(f"  MAE: {mae:.3f}")
        print(f"  Direction Accuracy: {direction_acc:.1%}")

    # Betting analysis based on interval confidence
    print("\nBetting Analysis Based on Prediction Intervals:")

    if conformal_result['type'] == 'mapie':
        y_pred, y_pis = conformal_result['mapie'].predict(X_test, alpha=0.2)  # 80% interval
        intervals = y_pis[:, :, 0]
    else:
        y_pred, intervals = conformal_result['models'][0.8].predict(X_test)

    # Games where 80% interval doesn't cross zero
    no_cross_zero = (intervals[:, 0] > 0) | (intervals[:, 1] < 0)
    print(f"\n  Games with 80% interval not crossing zero: {no_cross_zero.sum()} ({no_cross_zero.mean():.1%})")

    if no_cross_zero.sum() > 0:
        # Performance on these high-confidence bets
        confident_pred = y_pred[no_cross_zero]
        confident_actual = y_test[no_cross_zero]

        direction_correct = ((confident_pred > 0) == (confident_actual > 0)).sum()
        total = no_cross_zero.sum()
        profit = direction_correct * 0.91 - (total - direction_correct)
        roi = profit / total * 100

        print(f"  Direction accuracy: {direction_correct}/{total} = {direction_correct/total:.1%}")
        print(f"  Simulated ROI: {roi:+.1f}%")


def main():
    print("=" * 70)
    print("V17 CONFORMAL PREDICTION MODEL TRAINING")
    print("=" * 70)
    print(f"MAPIE available: {MAPIE_AVAILABLE}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()

    # Get available features
    feature_cols = [f for f in FEATURES if f in df.columns]
    print(f"\nUsing {len(feature_cols)} features")

    # Optimize base model
    xgb_params = optimize_base_model(df, feature_cols)

    # Train conformal model
    conformal_result = train_conformal_model(df, feature_cols, xgb_params)

    # Evaluate
    evaluate_conformal(conformal_result, df, feature_cols)

    # Save models and config
    print("\nSaving models...")
    joblib.dump(conformal_result, OUTPUT_MODEL)

    config = {
        'features': feature_cols,
        'xgb_params': xgb_params,
        'coverage_levels': COVERAGE_LEVELS,
        'mapie_available': MAPIE_AVAILABLE,
        'trained_on': datetime.now().isoformat(),
        'model_type': 'conformal_prediction',
    }
    joblib.dump(config, OUTPUT_CONFIG)

    print(f"\nSaved models to {OUTPUT_MODEL}")
    print(f"Saved config to {OUTPUT_CONFIG}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
