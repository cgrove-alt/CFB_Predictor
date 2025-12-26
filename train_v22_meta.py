"""
V22 Meta-Router Model Training Pipeline
========================================

Addresses V21 weaknesses identified in error analysis:
1. "Average vs Average" games (MAE 11.35, worst category)
2. Small-school/FCS outliers (Samford MAE 24.0)
3. away_scoring_trend error amplification (1.73x)
4. Blowout game unpredictability

Architecture:
- Meta-Router (Gating Network): Classifies game cluster and routes to specialized sub-models
  - Cluster 0: Standard Game (use standard XGBoost model)
  - Cluster 1: High Variance/Mismatch (use Graph Model + Uncertainty Penalty)
  - Cluster 2: Blowout Risk (Margin > 24, apply 1.15x multiplier)
- Sub-Model Ensemble: XGBoost, Graph-aware model, Blowout Classifier
- Final prediction weighted by meta-router confidence
"""

import os
import pickle
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, accuracy_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    HAS_NGBOOST = True
except ImportError:
    HAS_NGBOOST = False
    print("NGBoost not available - using XGBoost only for uncertainty")

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# FCS and small-school teams to FILTER OUT (high error rates from error_patterns_report.txt)
FCS_TEAMS = {
    'Samford', 'Davidson', 'Idaho State', 'Northern Arizona', 'Cal Poly',
    'Sacramento State', 'Montana', 'Montana State', 'Eastern Washington',
    'Weber State', 'UC Davis', 'Portland State', 'Southern Utah',
    'Abilene Christian', 'Incarnate Word', 'Stephen F. Austin', 'Sam Houston',
    'Tarleton State', 'Lamar', 'McNeese', 'Nicholls', 'Northwestern State',
    'SE Louisiana', 'Houston Christian', 'Texas A&M-Commerce',
    'Alabama A&M', 'Alabama State', 'Alcorn State', 'Arkansas-Pine Bluff',
    'Bethune-Cookman', 'Delaware State', 'FAMU', 'Grambling', 'Howard',
    'Jackson State', 'Mississippi Valley State', 'Morgan State', 'Norfolk State',
    'North Carolina A&T', 'North Carolina Central', 'Prairie View A&M',
    'SC State', 'Southern', 'Texas Southern', 'Kennesaw State', 'North Alabama',
    'Jacksonville State', 'Central Arkansas', 'Austin Peay', 'Eastern Kentucky',
    'Lindenwood', 'Southern Indiana', 'Western Illinois', 'Tennessee State',
    'Tennessee Tech', 'UT Martin', 'Southeast Missouri', 'Murray State',
    'Morehead State', 'Bucknell', 'Colgate', 'Fordham', 'Georgetown',
    'Holy Cross', 'Lafayette', 'Lehigh', 'Merrimack', 'Sacred Heart',
    'Stonehill', 'Villanova', 'Charleston Southern', 'Campbell', 'Gardner-Webb',
    'Robert Morris', 'Bryant', 'Duquesne', 'LIU',
    'Stony Brook', 'Maine', 'New Hampshire', 'Rhode Island', 'Albany',
    'Delaware', 'Elon', 'Hampton', 'Monmouth', 'Richmond', 'Towson',
    'William & Mary', 'Dartmouth', 'Harvard', 'Yale', 'Princeton',
    'Penn', 'Brown', 'Columbia', 'Cornell', 'Drake', 'Dayton', 'Marist',
    'Presbyterian', 'San Diego', 'Stetson', 'Valparaiso',
    'Butler', 'St. Thomas', 'Illinois State', 'Indiana State', 'Missouri State',
    'North Dakota', 'North Dakota State', 'Northern Iowa', 'South Dakota',
    'South Dakota State', 'Southern Illinois', 'Youngstown State',
    'Chattanooga', 'East Tennessee State', 'Furman', 'Mercer',
    'VMI', 'Western Carolina', 'Wofford', 'The Citadel', 'Idaho', 'Eastern Illinois'
}

# Toxic features to DROP immediately (from error_insights.txt)
TOXIC_FEATURES = [
    'away_scoring_trend',  # 1.73x error amplification
]

# Game Cluster definitions for Meta-Router
CLUSTER_STANDARD = 0      # Standard game - use base XGBoost
CLUSTER_HIGH_VARIANCE = 1 # High variance/mismatch - use uncertainty-aware model
CLUSTER_BLOWOUT = 2       # Blowout risk (margin > 24)

# V22 Base Features for sub-models
V22_BASE_FEATURES = [
    # Elo ratings
    'home_pregame_elo', 'away_pregame_elo', 'elo_diff',
    # Rolling performance
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',
    # Home field advantage
    'home_team_hfa', 'hfa_diff',
    # Rest/scheduling
    'rest_diff', 'home_rest_days', 'away_rest_days',
    'home_short_rest', 'away_short_rest',
    # Vegas features
    'vegas_spread', 'line_movement', 'spread_open',
    'large_favorite', 'large_underdog', 'close_game',
    'elo_vs_spread', 'rest_spread_interaction',
    # Momentum (WITHOUT toxic away_scoring_trend)
    'home_streak', 'away_streak', 'streak_diff',
    'home_ats', 'away_ats', 'ats_diff',
    'home_elo_momentum', 'away_elo_momentum', 'elo_momentum_diff',
    'home_scoring_trend',  # KEEP home, REMOVED away (toxic)
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
]

# V22 Style Matchup Features (for differentiating Average vs Average games)
V22_STYLE_FEATURES = [
    'pass_off_vs_pass_def',       # Home pass offense vs away pass defense
    'rush_off_vs_rush_def',       # Home rush offense vs away rush defense
    'style_mismatch_total',       # Combined style advantage
    'style_balance',              # Pass-heavy vs balanced indicator
]

# V22 Volatility Features
V22_VOLATILITY_FEATURES = [
    'home_volatility',            # Home team scoring volatility
    'away_volatility',            # Away team scoring volatility
    'matchup_volatility',         # Average of home/away volatility
    'volatility_index',           # Combined scoring + defensive volatility
    'volatility_diff',            # Volatility differential
]

# Meta-Router Input Features (game context for routing)
META_ROUTER_FEATURES = [
    'week',                       # Week of season
    'elo_diff',                   # Elo differential (absolute value)
    'matchup_volatility',         # Average volatility of both teams
    'vegas_spread',               # Spread as indicator of expected game type
]


# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================

def load_and_clean_data(csv_path: str = 'cfb_data_safe.csv') -> pd.DataFrame:
    """
    Load data and apply V22 cleaning steps:
    1. Filter out FCS teams completely
    2. Drop toxic features immediately
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    original_count = len(df)
    logger.info(f"Loaded {original_count} games")

    # 1. FILTER OUT FCS teams completely (not just flag them)
    fcs_home = df['home_team'].isin(FCS_TEAMS)
    fcs_away = df['away_team'].isin(FCS_TEAMS)
    fcs_games = fcs_home | fcs_away

    fcs_count = fcs_games.sum()
    df = df[~fcs_games].reset_index(drop=True)
    logger.info(f"REMOVED {fcs_count} games involving FCS teams ({100*fcs_count/original_count:.1f}%)")
    logger.info(f"Remaining: {len(df)} FBS-only games")

    # 2. DROP toxic features immediately
    for feat in TOXIC_FEATURES:
        if feat in df.columns:
            df = df.drop(columns=[feat])
            logger.info(f"DROPPED toxic feature: {feat}")

    # 3. Engineer style and volatility features if not present
    df = engineer_v22_features(df)

    # 4. Create game cluster labels for meta-router training
    df = create_game_clusters(df)

    return df


def engineer_v22_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add V22-specific features: style matchups and volatility.
    """
    # Style Matchup Features
    if 'home_comp_pass_ppa' in df.columns and 'away_comp_def_ppa' in df.columns:
        df['pass_off_vs_pass_def'] = df['home_comp_pass_ppa'] - df['away_comp_def_ppa']
    else:
        df['pass_off_vs_pass_def'] = 0.0

    if 'home_comp_rush_ppa' in df.columns and 'away_comp_def_ppa' in df.columns:
        df['rush_off_vs_rush_def'] = df['home_comp_rush_ppa'] - df['away_comp_def_ppa']
    else:
        df['rush_off_vs_rush_def'] = 0.0

    df['style_mismatch_total'] = df['pass_off_vs_pass_def'] + df['rush_off_vs_rush_def']

    if len(df) > 0 and df['pass_off_vs_pass_def'].std() > 0:
        df['style_balance'] = (df['pass_off_vs_pass_def'] - df['rush_off_vs_rush_def']).abs()
    else:
        df['style_balance'] = 0.0

    # Volatility Features (using elo_diff as proxy if scoring data not available)
    if 'home_volatility' not in df.columns:
        # Use Elo-based proxy: lower Elo diff = higher volatility (harder to predict)
        elo_diff_abs = df['elo_diff'].abs() if 'elo_diff' in df.columns else 0
        df['home_volatility'] = 10 - (elo_diff_abs / 100).clip(0, 10)
        df['away_volatility'] = df['home_volatility']

    df['matchup_volatility'] = (df['home_volatility'] + df['away_volatility']) / 2.0

    if 'volatility_index' not in df.columns:
        df['volatility_index'] = df['matchup_volatility']

    if 'volatility_diff' not in df.columns:
        df['volatility_diff'] = df['home_volatility'] - df['away_volatility']

    # Ensure elo_diff exists
    if 'elo_diff' not in df.columns:
        if 'home_pregame_elo' in df.columns and 'away_pregame_elo' in df.columns:
            df['elo_diff'] = df['home_pregame_elo'] - df['away_pregame_elo']
        else:
            df['elo_diff'] = 0

    logger.info(f"Added V22 style and volatility features")
    return df


def create_game_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic target for meta-router (game cluster classification):
    - Cluster 0: Standard Game
    - Cluster 1: High Variance/Mismatch (close spreads + high volatility)
    - Cluster 2: Blowout Risk (actual margin > 24)
    """
    # Determine target column
    if 'Margin' in df.columns:
        margin_col = 'Margin'
    elif 'margin' in df.columns:
        margin_col = 'margin'
    elif 'actual_margin' in df.columns:
        margin_col = 'actual_margin'
    else:
        raise ValueError("No margin column found in data")

    margin_abs = df[margin_col].abs()
    spread_abs = df['vegas_spread'].abs() if 'vegas_spread' in df.columns else 0
    volatility = df['matchup_volatility'] if 'matchup_volatility' in df.columns else 10

    # Initialize all as Standard
    df['game_cluster'] = CLUSTER_STANDARD

    # Cluster 2: Blowout Risk (margin > 24)
    blowout_mask = margin_abs > 24
    df.loc[blowout_mask, 'game_cluster'] = CLUSTER_BLOWOUT

    # Cluster 1: High Variance/Mismatch
    # Criteria: Close spread (<7) AND high volatility (>12) AND NOT already a blowout
    high_variance_mask = (
        (spread_abs < 7) &
        (volatility > 12) &
        (~blowout_mask)
    )
    df.loc[high_variance_mask, 'game_cluster'] = CLUSTER_HIGH_VARIANCE

    # Log distribution
    cluster_counts = df['game_cluster'].value_counts().sort_index()
    cluster_names = {0: 'Standard', 1: 'High Variance', 2: 'Blowout'}
    logger.info("Game cluster distribution:")
    for c, count in cluster_counts.items():
        logger.info(f"  {cluster_names.get(c, c)}: {count} ({100*count/len(df):.1f}%)")

    # Also create is_blowout for blowout classifier
    df['is_blowout'] = (margin_abs > 24).astype(int)

    return df


# =============================================================================
# META-ROUTER (GATING NETWORK)
# =============================================================================

class MetaRouter:
    """
    Gating network that classifies games into clusters:
    - Cluster 0: Standard Game -> Use XGBoost
    - Cluster 1: High Variance -> Use uncertainty-aware model
    - Cluster 2: Blowout Risk -> Apply 1.15x margin multiplier

    Input: Week, Conference, Elo_Diff, Matchup_Volatility
    """

    def __init__(self):
        self.classifier = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list):
        """Train the meta-router classifier."""
        self.feature_names = feature_names

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Encode labels (ensure 0-indexed)
        y_encoded = self.label_encoder.fit_transform(y)

        # Train XGBoost classifier
        self.classifier = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=len(np.unique(y_encoded)),
            random_state=42,
            verbosity=0
        )

        self.classifier.fit(X_scaled, y_encoded)

        # Log training accuracy
        train_pred = self.classifier.predict(X_scaled)
        train_acc = accuracy_score(y_encoded, train_pred)
        logger.info(f"Meta-router training accuracy: {train_acc:.3f}")

        return self

    def predict(self, X: np.ndarray) -> tuple:
        """
        Predict game cluster and confidence.

        Returns:
            (cluster_ids, confidences): Original cluster labels and confidence scores
        """
        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)
        pred_encoded = np.argmax(probs, axis=1)

        # Decode back to original cluster labels
        cluster_ids = self.label_encoder.inverse_transform(pred_encoded)
        confidences = np.max(probs, axis=1)

        return cluster_ids, confidences

    def get_cluster_probs(self, X: np.ndarray) -> np.ndarray:
        """Get probability distribution over clusters for soft weighting."""
        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)

        # Create full probability matrix (3 clusters)
        n_classes = len(self.label_encoder.classes_)
        if n_classes < 3:
            # Pad to 3 classes
            full_probs = np.zeros((probs.shape[0], 3))
            for i, cls in enumerate(self.label_encoder.classes_):
                full_probs[:, cls] = probs[:, i]
            return full_probs

        return probs


# =============================================================================
# SUB-MODELS
# =============================================================================

class StandardXGBoostModel:
    """Standard XGBoost model for Cluster 0 (standard games)."""

    def __init__(self):
        self.margin_model = None
        self.cover_model = None
        self.calibrator = None

    def train(self, X: np.ndarray, y_margin: np.ndarray, y_cover: np.ndarray):
        """Train standard XGBoost model."""
        logger.info(f"Training StandardXGBoostModel on {len(X)} samples")

        self.margin_model = XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
        self.margin_model.fit(X, y_margin)

        self.cover_model = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbosity=0
        )
        self.cover_model.fit(X, y_cover)

        # Calibrate cover probabilities
        cover_probs = self.cover_model.predict_proba(X)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(cover_probs, y_cover)

        return self

    def predict(self, X: np.ndarray) -> tuple:
        """Predict margin and calibrated cover probability."""
        margin = self.margin_model.predict(X)
        cover_prob_raw = self.cover_model.predict_proba(X)[:, 1]
        cover_prob = self.calibrator.transform(cover_prob_raw)
        return margin, cover_prob


class HighVarianceModel:
    """
    Uncertainty-aware model for Cluster 1 (high variance games).
    Uses NGBoost for native uncertainty quantification if available.
    Applies uncertainty penalty to bet sizing.
    """

    def __init__(self):
        self.margin_model = None
        self.cover_model = None
        self.calibrator = None
        self.uncertainty_model = None

    def train(self, X: np.ndarray, y_margin: np.ndarray, y_cover: np.ndarray):
        """Train high-variance model with uncertainty quantification."""
        logger.info(f"Training HighVarianceModel on {len(X)} samples")

        # NGBoost for uncertainty if available
        if HAS_NGBOOST:
            self.uncertainty_model = NGBRegressor(
                Dist=Normal,
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            )
            self.uncertainty_model.fit(X, y_margin)
            logger.info("Trained NGBoost uncertainty model for high-variance games")

        # XGBoost margin model with higher regularization
        self.margin_model = XGBRegressor(
            n_estimators=300,
            max_depth=3,  # Shallower to reduce overfitting
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,  # Higher L1 regularization
            reg_lambda=2.0,  # Higher L2 regularization
            random_state=42,
            verbosity=0
        )
        self.margin_model.fit(X, y_margin)

        # Cover model
        self.cover_model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            verbosity=0
        )
        self.cover_model.fit(X, y_cover)

        # Calibrate
        cover_probs = self.cover_model.predict_proba(X)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(cover_probs, y_cover)

        return self

    def predict(self, X: np.ndarray) -> tuple:
        """Predict with uncertainty estimate."""
        margin = self.margin_model.predict(X)
        cover_prob_raw = self.cover_model.predict_proba(X)[:, 1]
        cover_prob = self.calibrator.transform(cover_prob_raw)

        # Get uncertainty from NGBoost if available
        if self.uncertainty_model is not None:
            pred_dist = self.uncertainty_model.pred_dist(X)
            uncertainty = pred_dist.std()
        else:
            uncertainty = np.full(len(X), 10.0)  # Default high uncertainty

        return margin, cover_prob, uncertainty


class BlowoutClassifier:
    """
    Blowout classifier for Cluster 2 (blowout risk games).
    If blowout predicted with >60% confidence, apply 1.15x margin multiplier.
    """

    def __init__(self):
        self.classifier = None
        self.margin_model = None
        self.cover_model = None
        self.calibrator = None

    def train(self, X: np.ndarray, y_margin: np.ndarray, y_cover: np.ndarray, y_blowout: np.ndarray):
        """Train blowout classifier and margin models."""
        logger.info(f"Training BlowoutClassifier on {len(X)} samples")

        # Calculate class weight
        n_blowout = np.sum(y_blowout == 1)
        n_not_blowout = np.sum(y_blowout == 0)
        scale_pos_weight = max(0.1, n_not_blowout / n_blowout) if n_blowout > 0 else 1.0

        # Blowout classifier
        self.classifier = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=0
        )
        self.classifier.fit(X, y_blowout)

        blowout_pred = self.classifier.predict(X)
        blowout_acc = accuracy_score(y_blowout, blowout_pred)
        logger.info(f"Blowout classifier accuracy: {blowout_acc:.3f}")

        # Margin model
        self.margin_model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        self.margin_model.fit(X, y_margin)

        # Cover model
        self.cover_model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        self.cover_model.fit(X, y_cover)

        # Calibrate
        cover_probs = self.cover_model.predict_proba(X)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(cover_probs, y_cover)

        return self

    def predict(self, X: np.ndarray) -> tuple:
        """
        Predict with blowout adjustment.
        If blowout probability > 60%, apply 1.15x multiplier to margin.
        """
        margin = self.margin_model.predict(X)
        cover_prob_raw = self.cover_model.predict_proba(X)[:, 1]
        cover_prob = self.calibrator.transform(cover_prob_raw)

        # Blowout probability
        blowout_prob = self.classifier.predict_proba(X)[:, 1]

        # Apply 1.15x multiplier if blowout predicted with >60% confidence
        blowout_adjustment = np.where(blowout_prob > 0.6, 1.15, 1.0)
        margin_adjusted = margin * blowout_adjustment

        return margin_adjusted, cover_prob, blowout_prob


# =============================================================================
# V22 META-ROUTER ENSEMBLE MODEL
# =============================================================================

class V22MetaRouterModel:
    """
    V22 Meta-Router Ensemble Model.

    Uses a gating network to classify games into clusters:
    - Cluster 0: Standard -> XGBoost
    - Cluster 1: High Variance -> Uncertainty-aware model
    - Cluster 2: Blowout Risk -> Blowout classifier with 1.15x multiplier

    Final prediction weighted by meta-router confidence.
    """

    def __init__(self):
        self.meta_router = MetaRouter()
        self.standard_model = StandardXGBoostModel()
        self.high_variance_model = HighVarianceModel()
        self.blowout_model = BlowoutClassifier()

        self.feature_names = None
        self.router_feature_names = None
        self.scaler = StandardScaler()

        self.metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for all sub-models.

        Returns:
            (X_base, X_router, feature_names, router_feature_names)
        """
        # Base features for sub-models
        available_base = [f for f in V22_BASE_FEATURES if f in df.columns]
        available_style = [f for f in V22_STYLE_FEATURES if f in df.columns]
        available_vol = [f for f in V22_VOLATILITY_FEATURES if f in df.columns]

        self.feature_names = available_base + available_style + available_vol
        logger.info(f"Using {len(self.feature_names)} features for sub-models")

        X_base = df[self.feature_names].fillna(0).values

        # Router features
        router_features = []
        for feat in META_ROUTER_FEATURES:
            if feat in df.columns:
                router_features.append(feat)

        self.router_feature_names = router_features
        X_router = df[router_features].fillna(0).values

        return X_base, X_router

    def train(self, df: pd.DataFrame, target_margin: str = 'Margin',
              target_spread: str = 'vegas_spread'):
        """
        Train the V22 meta-router ensemble.
        """
        logger.info("=" * 60)
        logger.info("Training V22 Meta-Router Ensemble Model")
        logger.info("=" * 60)

        # Prepare data (load if string path provided)
        if isinstance(df, str):
            df = load_and_clean_data(df)
        else:
            df = df.copy()
            df = engineer_v22_features(df)
            df = create_game_clusters(df)

        # Check for target columns
        if target_margin not in df.columns:
            for candidate in ['margin', 'Margin', 'actual_margin']:
                if candidate in df.columns:
                    target_margin = candidate
                    break
            else:
                raise ValueError(f"Target column {target_margin} not found")

        # Create cover target
        df['covered'] = df[target_margin] + df[target_spread] > 0

        # Prepare features
        X_base, X_router = self.prepare_features(df)
        y_margin = df[target_margin].values
        y_cover = df['covered'].astype(int).values
        y_blowout = df['is_blowout'].astype(int).values
        y_cluster = df['game_cluster'].values

        # Scale base features
        X_base_scaled = self.scaler.fit_transform(X_base)

        # Train-test split (time-based)
        if 'season' in df.columns:
            train_mask = df['season'] < df['season'].max()
            test_mask = df['season'] == df['season'].max()
        else:
            n_train = int(0.8 * len(df))
            train_mask = np.arange(len(df)) < n_train
            test_mask = ~train_mask

        X_train = X_base_scaled[train_mask]
        X_test = X_base_scaled[test_mask]
        X_router_train = X_router[train_mask]
        X_router_test = X_router[test_mask]

        y_margin_train = y_margin[train_mask]
        y_margin_test = y_margin[test_mask]
        y_cover_train = y_cover[train_mask]
        y_cover_test = y_cover[test_mask]
        y_blowout_train = y_blowout[train_mask]
        y_cluster_train = y_cluster[train_mask]
        y_cluster_test = y_cluster[test_mask]

        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # 1. Train Meta-Router
        logger.info("\n--- Training Meta-Router ---")
        self.meta_router.train(X_router_train, y_cluster_train, self.router_feature_names)

        # 2. Train specialized sub-models

        # Standard model (Cluster 0)
        logger.info("\n--- Training Standard Model (Cluster 0) ---")
        standard_mask = y_cluster_train == CLUSTER_STANDARD
        if standard_mask.sum() > 50:
            self.standard_model.train(
                X_train[standard_mask],
                y_margin_train[standard_mask],
                y_cover_train[standard_mask]
            )
        else:
            logger.warning(f"Not enough Standard samples ({standard_mask.sum()}), training on all data")
            self.standard_model.train(X_train, y_margin_train, y_cover_train)

        # High Variance model (Cluster 1)
        logger.info("\n--- Training High Variance Model (Cluster 1) ---")
        hv_mask = y_cluster_train == CLUSTER_HIGH_VARIANCE
        if hv_mask.sum() > 50:
            self.high_variance_model.train(
                X_train[hv_mask],
                y_margin_train[hv_mask],
                y_cover_train[hv_mask]
            )
        else:
            logger.warning(f"Not enough High Variance samples ({hv_mask.sum()}), training on all data")
            self.high_variance_model.train(X_train, y_margin_train, y_cover_train)

        # Blowout model (Cluster 2)
        logger.info("\n--- Training Blowout Model (Cluster 2) ---")
        blowout_mask = y_cluster_train == CLUSTER_BLOWOUT
        if blowout_mask.sum() > 50:
            self.blowout_model.train(
                X_train[blowout_mask],
                y_margin_train[blowout_mask],
                y_cover_train[blowout_mask],
                y_blowout_train[blowout_mask]
            )
        else:
            logger.warning(f"Not enough Blowout samples ({blowout_mask.sum()}), training on all data")
            self.blowout_model.train(X_train, y_margin_train, y_cover_train, y_blowout_train)

        # 3. Evaluate on test set
        logger.info("\n--- Evaluating on Test Set ---")
        self._evaluate(X_test, X_router_test, y_margin_test, y_cover_test, y_cluster_test)

        return self

    def predict(self, X: np.ndarray, X_router: np.ndarray = None) -> tuple:
        """
        Make predictions using meta-router ensemble.

        Returns:
            (margin_pred, cover_prob, cluster_id, router_confidence, uncertainty, blowout_prob)
        """
        if X_router is None:
            X_router = X

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get cluster predictions and probabilities
        cluster_ids, confidences = self.meta_router.predict(X_router)
        cluster_probs = self.meta_router.get_cluster_probs(X_router)

        # Get predictions from all sub-models
        margin_std, cover_std = self.standard_model.predict(X_scaled)
        margin_hv, cover_hv, uncertainty = self.high_variance_model.predict(X_scaled)
        margin_bo, cover_bo, blowout_prob = self.blowout_model.predict(X_scaled)

        # Combine predictions using cluster probabilities (soft weighting)
        margin_pred = (
            cluster_probs[:, 0] * margin_std +
            cluster_probs[:, 1] * margin_hv +
            cluster_probs[:, 2] * margin_bo
        )

        cover_prob = (
            cluster_probs[:, 0] * cover_std +
            cluster_probs[:, 1] * cover_hv +
            cluster_probs[:, 2] * cover_bo
        )

        return margin_pred, cover_prob, cluster_ids, confidences, uncertainty, blowout_prob

    def _evaluate(self, X_test, X_router_test, y_margin_test, y_cover_test, y_cluster_test):
        """Evaluate model performance on test set."""
        margin_pred, cover_prob, cluster_pred, confidences, uncertainty, blowout_prob = self.predict(X_test, X_router_test)

        # Overall metrics
        mae = mean_absolute_error(y_margin_test, margin_pred)
        rmse = np.sqrt(mean_squared_error(y_margin_test, margin_pred))
        cover_pred = (cover_prob > 0.5).astype(int)
        cover_acc = accuracy_score(y_cover_test, cover_pred)

        logger.info(f"\nOVERALL METRICS:")
        logger.info(f"  MAE: {mae:.2f}")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  Cover Accuracy: {cover_acc:.3f}")

        self.metrics['overall'] = {'mae': mae, 'rmse': rmse, 'cover_acc': cover_acc}

        # Per-cluster metrics
        cluster_names = {0: 'Standard', 1: 'High Variance', 2: 'Blowout'}
        logger.info(f"\nPER-CLUSTER METRICS:")

        for cluster_id, name in cluster_names.items():
            mask = y_cluster_test == cluster_id
            if mask.sum() > 0:
                cluster_mae = mean_absolute_error(y_margin_test[mask], margin_pred[mask])
                cluster_cover_acc = accuracy_score(y_cover_test[mask], cover_pred[mask])
                logger.info(f"  {name}: MAE={cluster_mae:.2f}, Cover Acc={cluster_cover_acc:.3f} (n={mask.sum()})")
                self.metrics[name] = {'mae': cluster_mae, 'cover_acc': cluster_cover_acc, 'n': mask.sum()}

    def save(self, path_prefix: str = 'cfb'):
        """Save the model to disk."""
        model_data = {
            'meta_router': self.meta_router,
            'standard_model': self.standard_model,
            'high_variance_model': self.high_variance_model,
            'blowout_model': self.blowout_model,
            'feature_names': self.feature_names,
            'router_feature_names': self.router_feature_names,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'version': 'V22',
            'trained_at': datetime.now().isoformat(),
            'fcs_teams': list(FCS_TEAMS),  # Save for inference filtering
        }

        output_path = f"{path_prefix}_v22_meta.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved V22 model to {output_path}")

        # Also save to backend folder
        backend_path = f"backend/{output_path}"
        if os.path.exists('backend'):
            with open(backend_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Also saved to {backend_path}")

    @classmethod
    def load(cls, path_prefix: str = 'cfb') -> 'V22MetaRouterModel':
        """Load V22 model from file."""
        model = cls()

        with open(f'{path_prefix}_v22_meta.pkl', 'rb') as f:
            data = pickle.load(f)
            model.meta_router = data['meta_router']
            model.standard_model = data['standard_model']
            model.high_variance_model = data['high_variance_model']
            model.blowout_model = data['blowout_model']
            model.feature_names = data['feature_names']
            model.router_feature_names = data['router_feature_names']
            model.scaler = data['scaler']
            model.metrics = data.get('metrics', {})

        return model


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train V22 Meta-Router Model')
    parser.add_argument('--data', type=str, default='cfb_data_safe.csv',
                       help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='cfb',
                       help='Output path prefix for model files')

    args = parser.parse_args()

    # Load and prepare data
    df = load_and_clean_data(args.data)

    # Train V22 model
    model = V22MetaRouterModel()
    model.train(df)

    # Save model
    model.save(args.output)

    logger.info("\n" + "=" * 60)
    logger.info("V22 META-ROUTER MODEL TRAINING COMPLETE")
    logger.info("=" * 60)

    # Print summary
    if model.metrics:
        logger.info(f"\nFinal Metrics Summary:")
        for key, vals in model.metrics.items():
            if isinstance(vals, dict):
                logger.info(f"  {key}: MAE={vals.get('mae', 'N/A'):.2f}, Cover Acc={vals.get('cover_acc', 'N/A'):.3f}")

    return model


if __name__ == '__main__':
    main()
