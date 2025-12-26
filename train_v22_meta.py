"""
V22 Meta-Router Model Training Pipeline
========================================

Addresses V21 weaknesses identified in error analysis:
1. "Average vs Average" games (MAE 11.35, worst category)
2. Small-school/FCS outliers (Samford MAE 24.0)
3. away_scoring_trend error amplification (1.73x)
4. Blowout game unpredictability

Architecture:
- Meta-Router (Gating Network): Classifies game type and routes to specialized sub-models
- Sub-Model A: Elite matchups (high Elo teams, predictable outcomes)
- Sub-Model B: Average vs Average (enhanced with style matchup features)
- Sub-Model C: Mismatch games (FCS flags, blowout classifier)
- Final ensemble weighted by meta-router confidence
"""

import os
import pickle
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
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
    print("NGBoost not available - using XGBoost only")

import optuna
from optuna.samplers import TPESampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# FCS and small-school teams to flag (high error rates from error_patterns_report.txt)
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
    'North Carolina A&T', 'Robert Morris', 'Bryant', 'Duquesne', 'LIU',
    'Stony Brook', 'Maine', 'New Hampshire', 'Rhode Island', 'Albany',
    'Delaware', 'Elon', 'Hampton', 'Monmouth', 'Richmond', 'Towson',
    'William & Mary', 'Dartmouth', 'Harvard', 'Yale', 'Princeton',
    'Penn', 'Brown', 'Columbia', 'Cornell', 'Drake', 'Dayton', 'Marist',
    'Morehead State', 'Presbyterian', 'San Diego', 'Stetson', 'Valparaiso',
    'Butler', 'St. Thomas', 'Illinois State', 'Indiana State', 'Missouri State',
    'North Dakota', 'North Dakota State', 'Northern Iowa', 'South Dakota',
    'South Dakota State', 'Southern Illinois', 'Western Illinois', 'Youngstown State',
    'Chattanooga', 'East Tennessee State', 'Furman', 'Mercer', 'Samford',
    'VMI', 'Western Carolina', 'Wofford', 'The Citadel', 'Idaho', 'Eastern Illinois'
}

# Toxic features to drop (from error_insights.txt)
TOXIC_FEATURES = [
    'away_scoring_trend',  # 1.73x error amplification
    # 'home_scoring_trend',  # Keep for now, less toxic
]

# V22 Base Features (matching actual columns in cfb_data_safe.csv, minus toxic away_scoring_trend)
V22_BASE_FEATURES = [
    # Elo ratings (keep these - strong predictors)
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

    # Momentum (keep home_scoring_trend, drop away_scoring_trend)
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

# NEW V22 Style Matchup Features
V22_STYLE_FEATURES = [
    'pass_off_vs_pass_def_mismatch',  # Home pass offense vs away pass defense
    'rush_off_vs_rush_def_mismatch',  # Home rush offense vs away rush defense
    'volatility_index',  # Combined scoring/defensive variance
    'style_advantage_home',  # Net style matchup advantage
]

# Game Type Classification Features (for meta-router)
META_ROUTER_FEATURES = [
    'elo_diff_abs',  # Absolute Elo difference
    'avg_elo',  # Average Elo of both teams
    'is_fcs_game',  # FCS team involved
    'is_mismatch',  # Large spread (>14)
    'volatility_index',  # Game volatility
    'is_ranked_matchup',  # Both ranked
    'conference_matchup',  # Same conference
]


# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================

def load_and_clean_data(csv_path: str = 'cfb_data_safe.csv') -> pd.DataFrame:
    """Load data and apply V22 cleaning steps."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    original_count = len(df)
    logger.info(f"Loaded {original_count} games")

    # 1. Flag FCS teams
    df['is_home_fcs'] = df['home_team'].isin(FCS_TEAMS)
    df['is_away_fcs'] = df['away_team'].isin(FCS_TEAMS)
    df['is_fcs_game'] = df['is_home_fcs'] | df['is_away_fcs']

    fcs_count = df['is_fcs_game'].sum()
    logger.info(f"Flagged {fcs_count} games involving FCS teams ({100*fcs_count/len(df):.1f}%)")

    # 2. Drop toxic features
    for feat in TOXIC_FEATURES:
        if feat in df.columns:
            df = df.drop(columns=[feat])
            logger.info(f"Dropped toxic feature: {feat}")

    # 3. Create game classification columns
    df['elo_diff_abs'] = df['elo_diff'].abs() if 'elo_diff' in df.columns else 0
    df['avg_elo'] = (df['home_elo'] + df['away_elo']) / 2 if 'home_elo' in df.columns else 1500
    df['is_mismatch'] = df['vegas_spread'].abs() > 14 if 'vegas_spread' in df.columns else False

    # 4. Classify game types for meta-router training
    df = classify_game_types(df)

    # 5. Ensure is_blowout exists for blowout classifier
    if 'is_blowout' not in df.columns:
        if 'actual_margin' in df.columns:
            df['is_blowout'] = df['actual_margin'].abs() > 21
        elif 'margin' in df.columns:
            df['is_blowout'] = df['margin'].abs() > 21
        else:
            logger.warning("Cannot create is_blowout - no margin column found")
            df['is_blowout'] = False

    blowout_count = df['is_blowout'].sum()
    logger.info(f"Blowout games: {blowout_count} ({100*blowout_count/len(df):.1f}%)")

    return df


def classify_game_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify games into types for meta-router:
    - Type 0: Elite matchup (both teams Elo > 1600, spread < 10)
    - Type 1: Average vs Average (Elo diff < 100, spread < 7)
    - Type 2: Mismatch (spread > 14 or FCS game)
    - Type 3: Standard (everything else)
    """
    conditions = [
        # Elite: Both teams strong, competitive game
        (df['avg_elo'] > 1550) & (df['elo_diff_abs'] < 150) & (df['vegas_spread'].abs() < 10),

        # Average vs Average: Similar mediocre teams (THIS IS WHERE V21 STRUGGLES)
        (df['avg_elo'] < 1550) & (df['elo_diff_abs'] < 100) & (df['vegas_spread'].abs() < 7),

        # Mismatch: Blowout expected
        (df['vegas_spread'].abs() > 14) | df['is_fcs_game'],
    ]

    choices = [0, 1, 2]  # Elite, AvgVsAvg, Mismatch

    df['game_type'] = np.select(conditions, choices, default=3)  # 3 = Standard

    # Log distribution
    type_counts = df['game_type'].value_counts().sort_index()
    type_names = {0: 'Elite', 1: 'AvgVsAvg', 2: 'Mismatch', 3: 'Standard'}
    logger.info("Game type distribution:")
    for t, count in type_counts.items():
        logger.info(f"  {type_names.get(t, t)}: {count} ({100*count/len(df):.1f}%)")

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add style matchup and volatility features.

    These help with "Average vs Average" games where traditional
    Elo-based features don't differentiate well.
    """
    # Check for PPA (Predicted Points Added) columns - use comp_ prefix columns
    has_comp_ppa = 'home_comp_pass_ppa' in df.columns

    if has_comp_ppa:
        # Use comp_ppa columns (composite PPA from our data)
        home_pass_off = df.get('home_comp_pass_ppa', pd.Series([0]*len(df)))
        away_pass_def = df.get('away_comp_def_ppa', pd.Series([0]*len(df)))
        home_rush_off = df.get('home_comp_rush_ppa', pd.Series([0]*len(df)))
        away_rush_def = df.get('away_comp_def_ppa', pd.Series([0]*len(df)))

        away_pass_off = df.get('away_comp_pass_ppa', pd.Series([0]*len(df)))
        home_pass_def = df.get('home_comp_def_ppa', pd.Series([0]*len(df)))
        away_rush_off = df.get('away_comp_rush_ppa', pd.Series([0]*len(df)))
        home_rush_def = df.get('home_comp_def_ppa', pd.Series([0]*len(df)))

        # Style matchup: offense strength vs opposing defense weakness
        df['pass_off_vs_pass_def_mismatch'] = home_pass_off - away_pass_def
        df['rush_off_vs_rush_def_mismatch'] = home_rush_off - away_rush_def

        # Net style advantage for home team
        df['style_advantage_home'] = (
            (home_pass_off - away_pass_def) +
            (home_rush_off - away_rush_def) -
            (away_pass_off - home_pass_def) -
            (away_rush_off - home_rush_def)
        )

        logger.info("Added comp_PPA-based style matchup features")
    else:
        # Fallback: Use comp_off_ppa and comp_def_ppa
        if 'home_comp_off_ppa' in df.columns:
            df['pass_off_vs_pass_def_mismatch'] = (
                df['home_comp_off_ppa'] - df['away_comp_def_ppa']
            )
            df['rush_off_vs_rush_def_mismatch'] = (
                df['home_comp_off_ppa'] * 0.5 - df['away_comp_def_ppa'] * 0.5
            )
            df['style_advantage_home'] = (
                df['home_comp_off_ppa'] - df['away_comp_def_ppa'] -
                (df['away_comp_off_ppa'] - df['home_comp_def_ppa'])
            )
            logger.info("Added comp_off_ppa style matchup features")
        else:
            df['pass_off_vs_pass_def_mismatch'] = 0
            df['rush_off_vs_rush_def_mismatch'] = 0
            df['style_advantage_home'] = 0
            logger.warning("No offensive/defensive data for style features")

    # Volatility Index: How unpredictable is this game?
    # Higher volatility = less predictable outcome
    if 'home_cover_margin' in df.columns:
        home_var = df['home_cover_margin'].rolling(5, min_periods=1).std().fillna(df['home_cover_margin'].std())
        away_var = df['away_cover_margin'].rolling(5, min_periods=1).std().fillna(df['away_cover_margin'].std())
        df['volatility_index'] = (home_var + away_var) / 2
    else:
        # Fallback: Use Elo-based proxy for volatility
        df['volatility_index'] = 10 - (df['elo_diff_abs'] / 100).clip(0, 10)  # Lower Elo diff = higher volatility

    df['volatility_index'] = df['volatility_index'].fillna(5.0)  # Default mid volatility

    logger.info(f"Volatility index range: {df['volatility_index'].min():.2f} - {df['volatility_index'].max():.2f}")

    return df


# =============================================================================
# META-ROUTER MODEL (GATING NETWORK)
# =============================================================================

class MetaRouter:
    """
    Gating network that classifies game types and routes to specialized sub-models.

    Outputs:
    - game_type: 0=Elite, 1=AvgVsAvg, 2=Mismatch, 3=Standard
    - confidence: How confident the router is in classification
    """

    def __init__(self):
        self.classifier = None
        self.feature_names = None
        self.scaler = StandardScaler()

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list):
        """Train the meta-router classifier."""
        self.feature_names = feature_names

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Get unique classes and create label encoder
        unique_classes = np.unique(y)
        self.classes_ = unique_classes
        self.n_classes_ = len(unique_classes)

        # Map labels to 0-indexed
        self.label_map_ = {c: i for i, c in enumerate(unique_classes)}
        self.inverse_label_map_ = {i: c for c, i in self.label_map_.items()}
        y_mapped = np.array([self.label_map_[c] for c in y])

        # Train XGBoost classifier
        self.classifier = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=self.n_classes_,
            random_state=42,
            verbosity=0
        )

        self.classifier.fit(X_scaled, y_mapped)

        # Log training accuracy
        train_pred = self.classifier.predict(X_scaled)
        train_pred_original = np.array([self.inverse_label_map_[p] for p in train_pred])
        train_acc = accuracy_score(y, train_pred_original)
        logger.info(f"Meta-router training accuracy: {train_acc:.3f}")

        return self

    def predict(self, X: np.ndarray) -> tuple:
        """
        Predict game type and confidence.

        Returns:
            (game_types, confidences): Arrays of predictions and confidence scores
        """
        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)
        pred_mapped = np.argmax(probs, axis=1)
        # Map back to original labels
        game_types = np.array([self.inverse_label_map_[p] for p in pred_mapped])
        confidences = np.max(probs, axis=1)

        return game_types, confidences

    def get_routing_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get soft routing weights for ensemble combination.

        Returns probability distribution over game types for each sample.
        Note: These are probabilities over the ORIGINAL class indices (1, 2, 3) not 0-indexed.
        """
        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)

        # Pad to 4 classes if needed (for the 4 sub-models)
        # Original classes might be [1, 2, 3] but we need weights for [0, 1, 2, 3]
        if probs.shape[1] < 4:
            full_probs = np.zeros((probs.shape[0], 4))
            for mapped_idx, original_class in self.inverse_label_map_.items():
                full_probs[:, original_class] = probs[:, mapped_idx]
            return full_probs

        return probs


# =============================================================================
# SPECIALIZED SUB-MODELS
# =============================================================================

class EliteMatchupModel:
    """
    Sub-model for elite matchups (high Elo, competitive games).

    Focus: Precise margin prediction with low variance.
    """

    def __init__(self):
        self.margin_model = None
        self.cover_model = None
        self.calibrator = None

    def train(self, X: np.ndarray, y_margin: np.ndarray, y_cover: np.ndarray):
        """Train elite matchup model."""
        logger.info(f"Training EliteMatchupModel on {len(X)} samples")

        # XGBoost for margin - tuned for lower variance
        self.margin_model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
        self.margin_model.fit(X, y_margin)

        # Cover probability model
        self.cover_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
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


class AvgVsAvgModel:
    """
    Sub-model for "Average vs Average" games - WHERE V21 STRUGGLES.

    Enhanced features:
    - Style matchup features (pass/rush offense vs defense)
    - Volatility index
    - Recent head-to-head if available
    """

    def __init__(self):
        self.margin_model = None
        self.cover_model = None
        self.calibrator = None
        self.uncertainty_model = None  # NGBoost for uncertainty

    def train(self, X: np.ndarray, y_margin: np.ndarray, y_cover: np.ndarray):
        """Train average-vs-average model with enhanced features."""
        logger.info(f"Training AvgVsAvgModel on {len(X)} samples")

        # NGBoost for uncertainty quantification if available
        if HAS_NGBOOST:
            self.uncertainty_model = NGBRegressor(
                Dist=Normal,
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            )
            self.uncertainty_model.fit(X, y_margin)
            logger.info("Trained NGBoost uncertainty model for AvgVsAvg")

        # XGBoost margin model - higher regularization for noisy games
        self.margin_model = XGBRegressor(
            n_estimators=300,
            max_depth=3,  # Shallower to reduce overfitting
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,  # Higher regularization
            reg_lambda=2.0,
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
        """Predict with uncertainty if available."""
        margin = self.margin_model.predict(X)
        cover_prob_raw = self.cover_model.predict_proba(X)[:, 1]
        cover_prob = self.calibrator.transform(cover_prob_raw)

        # Get uncertainty from NGBoost if available
        uncertainty = None
        if self.uncertainty_model is not None:
            pred_dist = self.uncertainty_model.pred_dist(X)
            uncertainty = pred_dist.std()

        return margin, cover_prob, uncertainty


class MismatchModel:
    """
    Sub-model for mismatch games (FCS, large spreads).

    Special handling:
    - Blowout classifier
    - Margin multiplier for predicted blowouts
    - Higher uncertainty estimates
    """

    def __init__(self):
        self.margin_model = None
        self.cover_model = None
        self.blowout_classifier = None
        self.calibrator = None

    def train(self, X: np.ndarray, y_margin: np.ndarray, y_cover: np.ndarray, y_blowout: np.ndarray):
        """Train mismatch model with blowout classification."""
        logger.info(f"Training MismatchModel on {len(X)} samples")

        # Calculate class weight for blowout classifier
        n_blowout = np.sum(y_blowout == 1)
        n_not_blowout = np.sum(y_blowout == 0)
        scale_pos_weight = n_not_blowout / n_blowout if n_blowout > 0 else 1.0
        scale_pos_weight = max(0.1, scale_pos_weight)  # Ensure positive

        # Blowout classifier
        self.blowout_classifier = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=0
        )
        self.blowout_classifier.fit(X, y_blowout)

        blowout_pred = self.blowout_classifier.predict(X)
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
        """Predict with blowout adjustment."""
        margin = self.margin_model.predict(X)
        cover_prob_raw = self.cover_model.predict_proba(X)[:, 1]
        cover_prob = self.calibrator.transform(cover_prob_raw)

        # Blowout prediction
        blowout_prob = self.blowout_classifier.predict_proba(X)[:, 1]
        is_blowout_pred = blowout_prob > 0.5

        # Apply blowout multiplier to margin
        # If predicting blowout, increase margin by 15%
        margin_adjusted = np.where(is_blowout_pred, margin * 1.15, margin)

        return margin_adjusted, cover_prob, blowout_prob


class StandardModel:
    """Standard model for regular games."""

    def __init__(self):
        self.margin_model = None
        self.cover_model = None
        self.calibrator = None

    def train(self, X: np.ndarray, y_margin: np.ndarray, y_cover: np.ndarray):
        """Train standard model."""
        logger.info(f"Training StandardModel on {len(X)} samples")

        self.margin_model = XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
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

        cover_probs = self.cover_model.predict_proba(X)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(cover_probs, y_cover)

        return self

    def predict(self, X: np.ndarray) -> tuple:
        margin = self.margin_model.predict(X)
        cover_prob_raw = self.cover_model.predict_proba(X)[:, 1]
        cover_prob = self.calibrator.transform(cover_prob_raw)
        return margin, cover_prob


# =============================================================================
# V22 ENSEMBLE MODEL
# =============================================================================

class V22MetaRouterModel:
    """
    V22 Meta-Router Ensemble Model.

    Uses a gating network to route games to specialized sub-models,
    then combines predictions using soft routing weights.
    """

    def __init__(self):
        self.meta_router = MetaRouter()
        self.elite_model = EliteMatchupModel()
        self.avgvsavg_model = AvgVsAvgModel()
        self.mismatch_model = MismatchModel()
        self.standard_model = StandardModel()

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
        # Engineer style features if not present
        if 'volatility_index' not in df.columns:
            df = engineer_style_features(df)

        # Base features for sub-models
        available_base = [f for f in V22_BASE_FEATURES if f in df.columns]
        available_style = [f for f in V22_STYLE_FEATURES if f in df.columns]

        self.feature_names = available_base + available_style
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

        # Prepare data
        df = load_and_clean_data() if isinstance(df, str) else df.copy()
        df = engineer_style_features(df)
        df = classify_game_types(df)

        # Check for target columns
        if target_margin not in df.columns:
            if 'margin' in df.columns:
                target_margin = 'margin'
            elif 'Margin' in df.columns:
                target_margin = 'Margin'
            else:
                raise ValueError(f"Target column {target_margin} not found")

        # Create cover target
        df['covered'] = df[target_margin] + df[target_spread] > 0

        # Prepare features
        X_base, X_router = self.prepare_features(df)
        y_margin = df[target_margin].values
        y_cover = df['covered'].astype(int).values
        y_blowout = df['is_blowout'].astype(int).values
        y_game_type = df['game_type'].values

        # Scale base features
        X_base_scaled = self.scaler.fit_transform(X_base)

        # Train-test split (time-based)
        if 'season' in df.columns:
            train_mask = df['season'] < df['season'].max()
            test_mask = df['season'] == df['season'].max()
        else:
            train_mask = np.arange(len(df)) < int(0.8 * len(df))
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
        y_game_type_train = y_game_type[train_mask]
        y_game_type_test = y_game_type[test_mask]

        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # 1. Train Meta-Router
        logger.info("\n--- Training Meta-Router ---")
        self.meta_router.train(X_router_train, y_game_type_train, self.router_feature_names)

        # 2. Train specialized sub-models on filtered data
        logger.info("\n--- Training Elite Matchup Model ---")
        elite_mask = y_game_type_train == 0
        if elite_mask.sum() > 50:
            self.elite_model.train(
                X_train[elite_mask],
                y_margin_train[elite_mask],
                y_cover_train[elite_mask]
            )
        else:
            logger.warning(f"Not enough elite samples ({elite_mask.sum()}), training on all data")
            self.elite_model.train(X_train, y_margin_train, y_cover_train)

        logger.info("\n--- Training AvgVsAvg Model ---")
        avgvsavg_mask = y_game_type_train == 1
        if avgvsavg_mask.sum() > 50:
            self.avgvsavg_model.train(
                X_train[avgvsavg_mask],
                y_margin_train[avgvsavg_mask],
                y_cover_train[avgvsavg_mask]
            )
        else:
            logger.warning(f"Not enough AvgVsAvg samples ({avgvsavg_mask.sum()}), training on all data")
            self.avgvsavg_model.train(X_train, y_margin_train, y_cover_train)

        logger.info("\n--- Training Mismatch Model ---")
        mismatch_mask = y_game_type_train == 2
        if mismatch_mask.sum() > 50:
            self.mismatch_model.train(
                X_train[mismatch_mask],
                y_margin_train[mismatch_mask],
                y_cover_train[mismatch_mask],
                y_blowout_train[mismatch_mask]
            )
        else:
            logger.warning(f"Not enough mismatch samples ({mismatch_mask.sum()}), training on all data")
            self.mismatch_model.train(X_train, y_margin_train, y_cover_train, y_blowout_train)

        logger.info("\n--- Training Standard Model ---")
        standard_mask = y_game_type_train == 3
        if standard_mask.sum() > 50:
            self.standard_model.train(
                X_train[standard_mask],
                y_margin_train[standard_mask],
                y_cover_train[standard_mask]
            )
        else:
            self.standard_model.train(X_train, y_margin_train, y_cover_train)

        # 3. Evaluate on test set
        logger.info("\n--- Evaluating on Test Set ---")
        self._evaluate(X_test, X_router_test, y_margin_test, y_cover_test, y_game_type_test)

        return self

    def predict(self, X: np.ndarray, X_router: np.ndarray = None) -> tuple:
        """
        Make predictions using meta-router ensemble.

        Returns:
            (margin_pred, cover_prob, game_type, routing_confidence, uncertainty)
        """
        if X_router is None:
            X_router = X  # Use same features if router features not provided

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get routing weights (soft assignment)
        routing_weights = self.meta_router.get_routing_weights(X_router)
        game_types, confidences = self.meta_router.predict(X_router)

        # Get predictions from all sub-models
        margin_elite, cover_elite = self.elite_model.predict(X_scaled)
        margin_avg, cover_avg, uncertainty_avg = self.avgvsavg_model.predict(X_scaled)
        margin_mismatch, cover_mismatch, blowout_prob = self.mismatch_model.predict(X_scaled)
        margin_standard, cover_standard = self.standard_model.predict(X_scaled)

        # Combine using routing weights
        margins = np.stack([margin_elite, margin_avg, margin_mismatch, margin_standard], axis=1)
        covers = np.stack([cover_elite, cover_avg, cover_mismatch, cover_standard], axis=1)

        margin_pred = np.sum(margins * routing_weights, axis=1)
        cover_prob = np.sum(covers * routing_weights, axis=1)

        # Uncertainty estimate
        uncertainty = uncertainty_avg if uncertainty_avg is not None else np.full(len(X), 7.0)

        return margin_pred, cover_prob, game_types, confidences, uncertainty

    def _evaluate(self, X_test, X_router_test, y_margin_test, y_cover_test, y_game_type_test):
        """Evaluate model performance on test set."""
        margin_pred, cover_prob, game_types_pred, confidences, uncertainty = self.predict(X_test, X_router_test)

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

        # Per-game-type metrics
        type_names = {0: 'Elite', 1: 'AvgVsAvg', 2: 'Mismatch', 3: 'Standard'}
        logger.info(f"\nPER-GAME-TYPE METRICS:")

        for game_type, name in type_names.items():
            mask = y_game_type_test == game_type
            if mask.sum() > 0:
                type_mae = mean_absolute_error(y_margin_test[mask], margin_pred[mask])
                type_cover_acc = accuracy_score(y_cover_test[mask], cover_pred[mask])
                logger.info(f"  {name}: MAE={type_mae:.2f}, Cover Acc={type_cover_acc:.3f} (n={mask.sum()})")
                self.metrics[name] = {'mae': type_mae, 'cover_acc': type_cover_acc, 'n': mask.sum()}

    def save(self, path_prefix: str):
        """Save the model to disk."""
        model_data = {
            'meta_router': self.meta_router,
            'elite_model': self.elite_model,
            'avgvsavg_model': self.avgvsavg_model,
            'mismatch_model': self.mismatch_model,
            'standard_model': self.standard_model,
            'feature_names': self.feature_names,
            'router_feature_names': self.router_feature_names,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'version': 'V22',
            'trained_at': datetime.now().isoformat()
        }

        with open(f"{path_prefix}_v22_meta.pkl", 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved V22 model to {path_prefix}_v22_meta.pkl")


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
    parser.add_argument('--compare-v19', action='store_true',
                       help='Compare against V19 baseline')

    args = parser.parse_args()

    # Load and prepare data
    df = load_and_clean_data(args.data)
    df = engineer_style_features(df)

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
