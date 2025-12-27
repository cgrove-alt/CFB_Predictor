"""
V22 Meta-Router Model Training Pipeline (FBS-Only)
===================================================

Addresses V21 weaknesses identified in error analysis:
1. "Average vs Average" games (MAE 11.35, worst category)
2. Small-school/FCS outliers - NOW ELIMINATED via FBS allowlist
3. scoring_trend features - RE-ENABLED for clean FBS-only data
4. Blowout game unpredictability

**THE GOLDEN RULE**: Only train on FBS vs FBS matchups.
If a team is NOT in the FBS allowlist, the game is excluded.

Architecture:
- Meta-Router (Gating Network): Classifies game cluster and routes to specialized sub-models
  - Cluster 0: Standard Game (use standard XGBoost model)
  - Cluster 1: High Variance/Mismatch (use Graph Model + Uncertainty Penalty)
  - Cluster 2: Blowout Risk (Margin > 24, apply 1.15x multiplier)
- Sub-Model Ensemble: XGBoost, Graph-aware model, Blowout Classifier
- Conference matchup feature for P4 vs G5 differentiation
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

# Import FBS teams allowlist
from fbs_teams import (
    get_fbs_teams, is_fbs_game, is_fbs_team, normalize_team_name,
    get_team_conference, get_conference_matchup, get_conference_tier,
    TEAM_TO_CONFERENCE
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# FBS Allowlist - THE GOLDEN RULE
# Only games where BOTH teams are in this list are used for training/prediction
# This replaces the old FCS blacklist approach for cleaner, more reliable data
FBS_TEAMS = get_fbs_teams()

# Toxic features - REMOVED: scoring_trend features are now RE-ENABLED for FBS-only data
# The 1.73x error amplification was caused by FCS games introducing noise
# With FBS-only filtering, these features provide valuable momentum signal
TOXIC_FEATURES = []  # Empty - scoring trends are safe for FBS-only data

# Game Cluster definitions for Meta-Router
CLUSTER_STANDARD = 0      # Standard game - use base XGBoost
CLUSTER_HIGH_VARIANCE = 1 # High variance/mismatch - use uncertainty-aware model
CLUSTER_BLOWOUT = 2       # Blowout risk (margin > 24)
CLUSTER_AVG_VS_AVG = 3    # Average vs Average (both Elo 1400-1600) - specialized model

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
    # Momentum - RE-ENABLED: scoring_trend now safe for FBS-only data
    'home_streak', 'away_streak', 'streak_diff',
    'home_ats', 'away_ats', 'ats_diff',
    'home_elo_momentum', 'away_elo_momentum', 'elo_momentum_diff',
    'home_scoring_trend', 'away_scoring_trend',  # Both re-enabled for FBS-only
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

# V22 Conference Matchup Features (new for FBS-only model)
V22_CONFERENCE_FEATURES = [
    'home_is_p4',                 # Home team is Power 4 conference
    'away_is_p4',                 # Away team is Power 4 conference
    'is_conference_game',         # Same conference matchup
    'is_p4_vs_p4',                # Power 4 vs Power 4
    'is_p4_vs_g5',                # Power 4 vs Group of 5
    'is_g5_vs_g5',                # Group of 5 vs Group of 5
]

# Meta-Router Input Features (game context for routing)
META_ROUTER_FEATURES = [
    'week',                       # Week of season
    'elo_diff',                   # Elo differential (absolute value)
    'matchup_volatility',         # Average volatility of both teams
    'vegas_spread',               # Spread as indicator of expected game type
    'is_p4_vs_p4',                # Power 4 matchups (higher predictability)
    'is_p4_vs_g5',                # Talent gap indicator
]

# Features to GATE (exclude) for Avg-vs-Avg games - error amplifiers identified in analysis
# These features cause 1.2-1.3x error amplification in evenly-matched games
AVG_VS_AVG_GATED_FEATURES = [
    'home_ats', 'away_ats', 'ats_diff',           # 1.27x error amplifier - unreliable for avg teams
    'hfa_diff',                                    # 1.23x error amplifier - minimal impact between similar teams
    'home_streak', 'away_streak', 'streak_diff',  # 1.22x error amplifier - less predictive for avg teams
    'elo_vs_spread',                              # 48.9% of high-error games - noise source for close games
    'dominant_home', 'dominant_away',             # Never true for average teams by definition
]


# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================

def load_and_clean_data(csv_path: str = 'cfb_data_safe.csv') -> pd.DataFrame:
    """
    Load data and apply V22 cleaning steps.

    **THE GOLDEN RULE**: Only keep games where BOTH teams are FBS.
    Uses allowlist approach (whitelist) instead of blacklist for cleaner data.
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    original_count = len(df)
    logger.info(f"Loaded {original_count} games")

    # Normalize team names first
    df['home_team_normalized'] = df['home_team'].apply(normalize_team_name)
    df['away_team_normalized'] = df['away_team'].apply(normalize_team_name)

    # 1. STRICT FBS FILTERING (THE GOLDEN RULE)
    # Only keep games where BOTH teams are in the FBS allowlist
    fbs_home = df['home_team_normalized'].isin(FBS_TEAMS)
    fbs_away = df['away_team_normalized'].isin(FBS_TEAMS)
    fbs_both = fbs_home & fbs_away

    non_fbs_count = (~fbs_both).sum()
    df = df[fbs_both].reset_index(drop=True)

    logger.info(f"REMOVED {non_fbs_count} non-FBS games ({100*non_fbs_count/original_count:.1f}%)")
    logger.info(f"Remaining: {len(df)} FBS vs FBS games only")

    # Log which teams were filtered out
    if non_fbs_count > 0:
        all_teams_original = set(pd.read_csv(csv_path)['home_team'].unique()) | set(pd.read_csv(csv_path)['away_team'].unique())
        non_fbs_teams = all_teams_original - FBS_TEAMS
        if len(non_fbs_teams) <= 20:
            logger.info(f"Filtered teams (non-FBS): {sorted(non_fbs_teams)}")
        else:
            logger.info(f"Filtered {len(non_fbs_teams)} non-FBS teams from dataset")

    # 2. DROP toxic features (if any - currently empty for FBS-only)
    for feat in TOXIC_FEATURES:
        if feat in df.columns:
            df = df.drop(columns=[feat])
            logger.info(f"DROPPED toxic feature: {feat}")

    # 3. Engineer V22 features (style, volatility, conference matchup)
    df = engineer_v22_features(df)

    # 4. Add conference matchup features
    df = add_conference_matchup_features(df)

    # 5. Create game cluster labels for meta-router training
    df = create_game_clusters(df)

    # Clean up temporary columns
    df = df.drop(columns=['home_team_normalized', 'away_team_normalized'], errors='ignore')

    return df


def add_conference_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add conference-based features for FBS-only model.

    Features:
    - home_is_p4, away_is_p4: Power 4 conference flags
    - is_conference_game: Same conference matchup
    - is_p4_vs_p4, is_p4_vs_g5, is_g5_vs_g5: Matchup tier indicators
    - conference_matchup: String like "SEC vs SEC", "P4 vs G5"
    """
    logger.info("Adding conference matchup features...")

    # Get conferences for each team
    home_teams = df['home_team'].apply(normalize_team_name)
    away_teams = df['away_team'].apply(normalize_team_name)

    home_conf = home_teams.apply(get_team_conference)
    away_conf = away_teams.apply(get_team_conference)

    home_tier = home_conf.apply(lambda c: get_conference_tier(c) if c else 'Unknown')
    away_tier = away_conf.apply(lambda c: get_conference_tier(c) if c else 'Unknown')

    # Power 4 flags
    df['home_is_p4'] = (home_tier == 'P4').astype(int)
    df['away_is_p4'] = (away_tier == 'P4').astype(int)

    # Same conference
    df['is_conference_game'] = (home_conf == away_conf).astype(int)

    # Matchup tier indicators
    df['is_p4_vs_p4'] = ((home_tier == 'P4') & (away_tier == 'P4')).astype(int)
    df['is_p4_vs_g5'] = (
        ((home_tier == 'P4') & (away_tier == 'G5')) |
        ((home_tier == 'G5') & (away_tier == 'P4'))
    ).astype(int)
    df['is_g5_vs_g5'] = ((home_tier == 'G5') & (away_tier == 'G5')).astype(int)

    # String conference matchup (for debugging/logging)
    df['conference_matchup'] = df.apply(
        lambda row: get_conference_matchup(row['home_team'], row['away_team']),
        axis=1
    )

    # Log distribution
    matchup_counts = df['conference_matchup'].value_counts()
    logger.info(f"Conference matchup distribution (top 10):")
    for matchup, count in matchup_counts.head(10).items():
        logger.info(f"  {matchup}: {count} ({100*count/len(df):.1f}%)")

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
    - Cluster 3: Average vs Average (both Elo 1400-1600, elo_diff < 100)
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

    # Get Elo columns
    home_elo = df['home_pregame_elo'] if 'home_pregame_elo' in df.columns else 1500
    away_elo = df['away_pregame_elo'] if 'away_pregame_elo' in df.columns else 1500
    elo_diff = df['elo_diff'].abs() if 'elo_diff' in df.columns else abs(home_elo - away_elo)

    # Initialize all as Standard
    df['game_cluster'] = CLUSTER_STANDARD

    # Cluster 3: Average vs Average (HIGHEST PRIORITY after blowouts)
    # Both teams Elo 1400-1600, minimal Elo differential
    # This is the worst-performing segment (MAE 11.35, 55% direction accuracy)
    avg_vs_avg_mask = (
        (home_elo >= 1400) & (home_elo <= 1600) &
        (away_elo >= 1400) & (away_elo <= 1600) &
        (elo_diff < 100)
    )
    df.loc[avg_vs_avg_mask, 'game_cluster'] = CLUSTER_AVG_VS_AVG

    # Cluster 2: Blowout Risk (margin > 24) - overrides Avg-vs-Avg if blowout occurred
    blowout_mask = margin_abs > 24
    df.loc[blowout_mask, 'game_cluster'] = CLUSTER_BLOWOUT

    # Cluster 1: High Variance/Mismatch
    # Criteria: Close spread (<7) AND high volatility (>12) AND NOT blowout AND NOT avg-vs-avg
    high_variance_mask = (
        (spread_abs < 7) &
        (volatility > 12) &
        (~blowout_mask) &
        (~avg_vs_avg_mask)
    )
    df.loc[high_variance_mask, 'game_cluster'] = CLUSTER_HIGH_VARIANCE

    # Log distribution
    cluster_counts = df['game_cluster'].value_counts().sort_index()
    cluster_names = {0: 'Standard', 1: 'High Variance', 2: 'Blowout', 3: 'Avg vs Avg'}
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

    def train(self, X: np.ndarray, y_margin: np.ndarray, y_cover: np.ndarray, y_blowout: np.ndarray,
              X_full: np.ndarray = None, y_blowout_full: np.ndarray = None):
        """Train blowout classifier and margin models.

        Note: Blowout classifier needs both classes to train properly.
        If X_full and y_blowout_full are provided, use those for the classifier.
        Otherwise, skip classifier training if only one class present.
        """
        logger.info(f"Training BlowoutClassifier on {len(X)} samples")

        # Use full data for classifier if provided, otherwise use passed data
        X_clf = X_full if X_full is not None else X
        y_clf = y_blowout_full if y_blowout_full is not None else y_blowout

        # Check if we have both classes
        unique_classes = np.unique(y_clf)
        if len(unique_classes) < 2:
            logger.warning(f"Only one class in y_blowout ({unique_classes}), using dummy classifier")
            # Create a dummy classifier that always predicts the single class
            self.classifier = None
            self._single_class = unique_classes[0]
        else:
            # Calculate class weight
            n_blowout = np.sum(y_clf == 1)
            n_not_blowout = np.sum(y_clf == 0)
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
            self.classifier.fit(X_clf, y_clf)

            blowout_pred = self.classifier.predict(X_clf)
            blowout_acc = accuracy_score(y_clf, blowout_pred)
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

        # Blowout probability (handle dummy classifier case)
        if self.classifier is not None:
            blowout_prob = self.classifier.predict_proba(X)[:, 1]
        else:
            # Dummy classifier - use single class probability
            blowout_prob = np.full(len(X), float(getattr(self, '_single_class', 1)))

        # Apply 1.15x multiplier if blowout predicted with >60% confidence
        blowout_adjustment = np.where(blowout_prob > 0.6, 1.15, 1.0)
        margin_adjusted = margin * blowout_adjustment

        return margin_adjusted, cover_prob, blowout_prob


class AvgVsAvgModel:
    """
    Specialized model for Average vs Average games (Cluster 3).

    These games have the worst baseline performance (MAE 11.35, 55% direction accuracy).
    Key strategy: GATE error-amplifying features and use conservative hyperparameters.

    Features gated: home_ats, hfa_diff, home_streak, elo_vs_spread, dominant_*
    Prioritized: Style matchup features (pass_off_vs_pass_def, etc.)
    """

    def __init__(self):
        self.margin_model = None
        self.cover_model = None
        self.calibrator = None
        self.feature_mask = None
        self.active_feature_names = None

    def train(self, X: np.ndarray, y_margin: np.ndarray, y_cover: np.ndarray,
              feature_names: list):
        """Train Avg-vs-Avg model with feature gating."""
        logger.info(f"Training AvgVsAvgModel on {len(X)} samples")

        # Create feature mask - exclude error-amplifying features
        self.feature_mask = np.array([
            f not in AVG_VS_AVG_GATED_FEATURES for f in feature_names
        ])
        X_gated = X[:, self.feature_mask]

        self.active_feature_names = [f for f, mask in zip(feature_names, self.feature_mask) if mask]
        n_gated = (~self.feature_mask).sum()
        logger.info(f"Gated {n_gated} error-amplifying features, using {len(self.active_feature_names)} features")

        # XGBoost with conservative settings (high regularization to avoid overfitting to noise)
        self.margin_model = XGBRegressor(
            n_estimators=200,
            max_depth=3,          # Shallower - reduce overfitting
            learning_rate=0.02,   # Slower learning - more stable
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=1.0,        # Higher L1 regularization
            reg_lambda=3.0,       # Higher L2 regularization
            random_state=42,
            verbosity=0
        )
        self.margin_model.fit(X_gated, y_margin)

        # Cover model with same conservative settings
        self.cover_model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            reg_lambda=3.0,
            random_state=42,
            verbosity=0
        )
        self.cover_model.fit(X_gated, y_cover)

        # Calibrate cover probabilities
        cover_probs = self.cover_model.predict_proba(X_gated)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(cover_probs, y_cover)

        return self

    def predict(self, X: np.ndarray) -> tuple:
        """Predict with feature gating applied."""
        X_gated = X[:, self.feature_mask]
        margin = self.margin_model.predict(X_gated)
        cover_prob_raw = self.cover_model.predict_proba(X_gated)[:, 1]
        cover_prob = self.calibrator.transform(cover_prob_raw)
        return margin, cover_prob


# =============================================================================
# V22 META-ROUTER ENSEMBLE MODEL
# =============================================================================

class V22MetaRouterModel:
    """
    V22.1 Meta-Router Ensemble Model with Avg-vs-Avg specialization.

    Uses a gating network to classify games into clusters:
    - Cluster 0: Standard -> XGBoost
    - Cluster 1: High Variance -> Uncertainty-aware model
    - Cluster 2: Blowout Risk -> Blowout classifier with 1.15x multiplier
    - Cluster 3: Avg vs Avg -> Specialized model with feature gating

    Final prediction weighted by meta-router confidence (soft ensemble).
    """

    def __init__(self):
        self.meta_router = MetaRouter()
        self.standard_model = StandardXGBoostModel()
        self.high_variance_model = HighVarianceModel()
        self.blowout_model = BlowoutClassifier()
        self.avg_vs_avg_model = AvgVsAvgModel()  # NEW: Cluster 3

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
        available_conf = [f for f in V22_CONFERENCE_FEATURES if f in df.columns]

        self.feature_names = available_base + available_style + available_vol + available_conf
        logger.info(f"Using {len(self.feature_names)} features for sub-models")
        logger.info(f"  Base: {len(available_base)}, Style: {len(available_style)}, "
                   f"Volatility: {len(available_vol)}, Conference: {len(available_conf)}")

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
              target_spread: str = 'vegas_spread', skip_internal_split: bool = False):
        """
        Train the V22 meta-router ensemble.

        Args:
            df: Training data DataFrame
            target_margin: Name of margin column
            target_spread: Name of spread column
            skip_internal_split: If True, use all data for training (no train/test split).
                                 Use this for walk-forward evaluation where external split is done.
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
            df = add_conference_matchup_features(df)
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

        # Train-test split (time-based) or use all data
        if skip_internal_split:
            # Use all data for training (for walk-forward evaluation)
            train_mask = np.ones(len(df), dtype=bool)
            test_mask = np.zeros(len(df), dtype=bool)
        elif 'season' in df.columns and len(df['season'].unique()) > 1:
            train_mask = df['season'] < df['season'].max()
            test_mask = df['season'] == df['season'].max()
        else:
            # Single season or no season column - use 80/20 split
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
        # Note: Blowout classifier needs FULL data to distinguish blowouts from non-blowouts
        # But margin/cover models are trained on blowout-specific data
        logger.info("\n--- Training Blowout Model (Cluster 2) ---")
        blowout_mask = y_cluster_train == CLUSTER_BLOWOUT
        if blowout_mask.sum() > 50:
            self.blowout_model.train(
                X_train[blowout_mask],
                y_margin_train[blowout_mask],
                y_cover_train[blowout_mask],
                y_blowout_train[blowout_mask],
                X_full=X_train,  # Full data for classifier
                y_blowout_full=y_blowout_train  # Full labels for classifier
            )
        else:
            logger.warning(f"Not enough Blowout samples ({blowout_mask.sum()}), training on all data")
            self.blowout_model.train(X_train, y_margin_train, y_cover_train, y_blowout_train)

        # Avg vs Avg model (Cluster 3) - NEW
        logger.info("\n--- Training Avg vs Avg Model (Cluster 3) ---")
        avg_mask = y_cluster_train == CLUSTER_AVG_VS_AVG
        if avg_mask.sum() > 50:
            self.avg_vs_avg_model.train(
                X_train[avg_mask],
                y_margin_train[avg_mask],
                y_cover_train[avg_mask],
                self.feature_names  # Pass feature names for gating
            )
        else:
            logger.warning(f"Not enough Avg-vs-Avg samples ({avg_mask.sum()}), training on all data")
            self.avg_vs_avg_model.train(X_train, y_margin_train, y_cover_train, self.feature_names)

        # 3. Evaluate on test set (skip if no test data)
        if len(X_test) > 0:
            logger.info("\n--- Evaluating on Test Set ---")
            self._evaluate(X_test, X_router_test, y_margin_test, y_cover_test, y_cluster_test)
        else:
            logger.info("\n--- Skipping evaluation (no test data) ---")

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

        # Get predictions from all 4 sub-models
        margin_std, cover_std = self.standard_model.predict(X_scaled)
        margin_hv, cover_hv, uncertainty = self.high_variance_model.predict(X_scaled)
        margin_bo, cover_bo, blowout_prob = self.blowout_model.predict(X_scaled)
        margin_avg, cover_avg = self.avg_vs_avg_model.predict(X_scaled)  # Cluster 3

        # Combine predictions using cluster probabilities (soft weighting - 4 clusters)
        # Handle case where cluster_probs might only have 3 columns initially
        if cluster_probs.shape[1] >= 4:
            margin_pred = (
                cluster_probs[:, 0] * margin_std +
                cluster_probs[:, 1] * margin_hv +
                cluster_probs[:, 2] * margin_bo +
                cluster_probs[:, 3] * margin_avg
            )
            cover_prob = (
                cluster_probs[:, 0] * cover_std +
                cluster_probs[:, 1] * cover_hv +
                cluster_probs[:, 2] * cover_bo +
                cluster_probs[:, 3] * cover_avg
            )
        else:
            # Fallback for 3-cluster case
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
        cluster_names = {0: 'Standard', 1: 'High Variance', 2: 'Blowout', 3: 'Avg vs Avg'}
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
            'avg_vs_avg_model': self.avg_vs_avg_model,  # NEW: Cluster 3
            'feature_names': self.feature_names,
            'router_feature_names': self.router_feature_names,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'version': 'V22.1-FBS-Only-AvgVsAvg',  # Updated version
            'trained_at': datetime.now().isoformat(),
            'fbs_teams': list(FBS_TEAMS),  # Save FBS allowlist for inference filtering
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
            # Load Avg-vs-Avg model if present (V22.1+), otherwise create new
            model.avg_vs_avg_model = data.get('avg_vs_avg_model', AvgVsAvgModel())
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
