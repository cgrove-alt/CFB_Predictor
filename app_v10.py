"""
Sharp Picks CFB - Self-Learning Spread Error Model

This app uses the V16 Self-Learning model which:
1. Predicts SPREAD ERROR (how wrong Vegas will be)
2. Uses 68 safe pre-game features (58 V15 + 10 uncertainty features)
3. Learns from its own prediction errors via SHAP analysis
4. Includes uncertainty indicators: is_pickem, is_mismatch, historical team errors
5. Optimized with Optuna (150 XGB + 75 HGB trials)
6. Validated: wins 11 of 15 weeks vs V15, 0.4% lower MAE

Key insight: We don't try to predict the raw margin.
Instead, we predict how far off Vegas's spread is likely to be,
AND we know when the model is in a harder-to-predict situation.
"""

import logging
import os
import sys
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# Set up path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CFBD_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PASSWORD PROTECTION
# =============================================================================
def check_password():
    """Simple password protection for the app."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 Login Required")
        st.markdown("Enter your password to access Sharp Picks.")
        password = st.text_input("Password:", type="password")
        if st.button("Login", type="primary"):
            if password == st.secrets.get("APP_PASSWORD", ""):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

# =============================================================================
# CONFIGURATION
# =============================================================================
# V19 UPDATE: Stricter thresholds based on analysis
# - Only HIGH/MEDIUM-HIGH bets are profitable (+55.7% and +18.4% ROI)
# - MEDIUM/LOW bets drag down portfolio (-6.8% ROI for LOW)
SPREAD_ERROR_THRESHOLD = 4.5  # V19: Increased from 3.0 to 4.5 for stricter filtering
KELLY_FRACTION = 0.25
VARIANCE_THRESHOLD = 7.0

# V19: Game type filters - skip unprofitable game types
# Based on analysis: Average vs Average games (Elo diff < 100) have only 55% accuracy
SKIP_PICK_EM_GAMES = True         # |spread| < 3 - high variance, low edge
SKIP_AVG_VS_AVG_GAMES = True      # |elo_diff| < 100 - main source of losses (51% of games)
SKIP_EARLY_SEASON_GAMES = True    # week <= 2 - data staleness
SHOW_PASS_RECOMMENDATIONS = True  # Show PASS for games that should be skipped

# Safe features used by V16 model (NO LEAKAGE - 68 features)
# V16 = V15 (58 features) + 10 uncertainty features learned from error analysis
SAFE_FEATURES_V15 = [
    # Core Elo features
    'home_pregame_elo', 'away_pregame_elo', 'elo_diff',
    # Rolling stats (last 5 games)
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',
    # Home field advantage
    'home_team_hfa', 'hfa_diff',
    # Rest and scheduling
    'rest_diff',
    # Vegas lines
    'line_movement',
    'large_favorite', 'large_underdog', 'close_game',
    # Streaks
    'home_streak', 'away_streak', 'streak_diff',
    # ATS history
    'home_ats', 'away_ats', 'ats_diff',
    # Momentum features
    'home_elo_momentum', 'away_elo_momentum', 'elo_momentum_diff',
    'home_scoring_trend', 'away_scoring_trend',
    # Derived features
    'elo_vs_spread', 'rest_spread_interaction',
    'home_short_rest', 'away_short_rest',
    'expected_total',
    'west_coast_early', 'home_lookahead', 'away_lookahead',
    # V15: PPA efficiency features (season composites - pre-game)
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',
    'home_comp_pass_ppa', 'away_comp_pass_ppa',
    'home_comp_rush_ppa', 'away_comp_rush_ppa',
    'home_comp_success', 'away_comp_success',
    'home_comp_epa', 'away_comp_epa',
    'home_comp_ypp', 'away_comp_ypp',
    'pass_efficiency_diff',
    # V15: Composite features
    'matchup_efficiency',
    'home_pass_rush_balance', 'away_pass_rush_balance',
    'success_rate_diff',
    'elo_efficiency_interaction',
    'momentum_strength',
    'dominant_home', 'dominant_away',
    'rest_favorite_interaction',
    'has_line_movement',
]

# V16: Uncertainty features - learned from SHAP error analysis
V16_UNCERTAINTY_FEATURES = [
    'is_pickem',                    # Pick-em games are harder to predict
    'is_mismatch',                  # Large Elo mismatches are more predictable
    'is_early_season',              # Early season has cold start issues
    'is_rivalry_week',              # Rivalry weeks have upsets
    'home_team_historical_error',   # Some teams are harder to predict
    'away_team_historical_error',   # Some teams are harder to predict
    'spread_bucket_error',          # Historical error by spread magnitude
    'feature_completeness',         # More data = better prediction
    'is_post_bye',                  # Post-bye teams are harder
    'is_short_rest',                # Short rest games are volatile
]

# Full feature list for V16 (68 features)
SAFE_FEATURES = SAFE_FEATURES_V15 + V16_UNCERTAINTY_FEATURES

# Page config - must be first Streamlit command
st.set_page_config(page_title="Sharp Picks CFB", page_icon="$", layout="wide")

# Check password before showing any content
check_password()

# =============================================================================
# CUSTOM CSS
# =============================================================================
def get_custom_css():
    return """
    <style>
    .stApp {
        background-color: #0F172A;
    }

    .bet-card {
        background: linear-gradient(145deg, #1E293B, #334155);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        border-left: 5px solid;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .bet-card.buy {
        border-color: #10B981;
    }
    .bet-card.fade {
        border-color: #EF4444;
    }

    .bet-card.over {
        border-color: #22C55E;
        background: linear-gradient(145deg, #1E293B, #1a3328);
    }

    .bet-card.under {
        border-color: #3B82F6;
        background: linear-gradient(145deg, #1E293B, #1a2838);
    }

    .confidence-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .confidence-high {
        background: #10B981;
        color: white;
    }
    .confidence-medium-high {
        background: #34D399;
        color: black;
    }
    .confidence-medium {
        background: #F59E0B;
        color: black;
    }
    .confidence-low {
        background: #FB923C;
        color: black;
    }
    .confidence-very-low {
        background: #EF4444;
        color: white;
    }

    .bet-instruction {
        font-size: 26px;
        font-weight: 700;
        margin: 12px 0 4px 0;
        color: #F8FAFC !important;
    }
    .bet-instruction-hero {
        font-size: 32px;
        color: #F8FAFC !important;
    }
    .opponent {
        color: #B4C6DC;  /* Improved contrast from #94A3B8 */
        font-size: 14px;
        margin-bottom: 16px;
    }
    .bet-amount {
        font-size: 36px;
        font-weight: 700;
        color: #10B981;
        margin: 8px 0;
    }
    .bet-amount-hero {
        font-size: 42px;
    }
    .win-prob {
        color: #CBD5E1;
        font-size: 16px;
    }
    .spread-error {
        color: #F59E0B;
        font-size: 14px;
        margin-top: 8px;
    }

    .hero-section {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        padding: 24px;
        border-radius: 20px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 14px;
        font-weight: 600;
        color: #10B981;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }

    .pass-item {
        color: #64748B;
        padding: 8px 0;
        border-bottom: 1px solid #334155;
        font-size: 14px;
    }

    .metric-card {
        background: #1E293B;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 12px;
        color: #B4C6DC;  /* Improved contrast from #94A3B8 */
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .model-info {
        background: #1E3A5F;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        border-left: 4px solid #3B82F6;
    }
    .model-info-text {
        color: #93C5FD;
        font-size: 13px;
    }

    /* ==============================================
       UX FIX: Improve text contrast for readability
       ============================================== */

    /* Fix caption contrast - improved from #94A3B8 */
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #B4C6DC !important;
    }

    /* Fix tab labels visibility */
    .stTabs [data-baseweb="tab-list"] button {
        color: #CBD5E1 !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #F8FAFC !important;
        font-weight: 600;
    }

    /* Fix st.metric() component contrast */
    [data-testid="stMetricLabel"] {
        color: #B4C6DC !important;
    }
    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }

    /* Fix general markdown text */
    .stMarkdown p {
        color: #CBD5E1;
    }

    /* Fix sidebar text */
    [data-testid="stSidebar"] .stMarkdown p {
        color: #CBD5E1 !important;
    }

    /* Fix main title (st.title) contrast */
    h1, [data-testid="stHeading"] h1 {
        color: #F8FAFC !important;
    }

    /* Fix expander header contrast - updated selectors for newer Streamlit */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    .st-emotion-cache-p5msec,
    details summary {
        color: #E2E8F0 !important;
    }

    /* Fix expander text (legacy selector kept for compatibility) */
    .streamlit-expanderHeader {
        color: #F8FAFC !important;
    }
    .streamlit-expanderContent p {
        color: #CBD5E1 !important;
    }

    /* Fix select box labels */
    .stSelectbox label, .stMultiSelect label {
        color: #CBD5E1 !important;
    }

    /* Fix info/warning boxes text */
    .stAlert p {
        color: #1E293B !important;
    }

    /* Change Full Refresh button from red to orange (less alarming) */
    button[kind="primary"] {
        background-color: #F59E0B !important;
        border-color: #F59E0B !important;
    }
    button[kind="primary"]:hover {
        background-color: #D97706 !important;
        border-color: #D97706 !important;
    }

    /* Game status badges */
    .game-status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-left: 8px;
    }
    .game-status-live {
        background: #EF4444;
        color: #FFFFFF;
        animation: pulse 1.5s infinite;
    }
    .game-status-soon {
        background: #F59E0B;
        color: #1E293B;
    }
    .game-status-today {
        background: #3B82F6;
        color: #FFFFFF;
    }
    .game-status-future {
        background: #475569;
        color: #E2E8F0;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    </style>
    """

st.markdown(get_custom_css(), unsafe_allow_html=True)

# =============================================================================
# API SETUP
# =============================================================================
CFBD_BASE_URL = "https://api.collegefootballdata.com"

def get_api_headers():
    """Get authorization headers for CFBD API."""
    if not CFBD_API_KEY:
        st.error(
            "CFBD_API_KEY not set! Please set the environment variable:\n"
            "`export CFBD_API_KEY='your_key_here'`"
        )
        st.stop()
    return {'Authorization': f'Bearer {CFBD_API_KEY}'}

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================
class V18StackingWrapper:
    """Wrapper to make V18 stacking ensemble work like a simple sklearn model."""

    EXPECTED_FEATURES = 58  # V18 was trained on 58 features (V15 only, no V16 uncertainty)

    def __init__(self, model_dict):
        self.base_models = model_dict['base_models']
        self.meta_learner = model_dict['meta_learner']
        self.model_names = model_dict.get('model_names', list(self.base_models.keys()))

    def predict(self, X):
        """Generate predictions using stacking ensemble."""
        import numpy as np

        # Validate feature dimensions
        actual_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        if actual_features != self.EXPECTED_FEATURES:
            raise ValueError(
                f"V18 model expects {self.EXPECTED_FEATURES} features, got {actual_features}. "
                f"Call calculate_safe_features_for_game() with use_v16=False for V18 compatibility."
            )

        # Get predictions from each base model
        base_preds = []
        for name in self.model_names:
            if name in self.base_models:
                pred = self.base_models[name].predict(X)
                base_preds.append(pred)

        if not base_preds:
            return np.zeros(len(X))

        # Stack predictions for meta-learner
        X_meta = np.column_stack(base_preds)

        # Add original features for passthrough (if meta-learner expects it)
        try:
            # Try with passthrough features
            if hasattr(X, 'values'):
                X_meta_full = np.hstack([X_meta, X.values])
            else:
                X_meta_full = np.hstack([X_meta, X])
            return self.meta_learner.predict(X_meta_full)
        except Exception:
            # Fall back to just base predictions
            return self.meta_learner.predict(X_meta)


@st.cache_resource
def load_spread_error_model():
    """Load V18 stacking ensemble model (58 features).

    V18 improvements:
    - Stacking ensemble with XGBoost, HistGradientBoosting, CatBoost
    - MAE = 9.34 (2.57 points better than Vegas baseline)
    """
    try:
        raw_model = joblib.load('cfb_v18_stacking.pkl')
        model = V18StackingWrapper(raw_model)
        logger.info("Loaded V18 stacking ensemble")
        return model, 'cfb_v18_stacking.pkl'
    except Exception as e:
        st.error(f"Failed to load V18 model: {e}")
        return None, None


@st.cache_resource
def load_totals_model():
    """Load the totals error model."""
    model_files = [
        'cfb_totals_v2.pkl',  # V2 with improved features
        'cfb_totals_model.pkl',  # Original
    ]

    for model_file in model_files:
        try:
            model = joblib.load(model_file)
            logger.info(f"Loaded totals model from {model_file}")
            return model, model_file
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning(f"Error loading {model_file}: {e}")
            continue

    logger.warning("No totals model found - totals predictions disabled")
    return None, None


@st.cache_resource
def load_quantile_model():
    """Load V17 quantile model for prediction intervals.

    Returns dict with keys 0.1, 0.25, 0.5, 0.75, 0.9 for quantile regressors.
    Used to generate 80% prediction intervals for bet selection.
    """
    model_files = [
        'cfb_spread_error_v17_quantile.pkl',
    ]

    for model_file in model_files:
        try:
            model = joblib.load(model_file)
            logger.info(f"Loaded quantile model from {model_file}")
            return model, model_file
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning(f"Error loading quantile model {model_file}: {e}")
            continue

    logger.warning("No quantile model found - prediction intervals disabled")
    return None, None


@st.cache_resource
def load_quantile_config():
    """Load quantile model config with feature list."""
    try:
        config = joblib.load('cfb_v17_quantile_config.pkl')
        return config
    except:
        return None


@st.cache_resource
def load_totals_config():
    """Load the totals model config with feature list."""
    config_files = ['cfb_totals_v2_config.pkl']
    for config_file in config_files:
        try:
            config = joblib.load(config_file)
            logger.info(f"Loaded totals config from {config_file}")
            return config
        except:
            continue
    return None


@st.cache_resource
def load_model_config():
    """Load V18 model config (58 features)."""
    try:
        config = joblib.load('cfb_v18_stacking_config.pkl')
        logger.info(f"Loaded V18 config ({len(config.get('features', []))} features)")
        return config
    except:
        return {'features': SAFE_FEATURES}


@st.cache_data(ttl=300)
def load_safe_data():
    """Load the safe data (no leakage)."""
    try:
        df = pd.read_csv('cfb_data_safe.csv')
        return df
    except FileNotFoundError:
        try:
            # Fall back to smart data but warn
            df = pd.read_csv('cfb_data_smart.csv')
            st.warning("Using cfb_data_smart.csv - run prepare_safe_features.py for best results")
            return df
        except FileNotFoundError:
            st.error("No data file found! Run prepare_safe_features.py first.")
            return None


# =============================================================================
# FEATURE CALCULATION (SAFE - NO LEAKAGE)
# =============================================================================
def calculate_safe_features_for_game(home, away, history_df, season, week, vegas_spread, line_movement, use_v16=True):
    """Calculate safe features for a single game prediction.

    Args:
        use_v16: If True, returns 68 features (V16). If False, returns 58 features (V15).
    """

    def get_team_recent_stats(team, is_home):
        """Get most recent stats for a team from historical data."""
        if is_home:
            games = history_df[(history_df['home_team'] == team) &
                              ((history_df['season'] < season) |
                               ((history_df['season'] == season) & (history_df['week'] < week)))]
            prefix = 'home'
        else:
            games = history_df[(history_df['away_team'] == team) &
                              ((history_df['season'] < season) |
                               ((history_df['season'] == season) & (history_df['week'] < week)))]
            prefix = 'away'

        if len(games) > 0:
            recent = games.sort_values(['season', 'week'], ascending=False).iloc[0]
            return {
                'pregame_elo': recent.get(f'{prefix}_pregame_elo', 1500),
                'last5_score_avg': recent.get(f'{prefix}_last5_score_avg', 28),
                'last5_defense_avg': recent.get(f'{prefix}_last5_defense_avg', 24),
                'team_hfa': recent.get(f'{prefix}_team_hfa', 2.0 if is_home else 0),
                'rest_days': recent.get(f'{prefix}_rest_days', 7),
                'streak': recent.get(f'{prefix}_streak', 0),
                'ats': recent.get(f'{prefix}_ats', 0.5),
                'elo_momentum': recent.get(f'{prefix}_elo_momentum', 0),
                'scoring_trend': recent.get(f'{prefix}_scoring_trend', 0),
                'short_rest': recent.get(f'{prefix}_short_rest', 0),
                'lookahead': recent.get(f'{prefix}_lookahead', 0),
                # V15: PPA efficiency features
                'comp_off_ppa': recent.get(f'{prefix}_comp_off_ppa', 0),
                'comp_def_ppa': recent.get(f'{prefix}_comp_def_ppa', 0),
                'comp_pass_ppa': recent.get(f'{prefix}_comp_pass_ppa', 0),
                'comp_rush_ppa': recent.get(f'{prefix}_comp_rush_ppa', 0),
                'comp_success': recent.get(f'{prefix}_comp_success', 0),
                'comp_epa': recent.get(f'{prefix}_comp_epa', 0),
                'comp_ypp': recent.get(f'{prefix}_comp_ypp', 0),
            }
        else:
            # Check if team played as opposite role
            alt_prefix = 'away' if is_home else 'home'
            alt_games = history_df[(history_df[f'{alt_prefix}_team'] == team) &
                                   ((history_df['season'] < season) |
                                    ((history_df['season'] == season) & (history_df['week'] < week)))]
            if len(alt_games) > 0:
                recent = alt_games.sort_values(['season', 'week'], ascending=False).iloc[0]
                return {
                    'pregame_elo': recent.get(f'{alt_prefix}_pregame_elo', 1500),
                    'last5_score_avg': recent.get(f'{alt_prefix}_last5_score_avg', 28),
                    'last5_defense_avg': recent.get(f'{alt_prefix}_last5_defense_avg', 24),
                    'team_hfa': 2.0 if is_home else 0,
                    'rest_days': recent.get(f'{alt_prefix}_rest_days', 7),
                    'streak': recent.get(f'{alt_prefix}_streak', 0),
                    'ats': recent.get(f'{alt_prefix}_ats', 0.5),
                    'elo_momentum': recent.get(f'{alt_prefix}_elo_momentum', 0),
                    'scoring_trend': recent.get(f'{alt_prefix}_scoring_trend', 0),
                    'short_rest': recent.get(f'{alt_prefix}_short_rest', 0),
                    'lookahead': recent.get(f'{alt_prefix}_lookahead', 0),
                    # V15: PPA efficiency features
                    'comp_off_ppa': recent.get(f'{alt_prefix}_comp_off_ppa', 0),
                    'comp_def_ppa': recent.get(f'{alt_prefix}_comp_def_ppa', 0),
                    'comp_pass_ppa': recent.get(f'{alt_prefix}_comp_pass_ppa', 0),
                    'comp_rush_ppa': recent.get(f'{alt_prefix}_comp_rush_ppa', 0),
                    'comp_success': recent.get(f'{alt_prefix}_comp_success', 0),
                    'comp_epa': recent.get(f'{alt_prefix}_comp_epa', 0),
                    'comp_ypp': recent.get(f'{alt_prefix}_comp_ypp', 0),
                }

        # Default values if no history
        return {
            'pregame_elo': 1500,
            'last5_score_avg': 28,
            'last5_defense_avg': 24,
            'team_hfa': 2.0 if is_home else 0,
            'rest_days': 7,
            'streak': 0,
            'ats': 0.5,
            'elo_momentum': 0,
            'scoring_trend': 0,
            'short_rest': 0,
            'lookahead': 0,
            # V15: PPA defaults
            'comp_off_ppa': 0,
            'comp_def_ppa': 0,
            'comp_pass_ppa': 0,
            'comp_rush_ppa': 0,
            'comp_success': 0,
            'comp_epa': 0,
            'comp_ypp': 0,
        }

    home_stats = get_team_recent_stats(home, is_home=True)
    away_stats = get_team_recent_stats(away, is_home=False)

    # Calculate derived features
    elo_diff = home_stats['pregame_elo'] - away_stats['pregame_elo']
    hfa_diff = home_stats['team_hfa'] - away_stats['team_hfa']
    rest_diff = home_stats['rest_days'] - away_stats['rest_days']
    streak_diff = home_stats['streak'] - away_stats['streak']
    ats_diff = home_stats['ats'] - away_stats['ats']
    elo_momentum_diff = home_stats['elo_momentum'] - away_stats['elo_momentum']

    # Vegas-derived features
    large_favorite = 1 if vegas_spread < -14 else 0
    large_underdog = 1 if vegas_spread > 14 else 0
    close_game = 1 if abs(vegas_spread) < 7 else 0
    elo_vs_spread = (elo_diff / 25) - vegas_spread
    rest_spread_interaction = rest_diff * abs(vegas_spread) / 10
    expected_total = home_stats['last5_score_avg'] + away_stats['last5_score_avg']

    # West coast early flag (simplified)
    west_coast_early = 0  # Would need schedule data

    # V15: Efficiency differences
    pass_efficiency_diff = home_stats['comp_pass_ppa'] - away_stats['comp_pass_ppa']

    # V15: Composite features
    matchup_efficiency = (home_stats['comp_off_ppa'] - away_stats['comp_off_ppa']) + \
                         (away_stats['comp_def_ppa'] - home_stats['comp_def_ppa'])
    home_pass_rush_balance = home_stats['comp_pass_ppa'] - home_stats['comp_rush_ppa']
    away_pass_rush_balance = away_stats['comp_pass_ppa'] - away_stats['comp_rush_ppa']
    success_rate_diff = home_stats['comp_success'] - away_stats['comp_success']
    elo_efficiency_interaction = elo_diff * (home_stats['comp_off_ppa'] - away_stats['comp_off_ppa'])
    momentum_strength = (home_stats['streak'] - away_stats['streak']) * \
                        (home_stats['last5_score_avg'] - away_stats['last5_score_avg']) / 100
    dominant_home = 1 if (elo_diff > 150 and (home_stats['comp_off_ppa'] - away_stats['comp_off_ppa']) > 0.5) else 0
    dominant_away = 1 if (elo_diff < -150 and (away_stats['comp_off_ppa'] - home_stats['comp_off_ppa']) > 0.5) else 0
    rest_favorite_interaction = rest_diff * (1 if vegas_spread < -7 else 0)
    has_line_movement = 1 if (line_movement and abs(line_movement) > 0) else 0

    # Build feature array in correct order (58 features for V15)
    features = np.array([[
        # Core Elo features (3)
        home_stats['pregame_elo'],           # home_pregame_elo
        away_stats['pregame_elo'],           # away_pregame_elo
        elo_diff,                            # elo_diff
        # Rolling stats (4)
        home_stats['last5_score_avg'],       # home_last5_score_avg
        away_stats['last5_score_avg'],       # away_last5_score_avg
        home_stats['last5_defense_avg'],     # home_last5_defense_avg
        away_stats['last5_defense_avg'],     # away_last5_defense_avg
        # Home field advantage (2)
        home_stats['team_hfa'],              # home_team_hfa
        hfa_diff,                            # hfa_diff
        # Rest and scheduling (1)
        rest_diff,                           # rest_diff
        # Vegas lines (4)
        line_movement if line_movement else 0,  # line_movement
        large_favorite,                      # large_favorite
        large_underdog,                      # large_underdog
        close_game,                          # close_game
        # Streaks (3)
        home_stats['streak'],                # home_streak
        away_stats['streak'],                # away_streak
        streak_diff,                         # streak_diff
        # ATS history (3)
        home_stats['ats'],                   # home_ats
        away_stats['ats'],                   # away_ats
        ats_diff,                            # ats_diff
        # Momentum features (5)
        home_stats['elo_momentum'],          # home_elo_momentum
        away_stats['elo_momentum'],          # away_elo_momentum
        elo_momentum_diff,                   # elo_momentum_diff
        home_stats['scoring_trend'],         # home_scoring_trend
        away_stats['scoring_trend'],         # away_scoring_trend
        # Derived features (6)
        elo_vs_spread,                       # elo_vs_spread
        rest_spread_interaction,             # rest_spread_interaction
        home_stats['short_rest'],            # home_short_rest
        away_stats['short_rest'],            # away_short_rest
        expected_total,                      # expected_total
        west_coast_early,                    # west_coast_early
        home_stats['lookahead'],             # home_lookahead
        away_stats['lookahead'],             # away_lookahead
        # V15: PPA efficiency features (15)
        home_stats['comp_off_ppa'],          # home_comp_off_ppa
        away_stats['comp_off_ppa'],          # away_comp_off_ppa
        home_stats['comp_def_ppa'],          # home_comp_def_ppa
        away_stats['comp_def_ppa'],          # away_comp_def_ppa
        home_stats['comp_pass_ppa'],         # home_comp_pass_ppa
        away_stats['comp_pass_ppa'],         # away_comp_pass_ppa
        home_stats['comp_rush_ppa'],         # home_comp_rush_ppa
        away_stats['comp_rush_ppa'],         # away_comp_rush_ppa
        home_stats['comp_success'],          # home_comp_success
        away_stats['comp_success'],          # away_comp_success
        home_stats['comp_epa'],              # home_comp_epa
        away_stats['comp_epa'],              # away_comp_epa
        home_stats['comp_ypp'],              # home_comp_ypp
        away_stats['comp_ypp'],              # away_comp_ypp
        pass_efficiency_diff,                # pass_efficiency_diff
        # V15: Composite features (10)
        matchup_efficiency,                  # matchup_efficiency
        home_pass_rush_balance,              # home_pass_rush_balance
        away_pass_rush_balance,              # away_pass_rush_balance
        success_rate_diff,                   # success_rate_diff
        elo_efficiency_interaction,          # elo_efficiency_interaction
        momentum_strength,                   # momentum_strength
        dominant_home,                       # dominant_home
        dominant_away,                       # dominant_away
        rest_favorite_interaction,           # rest_favorite_interaction
        has_line_movement,                   # has_line_movement
    ]])

    # V16: Add uncertainty features if requested (10 additional features)
    if use_v16:
        # Calculate V16 uncertainty features
        is_pickem = 1 if abs(vegas_spread) < 3 else 0
        is_mismatch = 1 if abs(elo_diff) > 300 else 0
        is_early_season = 1 if week <= 3 else 0
        is_rivalry_week = 1 if week in [11, 12, 13] else 0  # Rivalry weeks

        # Historical team errors (use defaults for live prediction)
        # In training, these come from actual historical data
        home_team_historical_error = 0.0  # Default: neutral
        away_team_historical_error = 0.0  # Default: neutral

        # Spread bucket historical error (approximate from training data)
        if abs(vegas_spread) <= 3:
            spread_bucket_error = 1.2  # Pick-em games have higher error
        elif abs(vegas_spread) <= 7:
            spread_bucket_error = 1.0  # Small spreads
        elif abs(vegas_spread) <= 14:
            spread_bucket_error = 0.9  # Medium spreads
        else:
            spread_bucket_error = 0.85  # Large spreads (more predictable)

        # Feature completeness: count non-zero features
        v15_features_flat = features.flatten()
        feature_completeness = (v15_features_flat != 0).sum() / len(v15_features_flat)

        # Rest-based flags
        is_post_bye = 1 if home_stats['rest_days'] > 10 or away_stats['rest_days'] > 10 else 0
        is_short_rest_flag = 1 if home_stats['rest_days'] < 6 or away_stats['rest_days'] < 6 else 0

        # Append V16 uncertainty features
        v16_features = np.array([[
            is_pickem,
            is_mismatch,
            is_early_season,
            is_rivalry_week,
            home_team_historical_error,
            away_team_historical_error,
            spread_bucket_error,
            feature_completeness,
            is_post_bye,
            is_short_rest_flag,
        ]])

        features = np.concatenate([features, v16_features], axis=1)

    return features


# =============================================================================
# KELLY CRITERION
# =============================================================================
def kelly_bet_size(spread_error, bankroll=1000, odds=-110,
                   interval_width=None, interval_crosses_zero=True):
    """Calculate bet size based on predicted spread error with interval adjustment.

    V18 improvement: Win probability now adjusts based on quantile interval:
    - BOOST: If interval doesn't cross zero (high certainty), increase win_prob
    - PENALIZE: If interval is very wide (low certainty), decrease win_prob
    """
    # Spread error magnitude indicates edge
    edge = abs(spread_error) / 100  # Convert to decimal

    # Convert American odds to decimal
    decimal_odds = 1 + (100 / abs(odds)) if odds < 0 else 1 + (odds / 100)
    b = decimal_odds - 1

    # Base probability from spread error
    base_prob = 0.5 + (abs(spread_error) / 50)  # Scale: 3pt error = 56%, 5pt = 60%

    # V18: Dynamic adjustment based on quantile intervals
    if interval_width is not None:
        # BOOST if interval doesn't cross zero (high certainty)
        if not interval_crosses_zero:
            # Tighter interval = more confidence boost (up to +5%)
            certainty_boost = max(0, 0.05 - (interval_width / 300))
            base_prob += certainty_boost

        # PENALIZE if interval is very wide (low certainty)
        if interval_width > 20:
            uncertainty_penalty = min(0.05, (interval_width - 20) / 200)
            base_prob -= uncertainty_penalty

    win_prob = min(0.75, max(0.52, base_prob))

    q = 1 - win_prob
    kelly_fraction = max(0, (b * win_prob - q) / b)

    # Apply fractional Kelly and cap
    bet_size = min(bankroll * kelly_fraction * KELLY_FRACTION, bankroll * 0.05)

    return {
        'bet_size': round(bet_size, 2),
        'kelly_fraction': kelly_fraction,
        'win_prob': win_prob,
        'edge': edge,
        'interval_adjusted': interval_width is not None
    }


def get_confidence_tier(spread_error):
    """Get confidence tier based on spread error magnitude."""
    error_mag = abs(spread_error)
    # V19: Adjusted thresholds - be more conservative
    if error_mag >= 6.0:
        return 'HIGH', 'confidence-high', '🔥'
    elif error_mag >= 4.5:
        return 'MEDIUM-HIGH', 'confidence-medium-high', '✅'
    elif error_mag >= 3.0:
        return 'MEDIUM', 'confidence-medium', '⚠️'
    elif error_mag >= 1.5:
        return 'LOW', 'confidence-low', '⚡'
    else:
        return 'VERY LOW', 'confidence-very-low', '❄️'


def classify_game_type(row, week=None):
    """
    V19: Classify game type to identify PASS recommendations.

    Returns:
        dict with game type flags and recommendation
    """
    elo_diff = abs(row.get('elo_diff', 0) if pd.notna(row.get('elo_diff')) else 0)
    spread = abs(row.get('vegas_spread', 0) if pd.notna(row.get('vegas_spread')) else 0)
    home_elo = row.get('home_pregame_elo', 1500) if pd.notna(row.get('home_pregame_elo')) else 1500
    away_elo = row.get('away_pregame_elo', 1500) if pd.notna(row.get('away_pregame_elo')) else 1500

    game_type = {
        'is_pick_em': spread < 3,           # Pick-em games are hard to predict
        'is_avg_vs_avg': elo_diff < 100,    # Average vs average - 55% accuracy only
        'is_early_season': week is not None and week <= 2,  # Data staleness
        'is_large_mismatch': elo_diff > 300,  # Blowout potential
        'is_elite_matchup': home_elo > 1700 and away_elo > 1700,  # Top 25 vs Top 25
    }

    # Determine if this game should be PASSED
    pass_reasons = []
    if SKIP_PICK_EM_GAMES and game_type['is_pick_em']:
        pass_reasons.append('pick-em')
    if SKIP_AVG_VS_AVG_GAMES and game_type['is_avg_vs_avg']:
        pass_reasons.append('avg-vs-avg')
    if SKIP_EARLY_SEASON_GAMES and game_type['is_early_season']:
        pass_reasons.append('early-season')

    game_type['should_pass'] = len(pass_reasons) > 0
    game_type['pass_reasons'] = pass_reasons

    return game_type


def get_bet_recommendation(spread_error, game_type_info, confidence_tier):
    """
    V19: Get bet recommendation with PASS option.

    Returns:
        recommendation: 'BET', 'LEAN', or 'PASS'
        reason: explanation for the recommendation
    """
    error_mag = abs(spread_error)

    # PASS if game type is bad
    if game_type_info.get('should_pass', False):
        reasons = game_type_info.get('pass_reasons', [])
        return 'PASS', f"Skip: {', '.join(reasons)}"

    # BET only for HIGH/MEDIUM-HIGH confidence
    if confidence_tier in ['HIGH', 'MEDIUM-HIGH']:
        if error_mag >= SPREAD_ERROR_THRESHOLD:
            return 'BET', 'Strong edge'
        else:
            return 'LEAN', 'Moderate edge'

    # LEAN for MEDIUM confidence with decent edge
    if confidence_tier == 'MEDIUM' and error_mag >= 3.0:
        return 'LEAN', 'Marginal edge'

    # PASS for LOW/VERY LOW
    return 'PASS', 'Low confidence'


# =============================================================================
# FETCH DATA
# =============================================================================
@st.cache_data(ttl=300)
def fetch_schedule(season, week, season_type='regular'):
    """Fetch game schedule."""
    try:
        # Postseason (Bowl Games) doesn't use week parameter
        if season_type == 'postseason':
            url = f"{CFBD_BASE_URL}/games?year={season}&seasonType={season_type}"
        else:
            url = f"{CFBD_BASE_URL}/games?year={season}&week={week}&seasonType={season_type}"
        resp = requests.get(url, headers=get_api_headers())
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        st.error(f"Error fetching schedule: {e}")
        return []


@st.cache_data(ttl=60)  # Reduced from 300s for more real-time lines
def fetch_lines(season, week, season_type='regular'):
    """Fetch betting lines from CFBD API."""
    try:
        # Postseason (Bowl Games) doesn't use week parameter
        if season_type == 'postseason':
            url = f"{CFBD_BASE_URL}/lines?year={season}&seasonType={season_type}"
        else:
            url = f"{CFBD_BASE_URL}/lines?year={season}&week={week}&seasonType={season_type}"
        resp = requests.get(url, headers=get_api_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"Fetched lines for {len(data)} games")
            return data
        logger.warning(f"Lines API returned status {resp.status_code}")
        return []
    except requests.Timeout:
        logger.error("Lines API request timed out")
        return []
    except Exception as e:
        logger.error(f"Error fetching lines: {e}")
        return []


# Book preference order (consensus first, then major books)
PREFERRED_BOOKS = ['consensus', 'Caesars', 'DraftKings', 'FanDuel', 'BetMGM', 'Bovada']


def build_lines_dict(betting_lines):
    """Build lines dictionary from API response using preferred sportsbook."""
    lines = {}
    for line in betting_lines:
        line_books = line.get('lines', [])
        if not line_books:
            continue

        # Find best book based on preference order
        selected_book = None
        for preferred in PREFERRED_BOOKS:
            for book in line_books:
                provider = book.get('provider', '').lower()
                if preferred.lower() in provider:
                    if book.get('spread') is not None:
                        selected_book = book
                        break
            if selected_book:
                break

        # Fallback to first book with valid spread if no preferred found
        if not selected_book:
            for book in line_books:
                if book.get('spread') is not None:
                    selected_book = book
                    break

        if selected_book:
            spread = float(selected_book['spread'])
            opening = float(selected_book.get('spreadOpen', spread)) if selected_book.get('spreadOpen') else spread
            over_under = selected_book.get('overUnder')
            ou_opening = selected_book.get('overUnderOpen')
            provider = selected_book.get('provider', 'Unknown')

            lines[line['homeTeam']] = {
                'spread_current': spread,
                'spread_opening': opening,
                'line_movement': spread - opening,
                'over_under': float(over_under) if over_under else None,
                'ou_opening': float(ou_opening) if ou_opening else None,
                'provider': provider,
            }
    return lines


# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================
def generate_spread_error_predictions(games, lines_dict, model, history_df, season, week, bankroll, quantile_model=None):
    """Generate predictions using spread error approach with optional prediction intervals.

    Args:
        quantile_model: Optional V17 quantile model for 80% prediction intervals
    """
    predictions = []

    # Clear any previous prediction errors
    if 'prediction_error' in st.session_state:
        del st.session_state['prediction_error']

    for game in games:
        try:
            home, away = game['homeTeam'], game['awayTeam']
            if not home or not away or home not in lines_dict:
                continue

            vegas_spread = lines_dict[home]['spread_current']
            line_movement = lines_dict[home]['line_movement']

            # Calculate safe features - V18 uses 58 features
            features = calculate_safe_features_for_game(
                home, away, history_df, season, week, vegas_spread, line_movement,
                use_v16=False
            )

            # Predict spread error
            pred_spread_error = model.predict(features)[0]

            # Get prediction intervals from quantile model if available
            lower_bound = None
            upper_bound = None
            interval_width = None
            interval_crosses_zero = True  # Default to True (uncertain)

            if quantile_model is not None:
                try:
                    # Get 10th and 90th percentile for 80% interval
                    q10 = quantile_model[0.1].predict(features)[0]
                    q90 = quantile_model[0.9].predict(features)[0]
                    lower_bound = q10
                    upper_bound = q90
                    interval_width = q90 - q10

                    # Check if interval crosses zero (uncertain prediction)
                    interval_crosses_zero = (q10 < 0) and (q90 > 0)
                except Exception as e:
                    logger.warning(f"Error getting quantile prediction: {e}")

            # Determine signal - always assign BUY or FADE (no more PASS filtering)
            if pred_spread_error > 0:
                # Model predicts home will do better than Vegas expects
                signal = 'BUY'
                team_to_bet = home
                opponent = away
                spread_to_bet = vegas_spread
            else:
                # Model predicts away will do better than Vegas expects
                signal = 'FADE'
                team_to_bet = away
                opponent = home
                spread_to_bet = -vegas_spread

            # Calculate bet size and win probability (V18: now uses interval data)
            kelly_result = kelly_bet_size(
                pred_spread_error,
                bankroll,
                interval_width=interval_width,
                interval_crosses_zero=interval_crosses_zero
            )
            bet_size = kelly_result['bet_size']
            win_prob = kelly_result['win_prob']

            # Get confidence tier (enhanced with interval info)
            confidence_tier, confidence_class, confidence_emoji = get_confidence_tier(pred_spread_error)

            # Boost confidence if interval doesn't cross zero
            if not interval_crosses_zero and interval_width is not None:
                if interval_width < 15:  # Tight interval
                    if confidence_tier == 'MEDIUM':
                        confidence_tier = 'MEDIUM-HIGH'
                        confidence_class = 'confidence-medium-high'
                        confidence_emoji = '✅'
                    elif confidence_tier == 'MEDIUM-HIGH':
                        confidence_tier = 'HIGH'
                        confidence_class = 'confidence-high'
                        confidence_emoji = '🔥'

            predictions.append({
                'Home': home,
                'Away': away,
                'Game': f"{away} @ {home}",
                'Signal': signal,
                'team_to_bet': team_to_bet,
                'opponent': opponent,
                'spread_to_bet': spread_to_bet,
                'vegas_spread': vegas_spread,
                'spread_error': pred_spread_error,
                'win_prob': win_prob,
                'bet_size': bet_size,
                'line_movement': line_movement,
                'confidence_tier': confidence_tier,
                'confidence_class': confidence_class,
                'confidence_emoji': confidence_emoji,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': interval_width,
                'interval_crosses_zero': interval_crosses_zero,
                'start_date': game.get('start_date'),  # For game time badge
                'completed': game.get('completed', False),  # Game status
            })
        except Exception as e:
            logger.error(f"Error predicting {game.get('awayTeam', '?')} @ {game.get('homeTeam', '?')}: {e}")
            # Track first error to surface to user
            if 'prediction_error' not in st.session_state:
                st.session_state['prediction_error'] = str(e)

    return pd.DataFrame(predictions)


# =============================================================================
# TOTALS PREDICTIONS
# =============================================================================
def get_totals_confidence_tier(totals_error):
    """Get confidence tier for totals prediction."""
    error_mag = abs(totals_error)
    if error_mag >= 5.0:
        return 'HIGH', 'confidence-high', '🔥'
    elif error_mag >= 3.5:
        return 'MEDIUM-HIGH', 'confidence-medium-high', '✅'
    elif error_mag >= 2.0:
        return 'MEDIUM', 'confidence-medium', '⚠️'
    elif error_mag >= 1.0:
        return 'LOW', 'confidence-low', '⚡'
    else:
        return 'VERY LOW', 'confidence-very-low', '❄️'


def calculate_totals_features(home, away, history_df, vegas_ou, vegas_spread):
    """Calculate features for totals prediction."""
    # Get recent team stats
    def get_team_stats(team):
        # Get most recent data for this team
        home_games = history_df[history_df['home_team'] == team].tail(5)
        away_games = history_df[history_df['away_team'] == team].tail(5)

        if len(home_games) > 0:
            return {
                'last5_score_avg': home_games['home_last5_score_avg'].iloc[-1] if 'home_last5_score_avg' in home_games else 28,
                'last5_defense_avg': home_games['home_last5_defense_avg'].iloc[-1] if 'home_last5_defense_avg' in home_games else 24,
                'comp_off_ppa': home_games['home_comp_off_ppa'].iloc[-1] if 'home_comp_off_ppa' in home_games else 0,
                'comp_def_ppa': home_games['home_comp_def_ppa'].iloc[-1] if 'home_comp_def_ppa' in home_games else 0,
                'comp_pass_ppa': home_games['home_comp_pass_ppa'].iloc[-1] if 'home_comp_pass_ppa' in home_games else 0,
                'comp_rush_ppa': home_games['home_comp_rush_ppa'].iloc[-1] if 'home_comp_rush_ppa' in home_games else 0,
                'comp_success': home_games['home_comp_success'].iloc[-1] if 'home_comp_success' in home_games else 0.4,
                'pregame_elo': home_games['home_pregame_elo'].iloc[-1] if 'home_pregame_elo' in home_games else 1500,
            }
        elif len(away_games) > 0:
            return {
                'last5_score_avg': away_games['away_last5_score_avg'].iloc[-1] if 'away_last5_score_avg' in away_games else 28,
                'last5_defense_avg': away_games['away_last5_defense_avg'].iloc[-1] if 'away_last5_defense_avg' in away_games else 24,
                'comp_off_ppa': away_games['away_comp_off_ppa'].iloc[-1] if 'away_comp_off_ppa' in away_games else 0,
                'comp_def_ppa': away_games['away_comp_def_ppa'].iloc[-1] if 'away_comp_def_ppa' in away_games else 0,
                'comp_pass_ppa': away_games['away_comp_pass_ppa'].iloc[-1] if 'away_comp_pass_ppa' in away_games else 0,
                'comp_rush_ppa': away_games['away_comp_rush_ppa'].iloc[-1] if 'away_comp_rush_ppa' in away_games else 0,
                'comp_success': away_games['away_comp_success'].iloc[-1] if 'away_comp_success' in away_games else 0.4,
                'pregame_elo': away_games['away_pregame_elo'].iloc[-1] if 'away_pregame_elo' in away_games else 1500,
            }
        else:
            return {
                'last5_score_avg': 28, 'last5_defense_avg': 24,
                'comp_off_ppa': 0, 'comp_def_ppa': 0,
                'comp_pass_ppa': 0, 'comp_rush_ppa': 0,
                'comp_success': 0.4, 'pregame_elo': 1500,
            }

    home_stats = get_team_stats(home)
    away_stats = get_team_stats(away)

    # Calculate derived features matching training
    expected_total = home_stats['last5_score_avg'] + away_stats['last5_score_avg']
    off_ppa_sum = home_stats['comp_off_ppa'] + away_stats['comp_off_ppa']
    def_ppa_sum = home_stats['comp_def_ppa'] + away_stats['comp_def_ppa']
    net_efficiency = off_ppa_sum - def_ppa_sum
    pass_heavy_index = home_stats['comp_pass_ppa'] + away_stats['comp_pass_ppa']
    success_sum = home_stats['comp_success'] + away_stats['comp_success']
    spread_magnitude = abs(vegas_spread) if vegas_spread else 0
    is_blowout_expected = 1 if spread_magnitude > 17 else 0
    is_close_expected = 1 if spread_magnitude < 7 else 0
    elo_diff = home_stats['pregame_elo'] - away_stats['pregame_elo']

    # Build feature array matching training order
    features = np.array([[
        home_stats['last5_score_avg'],
        away_stats['last5_score_avg'],
        home_stats['last5_defense_avg'],
        away_stats['last5_defense_avg'],
        home_stats['comp_off_ppa'],
        away_stats['comp_off_ppa'],
        home_stats['comp_def_ppa'],
        away_stats['comp_def_ppa'],
        home_stats['comp_pass_ppa'],
        away_stats['comp_pass_ppa'],
        home_stats['comp_rush_ppa'],
        away_stats['comp_rush_ppa'],
        home_stats['comp_success'],
        away_stats['comp_success'],
        vegas_ou,
        vegas_spread if vegas_spread else 0,
        home_stats['pregame_elo'],
        away_stats['pregame_elo'],
        elo_diff,
        expected_total,
        off_ppa_sum,
        def_ppa_sum,
        net_efficiency,
        pass_heavy_index,
        success_sum,
        spread_magnitude,
        is_blowout_expected,
        is_close_expected,
    ]])

    return features


def generate_totals_predictions(games, lines_dict, model, history_df, bankroll):
    """Generate over/under predictions."""
    predictions = []

    for game in games:
        try:
            home, away = game['homeTeam'], game['awayTeam']
            if not home or home not in lines_dict:
                continue

            line_info = lines_dict[home]
            vegas_ou = line_info.get('over_under')
            vegas_spread = line_info.get('spread_current')

            if not vegas_ou:
                continue

            # Calculate features
            features = calculate_totals_features(home, away, history_df, vegas_ou, vegas_spread)

            # Predict totals error
            totals_error = model.predict(features)[0]

            # Positive error = model predicts OVER
            # Negative error = model predicts UNDER
            if totals_error > 0:
                signal = 'OVER'
            else:
                signal = 'UNDER'

            predicted_total = vegas_ou + totals_error

            # Get confidence tier
            confidence_tier, confidence_class, confidence_emoji = get_totals_confidence_tier(totals_error)

            # Calculate win probability based on error magnitude
            base_prob = 0.5 + (abs(totals_error) / 50)
            win_prob = min(0.75, max(0.52, base_prob))

            # Bet size similar to spread
            edge = abs(totals_error) / 100
            kelly_fraction = max(0, (win_prob - 0.5) * 2 * KELLY_FRACTION)
            bet_size = min(bankroll * kelly_fraction, bankroll * 0.05)

            predictions.append({
                'Home': home,
                'Away': away,
                'Game': f"{away} @ {home}",
                'Signal': signal,
                'vegas_ou': vegas_ou,
                'predicted_total': predicted_total,
                'totals_error': totals_error,
                'win_prob': win_prob,
                'bet_size': bet_size,
                'confidence_tier': confidence_tier,
                'confidence_class': confidence_class,
                'confidence_emoji': confidence_emoji,
            })
        except Exception as e:
            logger.error(f"Error predicting totals for {game.get('awayTeam', '?')} @ {game.get('homeTeam', '?')}: {e}")

    return pd.DataFrame(predictions)


def format_game_time_badge(start_date_str):
    """Format game time as a user-friendly badge.

    Returns tuple of (badge_text, badge_class) for CSS styling.
    """
    if not start_date_str:
        return None, None

    try:
        # Parse ISO format date from CFBD API
        # Format: "2024-12-21T18:00:00.000Z"
        game_time = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        now = datetime.now(game_time.tzinfo) if game_time.tzinfo else datetime.now()

        # Calculate time until kickoff
        time_diff = game_time - now
        total_seconds = time_diff.total_seconds()

        if total_seconds < 0:
            # Game has started or is over
            if total_seconds > -4 * 3600:  # Within 4 hours - probably still playing
                return "LIVE", "game-status-live"
            else:
                return None, None  # Game is over, no badge needed
        elif total_seconds < 3600:
            # Less than 1 hour
            mins = int(total_seconds / 60)
            return f"Kickoff in {mins}m", "game-status-soon"
        elif total_seconds < 86400:
            # Less than 24 hours
            hours = int(total_seconds / 3600)
            mins = int((total_seconds % 3600) / 60)
            if mins > 0:
                return f"Kickoff in {hours}h {mins}m", "game-status-today"
            return f"Kickoff in {hours}h", "game-status-today"
        else:
            # Show date and time
            # Convert to local display format
            local_time = game_time.strftime("%b %d, %I:%M %p")
            return local_time, "game-status-future"
    except Exception:
        return None, None


def render_totals_card(bet, is_hero=False):
    """Render a styled totals bet card."""
    signal_class = "over" if bet['Signal'] == 'OVER' else "under"
    signal_emoji = "⬆️" if bet['Signal'] == 'OVER' else "⬇️"

    # Get confidence tier
    conf_tier = bet.get('confidence_tier', 'MEDIUM')
    conf_class = bet.get('confidence_class', 'confidence-medium')
    conf_emoji = bet.get('confidence_emoji', '⚠️')
    conf_badge = f'<span class="confidence-badge {conf_class}">{conf_emoji} {conf_tier}</span>'

    instruction_class = "bet-instruction-hero" if is_hero else "bet-instruction"
    amount_class = "bet-amount-hero" if is_hero else "bet-amount"

    html = f"""
    <div class="bet-card {signal_class}">
        {conf_badge}
        <div class="{instruction_class}">{signal_emoji} {bet['Signal']} {bet['vegas_ou']:.1f}</div>
        <div class="opponent">{bet['Game']}</div>
        <div class="{amount_class}">${bet['bet_size']:.0f}</div>
        <div class="win-prob">{bet['win_prob']*100:.0f}% Est. Win Probability</div>
        <div class="spread-error">Model: {bet['predicted_total']:.1f} pts | Edge: {abs(bet['totals_error']):.1f} pts</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# BET CARD COMPONENT
# =============================================================================
def render_bet_card(bet, is_hero=False):
    """Render a styled bet card with optional prediction interval and game status."""
    signal_class = "buy" if bet['Signal'] == 'BUY' else "fade"

    # Get confidence tier from bet data (always show badge)
    conf_tier = bet.get('confidence_tier', 'MEDIUM')
    conf_class = bet.get('confidence_class', 'confidence-medium')
    conf_emoji = bet.get('confidence_emoji', '⚠️')
    conf_badge = f'<span class="confidence-badge {conf_class}">{conf_emoji} {conf_tier}</span>'

    # Get game status badge
    game_time_badge = ""
    start_date = bet.get('start_date')
    if start_date:
        badge_text, badge_class = format_game_time_badge(start_date)
        if badge_text:
            game_time_badge = f'<span class="game-status-badge {badge_class}">{badge_text}</span>'

    # Format spread
    spread_str = f"{bet['spread_to_bet']:+.1f}" if bet['spread_to_bet'] != 0 else "PK"

    instruction_class = "bet-instruction-hero" if is_hero else "bet-instruction"
    amount_class = "bet-amount-hero" if is_hero else "bet-amount"

    # Build interval display if available
    interval_html = ""
    lower = bet.get('lower_bound')
    upper = bet.get('upper_bound')
    crosses_zero = bet.get('interval_crosses_zero', True)

    if lower is not None and upper is not None:
        # Color based on whether interval crosses zero
        if not crosses_zero:
            interval_color = "#28a745"  # Green - confident
            interval_icon = "✓"
        else:
            interval_color = "#ffc107"  # Amber - uncertain
            interval_icon = "~"
        interval_html = f'<div class="interval" style="color: {interval_color}; font-size: 0.85rem;">{interval_icon} 80% range: {lower:+.1f} to {upper:+.1f}</div>'

    html = f"""
    <div class="bet-card {signal_class}">
        {conf_badge}
        <div class="{instruction_class}">BET: {bet['team_to_bet']} {spread_str}</div>
        <div class="opponent">vs {bet['opponent']}{game_time_badge}</div>
        <div class="{amount_class}">${bet['bet_size']:.0f}</div>
        <div class="win-prob">{bet['win_prob']*100:.0f}% Est. Win Probability</div>
        <div class="spread-error">Predicted edge: {abs(bet['spread_error']):.1f} pts vs Vegas</div>
        {interval_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_pass_item(bet):
    """Render a muted pass game item."""
    reason = f"Edge too small ({abs(bet['spread_error']):.1f} pts < {SPREAD_ERROR_THRESHOLD})"
    html = f"""
    <div class="pass-item">
        {bet['Game']} - {reason}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# DATA FRESHNESS
# =============================================================================
def get_data_freshness():
    """Get the last modified time of the data file."""
    data_files = ['cfb_data_safe.csv', 'cfb_data_smart.csv']
    for f in data_files:
        if os.path.exists(f):
            mod_time = os.path.getmtime(f)
            mod_date = datetime.fromtimestamp(mod_time)
            days_old = (datetime.now() - mod_date).days
            return mod_date, days_old, f
    return None, None, None


def refresh_data(full_refresh=False):
    """Run data refresh pipeline.

    Args:
        full_refresh: If True, also runs refresh_all_data.py and fetch_betting_lines.py to fetch from API.
                     If False (default), only runs prepare_safe_features.py.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if full_refresh:
        # Step 1: Fetch fresh data from API
        refresh_script = os.path.join(base_dir, 'refresh_all_data.py')
        result1 = subprocess.run(
            ['python3', refresh_script],
            capture_output=True,
            text=True,
            cwd=base_dir,
            timeout=300
        )
        if result1.returncode != 0:
            return False, result1.stdout, result1.stderr

        # Step 2: Fetch betting lines
        lines_script = os.path.join(base_dir, 'fetch_betting_lines.py')
        result2 = subprocess.run(
            ['python3', lines_script],
            capture_output=True,
            text=True,
            cwd=base_dir,
            timeout=180
        )
        if result2.returncode != 0:
            return False, result2.stdout, result2.stderr

    # Step 3: Prepare safe features
    prepare_script = os.path.join(base_dir, 'prepare_safe_features.py')
    result3 = subprocess.run(
        ['python3', prepare_script],
        capture_output=True,
        text=True,
        cwd=base_dir
    )
    return result3.returncode == 0, result3.stdout, result3.stderr


def get_scheduler_status():
    """Get status from the scheduler if running."""
    status_file = Path(__file__).parent / '.refresh_status.json'
    if status_file.exists():
        try:
            with open(status_file) as f:
                status = json.load(f)
            last_refresh = datetime.fromisoformat(status['last_refresh'])
            next_refresh = datetime.fromisoformat(status['next_refresh'])
            return {
                'active': True,
                'last_refresh': last_refresh,
                'next_refresh': next_refresh,
                'success': status.get('success', True),
                'message': status.get('message', ''),
                'is_gameday': status.get('is_gameday', False)
            }
        except Exception:
            pass
    return {'active': False}


# =============================================================================
# MAIN APP
# =============================================================================

# =============================================================================
# AUTO-DETECT SEASON, TYPE, AND WEEK
# =============================================================================
def get_auto_detected_settings():
    """Auto-detect season, season_type, and week based on current date."""
    today = datetime.now()
    current_year = today.year
    current_month = today.month

    # Determine season year
    # CFB season runs Aug-Jan, so Jan games belong to previous year's season
    if current_month == 1:
        season = current_year - 1
    else:
        season = current_year

    # Determine season type
    # Bowl season: December 14 - January 31
    # Regular season: Late August - Early December
    if current_month == 12 and today.day >= 14:
        season_type = "postseason"
        week = 1  # Bowl games are week 1 of postseason
    elif current_month == 1:
        season_type = "postseason"
        week = 1
    else:
        season_type = "regular"
        # Estimate week based on date (CFB typically starts last week of August)
        # Week 0: ~Aug 24, Week 1: ~Aug 31, etc.
        if current_month < 8:
            week = 1  # Off-season, default to week 1
        else:
            # Calculate approximate week
            season_start = datetime(current_year, 8, 24)  # Approximate Week 0
            days_since_start = (today - season_start).days
            week = max(0, min(15, days_since_start // 7))
            if week == 0:
                week = 1  # Default to week 1 if before season

    return season, season_type, week

# Get auto-detected defaults
auto_season, auto_season_type, auto_week = get_auto_detected_settings()

# Header - Simplified with auto-detection
col1, col2, col3 = st.columns([4, 3, 2])
with col1:
    st.title("Sharp Picks CFB")
with col2:
    # Show current selection as a nice display
    if auto_season_type == "postseason":
        period_display = f"🏆 Bowl Games {auto_season}"
    else:
        period_display = f"🏈 Week {auto_week}, {auto_season}"
    st.markdown(f"<div style='padding-top: 12px; font-size: 1.1rem; color: #B4C6DC;'>{period_display}</div>", unsafe_allow_html=True)
with col3:
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 1000
    bankroll = st.number_input("💰 Bankroll ($)", min_value=100, max_value=100000,
                               value=st.session_state.bankroll, step=100,
                               format="%d")
    st.session_state.bankroll = bankroll

# Use auto-detected values
season = auto_season
season_type = auto_season_type
week = auto_week

# Advanced settings expander for manual override
with st.expander("⚙️ Change Season/Week", expanded=False):
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    with adv_col1:
        season = st.selectbox("Season", [2025, 2024, 2023, 2022],
                             index=[2025, 2024, 2023, 2022].index(auto_season),
                             key="manual_season")
    with adv_col2:
        season_type = st.selectbox("Type", ["regular", "postseason"],
                                   index=0 if auto_season_type == "regular" else 1,
                                   format_func=lambda x: "Bowl Games" if x == "postseason" else "Regular Season",
                                   key="manual_type")
    with adv_col3:
        week = st.number_input("Week", min_value=0, max_value=20, value=auto_week, key="manual_week")

# Model info
model, model_file = load_spread_error_model()
totals_model, totals_model_file = load_totals_model()
quantile_model, quantile_model_file = load_quantile_model()

if model:
    models_info = f"Spread: {model_file}"
    if totals_model:
        models_info += f" | Totals: {totals_model_file}"
    if quantile_model:
        models_info += " | Intervals: V17"
    st.markdown(f"""
    <div class="model-info">
        <span class="model-info-text">
            {models_info} | Spread Validated: 59.9% win rate | Totals: 74.7% win rate
        </span>
    </div>
    """, unsafe_allow_html=True)

# Buttons row
col_btn1, col_btn2, col_btn3, col_btn4, col_spacer = st.columns([1, 1, 1, 1, 4])
with col_btn1:
    if st.button("🔄 Lines", type="secondary", use_container_width=True):
        # Clear lines cache for immediate refresh
        fetch_lines.clear()
        st.rerun()
with col_btn2:
    if st.button("Refresh Cache", type="secondary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_btn3:
    if st.button("Quick Refresh", type="secondary", use_container_width=True):
        with st.spinner("Preparing safe features..."):
            success, stdout, stderr = refresh_data(full_refresh=False)
            if success:
                st.cache_data.clear()
                st.success("Features refreshed!")
                st.rerun()
            else:
                st.error(f"Error: {stderr}")
with col_btn4:
    if st.button("Full Refresh", type="primary", use_container_width=True):
        with st.spinner("Fetching from API and preparing features (this may take a minute)..."):
            try:
                success, stdout, stderr = refresh_data(full_refresh=True)
                if success:
                    st.cache_data.clear()
                    st.success("Data fully refreshed from API!")
                    st.rerun()
                else:
                    st.error(f"Error: {stderr}")
            except subprocess.TimeoutExpired:
                st.error("Refresh timed out. Try again or check your connection.")

# Data freshness with scheduler status
mod_date, days_old, data_file = get_data_freshness()
scheduler_status = get_scheduler_status()
type_label = "Bowl Games" if season_type == "postseason" else f"Week {week}"

if mod_date:
    freshness_text = f"Data: {mod_date.strftime('%b %d, %Y %H:%M')}"

    # Add scheduler status if active
    if scheduler_status['active']:
        next_refresh = scheduler_status['next_refresh']
        time_until = next_refresh - datetime.now()
        if time_until.total_seconds() > 0:
            hours = int(time_until.total_seconds() // 3600)
            mins = int((time_until.total_seconds() % 3600) // 60)
            freshness_text += f" | Auto-refresh in {hours}h {mins}m"
            if scheduler_status['is_gameday']:
                freshness_text += " (Game Day)"

    if days_old > 7:
        st.warning(f"Data is {days_old} days old - Consider refreshing!")
    else:
        st.caption(f"{type_label}, {season} | Bankroll: ${bankroll:,} | {freshness_text}")
else:
    st.caption(f"{type_label}, {season} | Bankroll: ${bankroll:,}")

# Load data
if model is None:
    st.stop()

history_df = load_safe_data()
if history_df is None:
    st.stop()

# Fetch and generate predictions
with st.spinner("Loading picks..."):
    games = fetch_schedule(season, week, season_type)
    betting_lines = fetch_lines(season, week, season_type)

if not games:
    if season_type == "postseason":
        st.warning(f"No bowl games found for {season}. Try selecting 'Regular' for regular season games.")
    else:
        st.warning(f"No games found for Week {week}, {season}")
    st.stop()

# =============================================================================
# FILTER GAMES BY COMPLETION STATUS
# =============================================================================
# CFBD API returns 'completed' field - separate upcoming from finished games
upcoming_games = [g for g in games if not g.get('completed', False)]
completed_games = [g for g in games if g.get('completed', False)]

# Show game status counts
total_games = len(games)
upcoming_count = len(upcoming_games)
completed_count = len(completed_games)

lines_dict = build_lines_dict(betting_lines)

# Show lines status with game counts
if betting_lines:
    status_parts = [f"📊 {len(lines_dict)} games with lines"]
    if upcoming_count > 0:
        status_parts.append(f"🎯 {upcoming_count} upcoming")
    if completed_count > 0:
        status_parts.append(f"✅ {completed_count} completed")
    st.caption(" | ".join(status_parts))
else:
    st.warning("⚠️ No betting lines available from API - check connection")

# Generate predictions for UPCOMING games only (for picks tabs)
with st.spinner("Generating spread error predictions..."):
    df_predictions = generate_spread_error_predictions(
        upcoming_games if upcoming_games else games,  # Fall back to all if no upcoming
        lines_dict, model, history_df, season, week, bankroll,
        quantile_model=quantile_model
    )

# Generate predictions for COMPLETED games (for results tab)
df_completed_predictions = pd.DataFrame()
if completed_games:
    with st.spinner("Loading historical results..."):
        df_completed_predictions = generate_spread_error_predictions(
            completed_games, lines_dict, model, history_df, season, week, bankroll,
            quantile_model=quantile_model
        )

# Generate totals predictions if model is available (UPCOMING only)
df_totals = pd.DataFrame()
if totals_model:
    with st.spinner("Generating totals predictions..."):
        df_totals = generate_totals_predictions(
            upcoming_games if upcoming_games else games,
            lines_dict, totals_model, history_df, bankroll
        )

if df_predictions.empty:
    st.warning("No predictions generated.")
    # Show error details if available
    if 'prediction_error' in st.session_state:
        st.error(f"Error: {st.session_state['prediction_error']}")
        del st.session_state['prediction_error']
    st.stop()

# =============================================================================
# V17 IMPROVEMENT: Filter LOW confidence bets (historically -6.8% ROI)
# =============================================================================
# Confidence tier order from lowest to highest
CONFIDENCE_ORDER = ['VERY LOW', 'LOW', 'MEDIUM', 'MEDIUM-HIGH', 'HIGH']
MIN_CONFIDENCE_TIER = 'MEDIUM'  # Filter out LOW and VERY LOW

min_conf_idx = CONFIDENCE_ORDER.index(MIN_CONFIDENCE_TIER)

# Filter predictions to exclude unprofitable tiers
df_predictions_filtered = df_predictions[
    df_predictions['confidence_tier'].apply(
        lambda x: CONFIDENCE_ORDER.index(x) >= min_conf_idx if x in CONFIDENCE_ORDER else True
    )
]

# Show filter stats
filtered_count = len(df_predictions) - len(df_predictions_filtered)
if filtered_count > 0:
    st.caption(f"🎯 Filtered {filtered_count} low-confidence games (showing MEDIUM+ only for better ROI)")

# Use filtered predictions
df_predictions = df_predictions_filtered

# All games now have BUY or FADE signals - sort by confidence (spread error magnitude)
all_bets = df_predictions.sort_values('spread_error', key=lambda x: abs(x), ascending=False)

# Sort totals by confidence (totals error magnitude)
if not df_totals.empty:
    all_totals = df_totals.sort_values('totals_error', key=lambda x: abs(x), ascending=False)
else:
    all_totals = pd.DataFrame()

# =============================================================================
# TABS FOR SPREAD, TOTALS, AND RESULTS
# =============================================================================
spread_tab, totals_tab, results_tab = st.tabs(["🏈 Spread Picks", "📊 Totals (O/U)", "📜 Results"])

with spread_tab:
    hero_bets = all_bets.head(2)
    more_bets = all_bets.iloc[2:]

    # =============================================================================
    # HERO SECTION
    # =============================================================================
    st.markdown('<div class="hero-title">TOP PICKS (Spread Error Model)</div>', unsafe_allow_html=True)

    # Confidence explanation
    st.markdown("""
<div style="background: #1E3A5F; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; border-left: 4px solid #F59E0B;">
    <span style="color: #FCD34D; font-size: 13px;">
        📊 <b>Confidence = How wrong we think Vegas is</b><br>
        🔥 HIGH (5+ pts) • ✅ MEDIUM-HIGH (3.5-5 pts) • ⚠️ MEDIUM (2-3.5 pts) • ⚡ LOW (1-2 pts) • ❄️ VERY LOW (&lt;1 pt)
    </span>
</div>
""", unsafe_allow_html=True)

    if len(hero_bets) == 1:
        render_bet_card(hero_bets.iloc[0], is_hero=True)
    elif len(hero_bets) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            render_bet_card(hero_bets.iloc[0], is_hero=True)
        with col2:
            render_bet_card(hero_bets.iloc[1], is_hero=True)

    # =============================================================================
    # MORE PICKS
    # =============================================================================
    if len(more_bets) > 0:
        st.markdown("---")
        st.subheader(f"More Picks ({len(more_bets)})")

        for i in range(0, len(more_bets), 3):
            row_bets = more_bets.iloc[i:i+3]
            cols = st.columns(3)
            for j, (_, bet) in enumerate(row_bets.iterrows()):
                with cols[j]:
                    render_bet_card(bet, is_hero=False)

    # =============================================================================
    # SESSION SUMMARY
    # =============================================================================
    st.markdown("---")

    total_bets = len(all_bets)
    total_wagered = all_bets['bet_size'].sum() if len(all_bets) > 0 else 0
    avg_edge = all_bets['spread_error'].abs().mean() if len(all_bets) > 0 else 0
    best_edge = all_bets['spread_error'].abs().max() if len(all_bets) > 0 else 0

    # Count high confidence picks
    high_conf_bets = len(all_bets[all_bets['spread_error'].abs() >= 3.0])

    # Count confident intervals (don't cross zero)
    confident_intervals = 0
    avg_interval_width = None
    if 'interval_crosses_zero' in all_bets.columns:
        confident_intervals = len(all_bets[all_bets['interval_crosses_zero'] == False])
        if 'interval_width' in all_bets.columns and all_bets['interval_width'].notna().any():
            avg_interval_width = all_bets['interval_width'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Games", total_bets)
    with col2:
        # Show confident intervals if available, else high confidence
        if confident_intervals > 0:
            st.metric("Confident Picks", f"{confident_intervals} ✓", help="Bets where 80% prediction interval doesn't cross zero")
        else:
            st.metric("High Confidence", high_conf_bets)
    with col3:
        st.metric("Total Wagered", f"${total_wagered:.0f}")
    with col4:
        st.metric("Avg Edge", f"{avg_edge:.1f} pts")
    with col5:
        if avg_interval_width is not None:
            st.metric("Avg Interval", f"±{avg_interval_width/2:.1f} pts", help="Average 80% prediction interval half-width")
        else:
            st.metric("Best Edge", f"{best_edge:.1f} pts")

    # =============================================================================
    # COLLAPSED SECTIONS
    # =============================================================================
    with st.expander("Technical Details"):
        st.caption(f"""
        Model: Spread Error V14 (XGBoost) | Features: 33 safe pre-game only |
        Threshold: {SPREAD_ERROR_THRESHOLD} pts | Kelly Fraction: {KELLY_FRACTION}x |
        Validation: 59.9% win rate on 202 bets, +14.4% ROI, +29 units
        """)

        if not df_predictions.empty:
            # Build column list based on what's available
            display_cols = ['Game', 'Signal', 'confidence_tier', 'vegas_spread', 'spread_error', 'win_prob', 'bet_size']
            rename_map = {
                'confidence_tier': 'Confidence',
                'vegas_spread': 'Vegas',
                'spread_error': 'Pred Error',
                'win_prob': 'Est Win %',
                'bet_size': 'Kelly $',
            }

            # Add interval columns if available
            if 'lower_bound' in df_predictions.columns and df_predictions['lower_bound'].notna().any():
                display_cols.extend(['lower_bound', 'upper_bound'])
                rename_map['lower_bound'] = 'Q10'
                rename_map['upper_bound'] = 'Q90'

            st.dataframe(
                df_predictions[display_cols].rename(columns=rename_map),
                use_container_width=True,
                hide_index=True
            )

# =============================================================================
# TOTALS TAB
# =============================================================================
with totals_tab:
    if all_totals.empty:
        st.warning("No totals predictions available. The totals model may not be loaded or no games have O/U lines.")
    else:
        hero_totals = all_totals.head(2)
        more_totals = all_totals.iloc[2:]

        st.markdown('<div class="hero-title">TOP TOTALS PICKS (O/U Error Model)</div>', unsafe_allow_html=True)

        # Confidence explanation for totals
        st.markdown("""
<div style="background: #1E3A5F; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; border-left: 4px solid #22C55E;">
    <span style="color: #86EFAC; font-size: 13px;">
        📊 <b>Confidence = How wrong we think Vegas O/U is</b><br>
        🔥 HIGH (5+ pts) • ✅ MEDIUM-HIGH (3.5-5 pts) • ⚠️ MEDIUM (2-3.5 pts) • ⚡ LOW (1-2 pts) • ❄️ VERY LOW (&lt;1 pt)
    </span>
</div>
""", unsafe_allow_html=True)

        if len(hero_totals) == 1:
            render_totals_card(hero_totals.iloc[0], is_hero=True)
        elif len(hero_totals) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                render_totals_card(hero_totals.iloc[0], is_hero=True)
            with col2:
                render_totals_card(hero_totals.iloc[1], is_hero=True)

        # More totals picks
        if len(more_totals) > 0:
            st.markdown("---")
            st.subheader(f"More Totals Picks ({len(more_totals)})")

            for i in range(0, len(more_totals), 3):
                row_totals = more_totals.iloc[i:i+3]
                cols = st.columns(3)
                for j, (_, bet) in enumerate(row_totals.iterrows()):
                    with cols[j]:
                        render_totals_card(bet, is_hero=False)

        # Totals summary
        st.markdown("---")

        total_totals_bets = len(all_totals)
        total_totals_wagered = all_totals['bet_size'].sum() if len(all_totals) > 0 else 0
        avg_totals_edge = all_totals['totals_error'].abs().mean() if len(all_totals) > 0 else 0
        best_totals_edge = all_totals['totals_error'].abs().max() if len(all_totals) > 0 else 0
        high_conf_totals = len(all_totals[all_totals['totals_error'].abs() >= 3.0])

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Games", total_totals_bets)
        with col2:
            st.metric("High Confidence", high_conf_totals)
        with col3:
            st.metric("Total Wagered", f"${total_totals_wagered:.0f}")
        with col4:
            st.metric("Avg Edge", f"{avg_totals_edge:.1f} pts")
        with col5:
            st.metric("Best Edge", f"{best_totals_edge:.1f} pts")

        with st.expander("Totals Technical Details"):
            st.caption(f"""
            Model: Totals Error V2 (HistGradientBoosting) | Features: 28 |
            Kelly Fraction: {KELLY_FRACTION}x |
            Validation: 74.7% win rate, +42.6% ROI, Model MAE 10.47 vs Vegas 12.25
            """)

            if not df_totals.empty:
                st.dataframe(
                    df_totals[['Game', 'Signal', 'confidence_tier', 'vegas_ou', 'totals_error', 'predicted_total', 'win_prob', 'bet_size']].rename(columns={
                        'confidence_tier': 'Confidence',
                        'vegas_ou': 'Vegas O/U',
                        'totals_error': 'Pred Error',
                        'predicted_total': 'Model Total',
                        'win_prob': 'Est Win %',
                        'bet_size': 'Kelly $',
                    }),
                    use_container_width=True,
                    hide_index=True
                )

# =============================================================================
# RESULTS TAB - Historical Accuracy
# =============================================================================
with results_tab:
    st.markdown('<div class="hero-title">COMPLETED GAMES - Results & Accuracy</div>', unsafe_allow_html=True)

    if completed_games and not df_completed_predictions.empty:
        # Build results data with actual scores
        results_data = []
        for game in completed_games:
            home = game.get('homeTeam', '')
            away = game.get('awayTeam', '')
            home_score = game.get('homePoints', 0) or 0
            away_score = game.get('awayPoints', 0) or 0

            # Find the prediction for this game
            game_pred = df_completed_predictions[
                (df_completed_predictions['Home'] == home) |
                (df_completed_predictions['Away'] == away)
            ]

            if game_pred.empty:
                continue

            pred_row = game_pred.iloc[0]

            # Calculate actual result vs prediction
            actual_margin = home_score - away_score  # Positive = home won by X
            vegas_spread = pred_row.get('vegas_spread', 0)
            predicted_error = pred_row.get('spread_error', 0)

            # Determine if our pick was correct
            signal = pred_row.get('Signal', '')
            team_bet = pred_row.get('team_to_bet', '')
            spread_to_bet = pred_row.get('spread_to_bet', 0)

            # ATS result (Against The Spread)
            if signal == 'BUY':
                # We bet on home team to cover
                ats_result = actual_margin + vegas_spread  # If positive, home covered
                pick_won = ats_result > 0
            else:
                # We bet on away team to cover
                ats_result = -actual_margin - vegas_spread  # If positive, away covered
                pick_won = ats_result > 0

            results_data.append({
                'Game': f"{away} @ {home}",
                'Score': f"{away_score} - {home_score}",
                'Pick': f"{signal} {team_bet} {spread_to_bet:+.1f}",
                'Result': '✅ WIN' if pick_won else '❌ LOSS',
                'Margin': f"{ats_result:+.1f}",
                'Confidence': pred_row.get('confidence_tier', 'N/A'),
                'Bet Size': f"${pred_row.get('bet_size', 0):.0f}",
            })

        if results_data:
            df_results = pd.DataFrame(results_data)

            # Summary stats
            wins = sum(1 for r in results_data if '✅' in r['Result'])
            losses = len(results_data) - wins
            win_rate = wins / len(results_data) * 100 if results_data else 0

            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Games", len(results_data))
            with col2:
                st.metric("Record", f"{wins}-{losses}")
            with col3:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col4:
                # Color code win rate
                if win_rate >= 55:
                    st.metric("Status", "📈 Profitable")
                elif win_rate >= 50:
                    st.metric("Status", "📊 Break Even")
                else:
                    st.metric("Status", "📉 Review")

            st.markdown("---")

            # Results table
            st.dataframe(
                df_results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Result': st.column_config.TextColumn(width='small'),
                    'Margin': st.column_config.TextColumn(width='small'),
                }
            )
        else:
            st.info("No matched predictions found for completed games.")
    else:
        st.info(f"No completed games yet for this week. Check back after games finish!")
