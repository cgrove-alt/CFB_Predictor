"""
Sharp Picks - Intuitive Betting UI
College Football Spread Predictions - Action-Oriented Interface

This UI tells you exactly what to bet:
- Team name + Spread
- Dollar amount
- Confidence level
"""

import logging
import os
import sys
import subprocess
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# Set up path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CFBD_API_KEY

# Try to import from new package structure, fallback to legacy
try:
    from src.utils.config import get_config
    from src.utils.logging_config import setup_logging, get_logger
    from src.prediction.monte_carlo import simulate_game, get_bet_signal, format_win_prob
    USE_NEW_MODULES = True
except ImportError:
    from monte_carlo import simulate_game, get_bet_signal, format_win_prob
    USE_NEW_MODULES = False

# Set up logging
if USE_NEW_MODULES:
    setup_logging(level="INFO", enable_file=False)
    logger = get_logger(__name__)
else:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
if USE_NEW_MODULES:
    config = get_config()
    VARIANCE_THRESHOLD = config.betting.high_variance_threshold
    MONTE_CARLO_SIMS = config.betting.monte_carlo_simulations
    MONTE_CARLO_STD = config.betting.monte_carlo_std_dev
    KELLY_FRACTION = config.betting.kelly_fraction
    BUY_THRESHOLD = config.betting.buy_threshold
    FADE_THRESHOLD = config.betting.fade_threshold
    SIGNIFICANT_LINE_MOVE = config.betting.significant_line_move
else:
    VARIANCE_THRESHOLD = 7.0
    MONTE_CARLO_SIMS = 10000
    MONTE_CARLO_STD = 14.0
    KELLY_FRACTION = 0.25
    BUY_THRESHOLD = 0.55
    FADE_THRESHOLD = 0.45
    SIGNIFICANT_LINE_MOVE = 1.5

# Page config
st.set_page_config(page_title="Sharp Picks", page_icon="$", layout="wide")

# =============================================================================
# CUSTOM CSS
# =============================================================================
def get_custom_css():
    return """
    <style>
    /* Hide default streamlit styling for cleaner look */
    .stApp {
        background-color: #0F172A;
    }

    /* Bet card styling */
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

    /* Confidence badges */
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
    .confidence-medium {
        background: #F59E0B;
        color: black;
    }

    /* Typography */
    .bet-instruction {
        font-size: 26px;
        font-weight: 700;
        margin: 12px 0 4px 0;
        color: #F8FAFC;
    }
    .bet-instruction-hero {
        font-size: 32px;
    }
    .opponent {
        color: #94A3B8;
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

    /* Hero section */
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

    /* Pass card (muted) */
    .pass-item {
        color: #64748B;
        padding: 8px 0;
        border-bottom: 1px solid #334155;
        font-size: 14px;
    }

    /* Summary metrics */
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
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """

st.markdown(get_custom_css(), unsafe_allow_html=True)

# =============================================================================
# API SETUP - Direct HTTP requests (cfbd library has field mapping bug)
# =============================================================================
CFBD_BASE_URL = "https://api.collegefootballdata.com"

def get_api_headers():
    """Get authorization headers for CFBD API."""
    if not CFBD_API_KEY:
        st.error(
            "CFBD_API_KEY not set! Please set the environment variable:\n"
            "`export CFBD_API_KEY='your_key_here'`\n\n"
            "Get your free API key at: https://collegefootballdata.com/key"
        )
        st.stop()
    return {'Authorization': f'Bearer {CFBD_API_KEY}'}

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================
@st.cache_resource
def load_model():
    """Load the ensemble model with proper error handling."""
    model_files = ['cfb_stacking.pkl', 'cfb_ensemble.pkl', 'cfb_smart_model.pkl']

    for model_file in model_files:
        try:
            model = joblib.load(model_file)
            logger.info(f"Loaded model from {model_file}")
            return model
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning(f"Error loading {model_file}: {e}")
            continue

    st.error("No model file found! Run training first.")
    return None


def get_model_variance(model, features):
    """Calculate variance across base models (confusion filter)."""
    predictions = []

    try:
        if hasattr(model, 'estimators_'):
            for name, estimator in model.estimators_:
                try:
                    pred = estimator.predict(features)[0]
                    predictions.append(pred)
                except Exception:
                    pass
        elif hasattr(model, 'named_estimators_'):
            for name in model.named_estimators_:
                try:
                    est = model.named_estimators_[name]
                    pred = est.predict(features)[0]
                    predictions.append(pred)
                except Exception:
                    pass
        elif hasattr(model, 'named_steps'):
            for step_name, step in model.named_steps.items():
                if hasattr(step, 'estimators_'):
                    for name, estimator in step.estimators_:
                        try:
                            pred = estimator.predict(features)[0]
                            predictions.append(pred)
                        except Exception:
                            pass
    except Exception:
        pass

    if len(predictions) >= 2:
        return np.std(predictions), predictions
    return 0.0, predictions


@st.cache_data(ttl=300)
def load_history():
    """Load historical data."""
    try:
        df = pd.read_csv('cfb_data_smart.csv')
        df['home_rest'] = df['home_rest_days']
        df['away_rest'] = df['away_rest_days']
        df['net_epa'] = df['home_comp_off_ppa'] - df['away_comp_def_ppa']
        return df
    except FileNotFoundError:
        st.warning("Historical data not found. Run fetch_cfb_data.py first.")
        return None


@st.cache_data
def build_hfa_lookup():
    """Build home field advantage lookup."""
    df = load_history()
    if df is None:
        return {}
    hfa = {}
    for _, row in df[['home_team', 'home_team_hfa']].dropna().drop_duplicates().iterrows():
        hfa[row['home_team']] = row['home_team_hfa']
    return hfa


# =============================================================================
# KELLY CRITERION
# =============================================================================
def kelly_recommendation(model_margin, vegas_line, bankroll=1000, odds=-110):
    """Calculate Kelly Criterion bet recommendation."""
    try:
        edge = vegas_line - (-model_margin)
        win_prob = max(0.01, min(0.99, 0.50 + (edge / 100)))

        decimal_odds = 1 + (100 / abs(odds)) if odds < 0 else 1 + (odds / 100)
        b = decimal_odds - 1
        q = 1 - win_prob

        kelly_fraction = max(0, (b * win_prob - q) / b)
        bet_size = min(bankroll * kelly_fraction * KELLY_FRACTION, bankroll * 0.10)

        return {
            'bet_size': round(bet_size, 2),
            'kelly_fraction': kelly_fraction,
            'win_prob': win_prob,
            'edge': edge
        }
    except Exception:
        return {'bet_size': 0, 'kelly_fraction': 0, 'win_prob': 0.5, 'edge': 0}


# =============================================================================
# HELPERS
# =============================================================================
def sanitize_features(features_array):
    """Sanitize features - replace None/NaN with 0.0."""
    sanitized = []
    for v in features_array.flatten():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            sanitized.append(0.0)
        else:
            sanitized.append(v)
    return np.array([sanitized])


def get_team_stats(team, history_df, season, week):
    """Get rolling stats for a team."""
    if history_df is None:
        return None

    try:
        home_games = history_df[(history_df['home_team'] == team) &
                                ((history_df['season'] < season) |
                                 ((history_df['season'] == season) & (history_df['week'] < week)))]
        away_games = history_df[(history_df['away_team'] == team) &
                                ((history_df['season'] < season) |
                                 ((history_df['season'] == season) & (history_df['week'] < week)))]

        if len(home_games) > 0:
            recent = home_games.sort_values(['season', 'week'], ascending=False).iloc[0]
            return {
                'pregame_elo': recent.get('home_pregame_elo', 1500),
                'last5_score_avg': recent.get('home_last5_score_avg', 28),
                'last5_defense_avg': recent.get('home_last5_defense_avg', 24),
                'comp_off_ppa': recent.get('home_comp_off_ppa', 0),
                'comp_def_ppa': recent.get('home_comp_def_ppa', 0),
                'hfa': recent.get('home_team_hfa', 2.0),
                'rest_days': recent.get('home_rest_days', 7),
                # Additional features for 21-feature model
                'comp_pass_ppa': recent.get('home_comp_pass_ppa', 0),
                'comp_success': recent.get('home_comp_success', 0.4),
            }
        elif len(away_games) > 0:
            recent = away_games.sort_values(['season', 'week'], ascending=False).iloc[0]
            return {
                'pregame_elo': recent.get('away_pregame_elo', 1500),
                'last5_score_avg': recent.get('away_last5_score_avg', 28),
                'last5_defense_avg': recent.get('away_last5_defense_avg', 24),
                'comp_off_ppa': recent.get('away_comp_off_ppa', 0),
                'comp_def_ppa': recent.get('away_comp_def_ppa', 0),
                'hfa': recent.get('away_team_hfa', 0),
                'rest_days': recent.get('away_rest_days', 7),
                # Additional features for 21-feature model
                'comp_pass_ppa': recent.get('away_comp_pass_ppa', 0),
                'comp_success': recent.get('away_comp_success', 0.4),
            }
    except Exception:
        pass

    return {
        'pregame_elo': 1500, 'last5_score_avg': 28, 'last5_defense_avg': 24,
        'comp_off_ppa': 0, 'comp_def_ppa': 0, 'hfa': 2.0, 'rest_days': 7,
        'comp_pass_ppa': 0, 'comp_success': 0.4
    }


# =============================================================================
# FETCH DATA - Direct API calls (cfbd library has camelCase mapping bug)
# =============================================================================
@st.cache_data(ttl=300)
def fetch_schedule(season, week):
    """Fetch game schedule using direct API call."""
    try:
        url = f"{CFBD_BASE_URL}/games?year={season}&week={week}"
        resp = requests.get(url, headers=get_api_headers())
        if resp.status_code == 200:
            return resp.json()  # Returns list of dicts with camelCase keys
        else:
            st.error(f"Error fetching schedule: {resp.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching schedule: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_lines(season, week):
    """Fetch betting lines using direct API call."""
    try:
        url = f"{CFBD_BASE_URL}/lines?year={season}&week={week}"
        resp = requests.get(url, headers=get_api_headers())
        if resp.status_code == 200:
            return resp.json()  # Returns list of dicts with camelCase keys
        return []
    except Exception:
        return []


def build_lines_dict(betting_lines):
    """Build lines dictionary from API response (camelCase keys)."""
    lines = {}
    for line in betting_lines:
        # API returns camelCase keys
        line_books = line.get('lines', [])
        if line_books and len(line_books) > 0:
            for book in line_books:
                spread_val = book.get('spread')
                if spread_val is not None:
                    spread = float(spread_val)
                    opening = float(book.get('spreadOpen', spread)) if book.get('spreadOpen') else spread
                    lines[line['homeTeam']] = {
                        'spread_current': spread,
                        'spread_opening': opening,
                        'line_movement': spread - opening,
                    }
                    break
    return lines


# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================
def generate_predictions(games, lines_dict, model, history_df, hfa_lookup, season, week, bankroll):
    """Generate predictions for all games (API returns camelCase keys)."""
    predictions = []

    for game in games:
        try:
            # API returns camelCase keys (homeTeam, awayTeam)
            home, away = game['homeTeam'], game['awayTeam']
            if not home or not away or home not in lines_dict:
                continue

            vegas_line = lines_dict[home]['spread_current']
            line_move = lines_dict[home]['line_movement']

            home_stats = get_team_stats(home, history_df, season, week)
            away_stats = get_team_stats(away, history_df, season, week)
            if home_stats is None or away_stats is None:
                continue

            net_epa = home_stats['comp_off_ppa'] - away_stats['comp_def_ppa']
            rest_advantage = home_stats['rest_days'] - away_stats['rest_days']

            # Compute the 5 interaction features required by the 21-feature model
            rest_diff = home_stats['rest_days'] - away_stats['rest_days']
            elo_diff = home_stats['pregame_elo'] - away_stats['pregame_elo']
            pass_efficiency_diff = home_stats['comp_pass_ppa'] - away_stats['comp_pass_ppa']
            epa_elo_interaction = net_epa * elo_diff
            success_diff = home_stats['comp_success'] - away_stats['comp_success']

            features = sanitize_features(np.array([[
                # Original 16 features
                home_stats['pregame_elo'], away_stats['pregame_elo'],
                home_stats['last5_score_avg'], away_stats['last5_score_avg'],
                home_stats['last5_defense_avg'], away_stats['last5_defense_avg'],
                home_stats['comp_off_ppa'], away_stats['comp_off_ppa'],
                home_stats['comp_def_ppa'], away_stats['comp_def_ppa'],
                net_epa, hfa_lookup.get(home, 2.0), hfa_lookup.get(away, 0.0),
                home_stats['rest_days'], away_stats['rest_days'], rest_advantage,
                # 5 interaction features (17-21)
                rest_diff, elo_diff, pass_efficiency_diff, epa_elo_interaction, success_diff
            ]]))

            model_margin = model.predict(features)[0]
            model_line = -model_margin
            edge = vegas_line - model_line

            model_std, _ = get_model_variance(model, features)
            high_variance = model_std > VARIANCE_THRESHOLD

            mc_result = simulate_game(model_margin, vegas_line, std_dev=MONTE_CARLO_STD, simulations=MONTE_CARLO_SIMS)
            cover_prob = mc_result['cover_probability']
            _, signal_label = get_bet_signal(cover_prob, BUY_THRESHOLD, FADE_THRESHOLD)

            if high_variance:
                signal_label = "PASS"

            kelly_result = kelly_recommendation(model_margin, vegas_line, bankroll=bankroll)
            bet_size = 0 if high_variance else kelly_result['bet_size']

            # Determine team to bet on and the spread to display
            if signal_label == 'BUY':
                team_to_bet = home
                opponent = away
                spread_to_bet = vegas_line
            elif signal_label == 'FADE':
                team_to_bet = away
                opponent = home
                spread_to_bet = -vegas_line
            else:
                team_to_bet = None
                opponent = None
                spread_to_bet = None

            predictions.append({
                'Home': home,
                'Away': away,
                'Game': f"{away} @ {home}",
                'Signal': signal_label,
                'team_to_bet': team_to_bet,
                'opponent': opponent,
                'spread_to_bet': spread_to_bet,
                'vegas_line': vegas_line,
                'model_line': model_line,
                'edge': edge,
                'cover_prob': cover_prob,
                'bet_size': bet_size,
                'variance': model_std,
                'high_variance': high_variance,
                'line_move': line_move,
            })
        except Exception as e:
            logger.error(f"Error predicting {game.get('awayTeam', '?')} @ {game.get('homeTeam', '?')}: {e}")

    return pd.DataFrame(predictions)


# =============================================================================
# BET CARD COMPONENT
# =============================================================================
def render_bet_card(bet, is_hero=False):
    """Render a styled bet card."""
    signal_class = "buy" if bet['Signal'] == 'BUY' else "fade"

    # Confidence level
    if bet['cover_prob'] >= 0.60:
        conf_badge = '<span class="confidence-badge confidence-high">HIGH CONFIDENCE</span>'
    elif bet['cover_prob'] >= 0.55 or bet['cover_prob'] <= 0.45:
        conf_badge = '<span class="confidence-badge confidence-medium">MEDIUM CONFIDENCE</span>'
    else:
        conf_badge = ''

    # For FADE, we show probability of the other side
    display_prob = bet['cover_prob'] if bet['Signal'] == 'BUY' else (1 - bet['cover_prob'])

    # Format spread
    spread_str = f"{bet['spread_to_bet']:+.1f}" if bet['spread_to_bet'] != 0 else "PK"

    instruction_class = "bet-instruction-hero" if is_hero else "bet-instruction"
    amount_class = "bet-amount-hero" if is_hero else "bet-amount"

    html = f"""
    <div class="bet-card {signal_class}">
        {conf_badge}
        <div class="{instruction_class}">BET: {bet['team_to_bet']} {spread_str}</div>
        <div class="opponent">vs {bet['opponent']}</div>
        <div class="{amount_class}">${bet['bet_size']:.0f}</div>
        <div class="win-prob">{display_prob*100:.0f}% Win Probability</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_pass_item(bet):
    """Render a muted pass game item."""
    reason = "Model uncertainty" if bet['high_variance'] else "Low edge"
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
    data_files = ['cfb_data_smart.csv', 'cfb_data.csv']
    for f in data_files:
        if os.path.exists(f):
            mod_time = os.path.getmtime(f)
            mod_date = datetime.fromtimestamp(mod_time)
            days_old = (datetime.now() - mod_date).days
            return mod_date, days_old, f
    return None, None, None


def refresh_data():
    """Run refresh_all_data.py to fetch all required data (games, PPA, stats)."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'refresh_all_data.py')
    result = subprocess.run(['python3', script_path], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0, result.stdout, result.stderr


# =============================================================================
# MAIN APP
# =============================================================================

# Header
col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
with col1:
    st.title("Sharp Picks")
with col2:
    season = st.selectbox("Season", [2025, 2024, 2023, 2022], index=0, label_visibility="collapsed")
with col3:
    week = st.number_input("Week", min_value=1, max_value=20, value=15, label_visibility="collapsed")
with col4:
    # Bankroll with session state persistence
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 1000
    bankroll = st.number_input("Bankroll", min_value=100, max_value=100000, value=st.session_state.bankroll, step=100, label_visibility="collapsed", format="%d")
    st.session_state.bankroll = bankroll

# Buttons row
col_btn1, col_btn2, col_spacer = st.columns([1, 1, 6])
with col_btn1:
    if st.button("Refresh Cache", type="secondary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_btn2:
    if st.button("Refresh Data", type="primary", use_container_width=True):
        with st.spinner("Fetching latest data from CFBD API..."):
            success, stdout, stderr = refresh_data()
            if success:
                st.cache_data.clear()
                st.success("Data refreshed successfully!")
                st.rerun()
            else:
                st.error(f"Error refreshing data: {stderr}")

# Data freshness display
mod_date, days_old, data_file = get_data_freshness()
if mod_date:
    freshness_text = f"Data last updated: {mod_date.strftime('%b %d, %Y at %I:%M %p')}"
    if days_old > 7:
        st.warning(f"⚠️ {freshness_text} ({days_old} days old) - Consider refreshing data!")
    else:
        st.caption(f"Week {week}, {season} College Football | Bankroll: ${bankroll:,} | {freshness_text}")
else:
    st.caption(f"Week {week}, {season} College Football | Bankroll: ${bankroll:,}")

# Load data
model = load_model()
history_df = load_history()
hfa_lookup = build_hfa_lookup()

if model is None:
    st.stop()

# Fetch and generate predictions
with st.spinner("Loading picks..."):
    games = fetch_schedule(season, week)
    betting_lines = fetch_lines(season, week)

if not games:
    st.warning(f"No games found for Week {week}, {season}")
    st.stop()

lines_dict = build_lines_dict(betting_lines)

with st.spinner("Generating predictions..."):
    df_predictions = generate_predictions(games, lines_dict, model, history_df, hfa_lookup, season, week, bankroll)

if df_predictions.empty:
    st.warning("No predictions generated.")
    st.stop()

# Classify bets
actionable = df_predictions[
    (df_predictions['Signal'].isin(['BUY', 'FADE'])) &
    (~df_predictions['high_variance'])
].copy()

# Sort by conviction (distance from 50%)
actionable = actionable.sort_values(
    'cover_prob',
    key=lambda x: abs(x - 0.5),
    ascending=False
)

pass_games = df_predictions[
    (df_predictions['Signal'] == 'PASS') | (df_predictions['high_variance'])
]

hero_bets = actionable.head(2)
more_bets = actionable.iloc[2:]

# =============================================================================
# HERO SECTION - TOP PICKS
# =============================================================================
if len(actionable) > 0:
    st.markdown('<div class="hero-title">TODAY\'S TOP PICKS</div>', unsafe_allow_html=True)

    if len(hero_bets) == 1:
        render_bet_card(hero_bets.iloc[0], is_hero=True)
    elif len(hero_bets) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            render_bet_card(hero_bets.iloc[0], is_hero=True)
        with col2:
            render_bet_card(hero_bets.iloc[1], is_hero=True)
else:
    st.info("No strong picks this week. Check back later!")

# =============================================================================
# MORE PICKS SECTION
# =============================================================================
if len(more_bets) > 0:
    st.markdown("---")
    st.subheader(f"More Picks ({len(more_bets)})")

    # Display in rows of 3
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

total_bets = len(actionable)
total_wagered = actionable['bet_size'].sum() if len(actionable) > 0 else 0
avg_prob = actionable['cover_prob'].apply(lambda x: x if x > 0.5 else 1-x).mean() * 100 if len(actionable) > 0 else 0
best_edge = actionable['edge'].abs().max() if len(actionable) > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Bets", total_bets)
with col2:
    st.metric("Total Wagered", f"${total_wagered:.0f}")
with col3:
    st.metric("Avg Win Prob", f"{avg_prob:.0f}%")
with col4:
    st.metric("Best Edge", f"{best_edge:.1f} pts")

# =============================================================================
# COLLAPSED SECTIONS
# =============================================================================
with st.expander(f"No Action ({len(pass_games)} games)"):
    if len(pass_games) > 0:
        for _, bet in pass_games.iterrows():
            render_pass_item(bet)
    else:
        st.write("All games have actionable signals!")

with st.expander("Technical Details"):
    st.caption(f"Model: Stacking Ensemble | Monte Carlo: {MONTE_CARLO_SIMS:,} sims | Variance Threshold: {VARIANCE_THRESHOLD} | Kelly Fraction: {KELLY_FRACTION}x")

    if not df_predictions.empty:
        st.dataframe(
            df_predictions[['Game', 'Signal', 'vegas_line', 'model_line', 'edge', 'cover_prob', 'bet_size', 'variance']].rename(columns={
                'vegas_line': 'Vegas',
                'model_line': 'Model',
                'edge': 'Edge',
                'cover_prob': 'Cover Prob',
                'bet_size': 'Kelly $',
                'variance': 'Variance'
            }),
            use_container_width=True,
            hide_index=True
        )
