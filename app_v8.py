"""
Sharp Betting Terminal V8 - CFB Only
College Football Spread Predictions with Ensemble Model + Monte Carlo

Improvements in V8:
- Environment variable API keys (security)
- Proper error handling with logging
- Configuration from centralized config
- Data validation
- Line movement tracking
- Improved caching
"""

import logging
import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cfbd
from urllib.parse import quote_plus

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
st.set_page_config(page_title="Sharp Betting Terminal V8", page_icon="$", layout="wide")

# =============================================================================
# API SETUP WITH ERROR HANDLING
# =============================================================================
@st.cache_resource
def setup_api():
    """Set up CFBD API client with error handling."""
    if not CFBD_API_KEY:
        st.error(
            "CFBD_API_KEY not set! Please set the environment variable:\n"
            "`export CFBD_API_KEY='your_key_here'`\n\n"
            "Get your free API key at: https://collegefootballdata.com/key"
        )
        st.stop()

    configuration = cfbd.Configuration()
    configuration.access_token = CFBD_API_KEY
    api_client = cfbd.ApiClient(configuration)

    return {
        'games': cfbd.GamesApi(api_client),
        'betting': cfbd.BettingApi(api_client),
    }

apis = setup_api()
games_api = apis['games']
betting_api = apis['betting']

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
            logger.debug(f"Model file {model_file} not found, trying next...")
            continue
        except Exception as e:
            logger.warning(f"Error loading {model_file}: {e}")
            continue

    st.error(
        "No model file found! Please run one of the training scripts first:\n"
        "- `python train_pro_stacking.py` (recommended)\n"
        "- `python train_ensemble.py`"
    )
    return None


def get_model_variance(model, features):
    """
    Calculate the standard deviation of predictions from base models.
    Used as a 'Confusion Filter' - high variance = models disagree.
    """
    predictions = []

    try:
        if hasattr(model, 'estimators_'):
            for name, estimator in model.estimators_:
                try:
                    pred = estimator.predict(features)[0]
                    predictions.append(pred)
                except Exception as e:
                    logger.debug(f"Estimator {name} prediction failed: {e}")

        elif hasattr(model, 'named_estimators_'):
            for name in model.named_estimators_:
                try:
                    est = model.named_estimators_[name]
                    pred = est.predict(features)[0]
                    predictions.append(pred)
                except Exception as e:
                    logger.debug(f"Estimator {name} prediction failed: {e}")

        elif hasattr(model, 'named_steps'):
            for step_name, step in model.named_steps.items():
                if hasattr(step, 'estimators_'):
                    for name, estimator in step.estimators_:
                        try:
                            pred = estimator.predict(features)[0]
                            predictions.append(pred)
                        except Exception:
                            pass

    except Exception as e:
        logger.warning(f"Error calculating model variance: {e}")

    if len(predictions) >= 2:
        return np.std(predictions), predictions
    return 0.0, predictions


@st.cache_data(ttl=300)
def load_history():
    """Load historical data with error handling."""
    try:
        df = pd.read_csv('cfb_data_smart.csv')
        df['home_rest'] = df['home_rest_days']
        df['away_rest'] = df['away_rest_days']
        df['net_epa'] = df['home_comp_off_ppa'] - df['away_comp_def_ppa']
        logger.info(f"Loaded {len(df)} games from history")
        return df
    except FileNotFoundError:
        logger.warning("Historical data file not found")
        st.warning("Historical data not found. Run fetch_cfb_data.py and feature_engineer.py first.")
        return None
    except Exception as e:
        logger.error(f"Error loading history: {e}")
        return None


@st.cache_data
def build_hfa_lookup():
    """Build home field advantage lookup dictionary."""
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
    except Exception as e:
        logger.error(f"Kelly calculation error: {e}")
        return {'bet_size': 0, 'kelly_fraction': 0, 'win_prob': 0.5, 'edge': 0}


# =============================================================================
# HELPERS
# =============================================================================
def sanitize_features(features_array):
    """Sanitize all features - replace None/NaN with 0.0."""
    sanitized = []
    for v in features_array.flatten():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            sanitized.append(0.0)
        else:
            sanitized.append(v)
    return np.array([sanitized])


def get_team_stats(team, history_df, season, week):
    """Get rolling stats for a team from historical data."""
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
                'rest_days': recent.get('home_rest_days', 7)
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
                'rest_days': recent.get('away_rest_days', 7)
            }
    except Exception as e:
        logger.warning(f"Error getting stats for {team}: {e}")

    return {
        'pregame_elo': 1500, 'last5_score_avg': 28, 'last5_defense_avg': 24,
        'comp_off_ppa': 0, 'comp_def_ppa': 0, 'hfa': 2.0, 'rest_days': 7
    }


# =============================================================================
# FETCH DATA
# =============================================================================
@st.cache_data(ttl=300)
def fetch_schedule(season, week):
    """Fetch game schedule."""
    try:
        games = games_api.get_games(year=season, week=week)
        logger.info(f"Fetched {len(games)} games for {season} week {week}")
        return games
    except Exception as e:
        logger.error(f"Error fetching schedule: {e}")
        st.error(f"Error fetching schedule: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_lines(season, week):
    """Fetch betting lines."""
    try:
        lines = betting_api.get_lines(year=season, week=week)
        return lines
    except Exception as e:
        logger.error(f"Error fetching lines: {e}")
        return []


def build_lines_dict(betting_lines):
    """Build lines dictionary with line movement."""
    lines = {}
    for line in betting_lines:
        if line.lines and len(line.lines) > 0:
            for book in line.lines:
                if book.spread is not None:
                    spread = float(book.spread)
                    opening = float(book.spread_open) if hasattr(book, 'spread_open') and book.spread_open else spread
                    lines[line.home_team] = {
                        'spread_current': spread,
                        'spread_opening': opening,
                        'line_movement': spread - opening,
                    }
                    break
    return lines


# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================
def generate_predictions(games, lines_dict, model, history_df, hfa_lookup, season, week):
    """Generate predictions for all games."""
    predictions = []

    for game in games:
        try:
            home, away = game.home_team, game.away_team
            if home not in lines_dict:
                continue

            vegas_line = lines_dict[home]['spread_current']
            line_move = lines_dict[home]['line_movement']

            home_stats = get_team_stats(home, history_df, season, week)
            away_stats = get_team_stats(away, history_df, season, week)
            if home_stats is None or away_stats is None:
                continue

            net_epa = home_stats['comp_off_ppa'] - away_stats['comp_def_ppa']
            rest_advantage = home_stats['rest_days'] - away_stats['rest_days']

            features = sanitize_features(np.array([[
                home_stats['pregame_elo'], away_stats['pregame_elo'],
                home_stats['last5_score_avg'], away_stats['last5_score_avg'],
                home_stats['last5_defense_avg'], away_stats['last5_defense_avg'],
                home_stats['comp_off_ppa'], away_stats['comp_off_ppa'],
                home_stats['comp_def_ppa'], away_stats['comp_def_ppa'],
                net_epa, hfa_lookup.get(home, 2.0), hfa_lookup.get(away, 0.0),
                home_stats['rest_days'], away_stats['rest_days'], rest_advantage
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

            kelly_result = kelly_recommendation(model_margin, vegas_line)
            bet_size = 0 if high_variance else kelly_result['bet_size']

            bet_rec = "Pass"
            if not high_variance:
                if edge > 4:
                    bet_rec = f"HOME {home} {vegas_line:+.1f}"
                elif edge < -4:
                    bet_rec = f"AWAY {away} {-vegas_line:+.1f}"

            context_parts = []
            if abs(line_move) > SIGNIFICANT_LINE_MOVE:
                context_parts.append(f"Line Move {'toward HOME' if line_move > 0 else 'toward AWAY'}")
            if high_variance:
                context_parts.append("HIGH VAR")
            context = " | ".join(context_parts) if context_parts else "-"

            predictions.append({
                'Game': f"{away} @ {home}", 'Home': home, 'Away': away,
                'Bet': bet_rec, 'Vegas': f"{vegas_line:+.1f}", 'Model': f"{model_line:+.1f}",
                'Edge': f"{edge:+.1f}", 'Win Prob': f"{cover_prob*100:.0f}%", 'Signal': signal_label,
                'Kelly Size': f"${bet_size:.0f}" if bet_size > 0 else "Pass",
                'Line Move': f"{line_move:+.1f}" if line_move != 0 else "-",
                'Context': context, 'Variance': f"{model_std:.1f}",
                'edge_value': edge, 'bet_size_value': bet_size, 'cover_prob_value': cover_prob,
                'variance_value': model_std, 'high_variance': high_variance, 'line_move_value': line_move,
            })
        except Exception as e:
            logger.error(f"Error predicting {game.away_team} @ {game.home_team}: {e}")

    return pd.DataFrame(predictions)


# =============================================================================
# STYLING
# =============================================================================
def style_signal(val):
    if val == 'BUY': return 'background-color: #90EE90; color: black; font-weight: bold'
    elif val == 'FADE': return 'background-color: #FFB6C1; color: black; font-weight: bold'
    return ''

def style_win_prob(val):
    try:
        prob = int(val.replace('%', ''))
        if prob >= 55: return 'background-color: #90EE90; color: black; font-weight: bold'
        elif prob <= 45: return 'background-color: #FFB6C1; color: black; font-weight: bold'
    except: pass
    return ''

def style_variance(val):
    try:
        if float(val) > VARIANCE_THRESHOLD: return 'background-color: #FFFF00; color: black; font-weight: bold'
    except: pass
    return ''

def style_line_move(val):
    try:
        if val != "-" and abs(float(val)) > SIGNIFICANT_LINE_MOVE:
            return 'background-color: #87CEEB; color: black'
    except: pass
    return ''


# =============================================================================
# MAIN APP
# =============================================================================
st.title("Sharp Betting Terminal V8")
st.markdown("*College Football | Ensemble Model + Monte Carlo + Line Movement*")

st.sidebar.header("Settings")
season = st.sidebar.selectbox("Season", [2025, 2024, 2023, 2022], index=0)
week = st.sidebar.number_input("Week", min_value=1, max_value=20, value=15)

if st.sidebar.button("Refresh", type="primary"):
    st.cache_data.clear()
    st.rerun()

model = load_model()
history_df = load_history()
hfa_lookup = build_hfa_lookup()

if model is None:
    st.stop()

with st.spinner("Fetching schedule and lines..."):
    games = fetch_schedule(season, week)
    betting_lines = fetch_lines(season, week)

if not games:
    st.warning(f"No games found for Week {week}, {season}")
    st.stop()

lines_dict = build_lines_dict(betting_lines)
st.info(f"Found {len(games)} games, {len(lines_dict)} with betting lines")

with st.spinner("Generating predictions..."):
    df_predictions = generate_predictions(games, lines_dict, model, history_df, hfa_lookup, season, week)

if df_predictions.empty:
    st.warning("No predictions generated.")
    st.stop()

# Filter games
actionable = df_predictions[
    ((df_predictions['cover_prob_value'] >= BUY_THRESHOLD) | (df_predictions['cover_prob_value'] <= FADE_THRESHOLD)) &
    (~df_predictions['high_variance'])
].copy()
high_var_games = df_predictions[df_predictions['high_variance']].copy()
line_move_games = df_predictions[df_predictions['line_move_value'].abs() > SIGNIFICANT_LINE_MOVE].copy()

tab1, tab2, tab3, tab4 = st.tabs(["Actionable Bets", "All Games", "Line Movement", "High Variance"])

with tab1:
    st.subheader(f"Actionable Bets ({len(actionable)} games)")
    if actionable.empty:
        st.info("No actionable bets this week")
    else:
        actionable = actionable.sort_values('cover_prob_value', key=lambda x: abs(x - 0.5), ascending=False)
        cols = ['Game', 'Signal', 'Win Prob', 'Vegas', 'Model', 'Edge', 'Line Move', 'Variance', 'Kelly Size']
        styled = actionable[cols].style.map(style_signal, subset=['Signal']).map(style_win_prob, subset=['Win Prob']).map(style_variance, subset=['Variance']).map(style_line_move, subset=['Line Move'])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown(f"**BUY:** {len(actionable[actionable['Signal']=='BUY'])} | **FADE:** {len(actionable[actionable['Signal']=='FADE'])} | **Total:** ${actionable['bet_size_value'].sum():.0f}")

with tab2:
    st.subheader(f"All Games ({len(df_predictions)} games)")
    all_games = df_predictions.sort_values('cover_prob_value', key=lambda x: abs(x - 0.5), ascending=False)
    cols = ['Game', 'Signal', 'Win Prob', 'Vegas', 'Model', 'Edge', 'Line Move', 'Variance', 'Kelly Size']
    styled = all_games[cols].style.map(style_signal, subset=['Signal']).map(style_win_prob, subset=['Win Prob']).map(style_variance, subset=['Variance']).map(style_line_move, subset=['Line Move'])
    st.dataframe(styled, use_container_width=True, hide_index=True)

with tab3:
    st.subheader(f"Line Movement ({len(line_move_games)} games)")
    if line_move_games.empty:
        st.info("No significant line movement")
    else:
        st.dataframe(line_move_games[['Game', 'Signal', 'Vegas', 'Model', 'Line Move', 'Context']], use_container_width=True, hide_index=True)

with tab4:
    st.subheader(f"High Variance ({len(high_var_games)} games)")
    st.warning("Models disagree - PASS recommended")
    if not high_var_games.empty:
        st.dataframe(high_var_games[['Game', 'Vegas', 'Model', 'Variance', 'Context']], use_container_width=True, hide_index=True)

st.divider()
st.caption(f"Model: Stacking Ensemble | Monte Carlo: {MONTE_CARLO_SIMS:,} sims | Variance Threshold: {VARIANCE_THRESHOLD} | Kelly: {KELLY_FRACTION}x")
