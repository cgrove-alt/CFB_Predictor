"""
Sharp Betting Terminal V7 - CFB Only
College Football Spread Predictions with Ensemble Model + Monte Carlo
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cfbd
from urllib.parse import quote_plus

from config import CFBD_API_KEY
from monte_carlo import simulate_game, get_bet_signal, format_win_prob

# Page config
st.set_page_config(page_title="Sharp Betting Terminal V7", page_icon="💰", layout="wide")

# Configure CFBD API
configuration = cfbd.Configuration()
configuration.access_token = CFBD_API_KEY
api_client = cfbd.ApiClient(configuration)
games_api = cfbd.GamesApi(api_client)
betting_api = cfbd.BettingApi(api_client)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================
@st.cache_resource
def load_model():
    """Load the ensemble model (or fallback to single model)."""
    try:
        # Try stacking model first (has individual estimators)
        return joblib.load('cfb_stacking.pkl')
    except Exception:
        try:
            # Try ensemble
            return joblib.load('cfb_ensemble.pkl')
        except Exception:
            try:
                # Fallback to single model
                return joblib.load('cfb_smart_model.pkl')
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None

def get_model_variance(model, features):
    """
    Calculate the standard deviation of predictions from base models.
    Used as a 'Confusion Filter' - high variance = models disagree.

    Returns: (std_dev, individual_predictions)
    """
    predictions = []

    # Check if it's a StackingRegressor
    if hasattr(model, 'estimators_'):
        # Get predictions from each fitted estimator
        for name, estimator in model.estimators_:
            try:
                pred = estimator.predict(features)[0]
                predictions.append(pred)
            except Exception:
                pass
    # Check if it's a VotingRegressor
    elif hasattr(model, 'estimators_') or hasattr(model, 'named_estimators_'):
        try:
            for name in model.named_estimators_:
                est = model.named_estimators_[name]
                pred = est.predict(features)[0]
                predictions.append(pred)
        except Exception:
            pass
    # Check if it's a Pipeline with a VotingRegressor or StackingRegressor
    elif hasattr(model, 'named_steps'):
        # Get the final estimator
        for step_name, step in model.named_steps.items():
            if hasattr(step, 'estimators_'):
                for name, estimator in step.estimators_:
                    try:
                        # Need to transform features first through earlier pipeline steps
                        pred = estimator.predict(features)[0]
                        predictions.append(pred)
                    except Exception:
                        pass
            elif hasattr(step, 'named_estimators_'):
                for name in step.named_estimators_:
                    try:
                        est = step.named_estimators_[name]
                        pred = est.predict(features)[0]
                        predictions.append(pred)
                    except Exception:
                        pass

    if len(predictions) >= 2:
        return np.std(predictions), predictions
    else:
        return 0.0, predictions

@st.cache_data(ttl=300)
def load_history():
    try:
        df = pd.read_csv('cfb_data_smart.csv')
        # Create derived columns
        df['home_rest'] = df['home_rest_days']
        df['away_rest'] = df['away_rest_days']
        df['net_epa'] = df['home_comp_off_ppa'] - df['away_comp_def_ppa']
        return df
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return None

# Build HFA lookup dictionary
@st.cache_data
def build_hfa_lookup():
    df = load_history()
    if df is None:
        return {}
    hfa = {}
    for _, row in df[['home_team', 'home_team_hfa']].dropna().drop_duplicates().iterrows():
        hfa[row['home_team']] = row['home_team_hfa']
    return hfa

# ============================================================
# KELLY CRITERION
# ============================================================
def kelly_recommendation(model_margin, vegas_line, bankroll=1000, odds=-110):
    """Calculate Kelly Criterion bet recommendation."""
    edge = vegas_line - (-model_margin)

    # Estimate win probability from edge
    if edge > 0:
        win_prob = 0.50 + (edge / 100)
    else:
        win_prob = 0.50 + (edge / 100)

    win_prob = max(0.01, min(0.99, win_prob))

    # Kelly formula for standard -110 odds
    decimal_odds = 1 + (100 / abs(odds)) if odds < 0 else 1 + (odds / 100)
    b = decimal_odds - 1
    q = 1 - win_prob

    kelly_fraction = (b * win_prob - q) / b
    kelly_fraction = max(0, kelly_fraction)

    # Use fractional Kelly (0.25x)
    fractional = 0.25
    bet_size = bankroll * kelly_fraction * fractional
    bet_size = min(bet_size, bankroll * 0.10)  # Max 10% of bankroll

    return {
        'bet_size': round(bet_size, 2),
        'kelly_fraction': kelly_fraction,
        'win_prob': win_prob,
        'edge': edge
    }

def format_bet_size(bet_size):
    if bet_size <= 0:
        return "Pass"
    elif bet_size < 25:
        return f"${bet_size:.0f}"
    elif bet_size < 50:
        return f"${bet_size:.0f}"
    else:
        return f"${bet_size:.0f}"

# ============================================================
# HELPERS
# ============================================================
def sanitize_value(v, default=0.0):
    """Sanitize a value - replace None/NaN with default."""
    if v is None:
        return default
    try:
        if np.isnan(v):
            return default
    except (TypeError, ValueError):
        pass
    return v

def sanitize_features(features_array):
    """Sanitize all features in an array - replace None/NaN with 0.0."""
    sanitized = []
    for v in features_array.flatten():
        sanitized.append(sanitize_value(v, 0.0))
    return np.array([sanitized])

def get_injury_url(team):
    """Get injury report URL for a team."""
    return f"https://www.google.com/search?q={quote_plus(team + ' football injury report')}"

def get_recommended_bet(edge, home_team, away_team, vegas_line):
    if edge > 4:
        return f"HOME {home_team} {vegas_line:+.1f}"
    elif edge < -4:
        return f"AWAY {away_team} {-vegas_line:+.1f}"
    else:
        return "Pass"

def get_context(wind_speed, line_move):
    """Get situational context indicators."""
    alerts = []
    if wind_speed is not None and wind_speed > 15:
        alerts.append("Wind")
    if line_move is not None and abs(line_move) > 1.5:
        alerts.append("Line Move")
    return " | ".join(alerts) if alerts else "-"

# ============================================================
# GET TEAM STATS FROM HISTORY
# ============================================================
def get_team_stats(team, history_df, season, week):
    """Get rolling stats for a team from historical data."""
    if history_df is None:
        return None

    # Get games where this team played (home or away)
    home_games = history_df[(history_df['home_team'] == team) &
                            ((history_df['season'] < season) |
                             ((history_df['season'] == season) & (history_df['week'] < week)))]
    away_games = history_df[(history_df['away_team'] == team) &
                            ((history_df['season'] < season) |
                             ((history_df['season'] == season) & (history_df['week'] < week)))]

    # Get most recent stats
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

    # Default values
    return {
        'pregame_elo': 1500,
        'last5_score_avg': 28,
        'last5_defense_avg': 24,
        'comp_off_ppa': 0,
        'comp_def_ppa': 0,
        'hfa': 2.0,
        'rest_days': 7
    }

# ============================================================
# FETCH SCHEDULE AND LINES
# ============================================================
@st.cache_data(ttl=300)
def fetch_schedule(season, week):
    try:
        return games_api.get_games(year=season, week=week)
    except Exception as e:
        st.error(f"Error fetching schedule: {e}")
        return []

@st.cache_data(ttl=300)
def fetch_lines(season, week):
    try:
        return betting_api.get_lines(year=season, week=week)
    except Exception as e:
        st.error(f"Error fetching lines: {e}")
        return []

def build_lines_dict(betting_lines):
    """Build lines dictionary with spread and opening spread."""
    lines = {}
    for line in betting_lines:
        if line.lines and len(line.lines) > 0:
            for book in line.lines:
                if book.spread is not None:
                    opening = book.spread_open if hasattr(book, 'spread_open') and book.spread_open else book.spread
                    lines[line.home_team] = {
                        'spread_current': float(book.spread),
                        'spread_opening': float(opening) if opening else float(book.spread)
                    }
                    break
    return lines

# ============================================================
# GENERATE PREDICTIONS WITH MONTE CARLO
# ============================================================
def generate_predictions(games, lines_dict, model, history_df, hfa_lookup, season, week):
    predictions = []

    for game in games:
        home = game.home_team
        away = game.away_team

        # Skip games without lines
        if home not in lines_dict:
            continue

        vegas_line = lines_dict[home]['spread_current']
        opening_line = lines_dict[home]['spread_opening']
        line_move = vegas_line - opening_line

        # Get team stats
        home_stats = get_team_stats(home, history_df, season, week)
        away_stats = get_team_stats(away, history_df, season, week)

        if home_stats is None or away_stats is None:
            continue

        # Calculate derived features
        net_epa = home_stats['comp_off_ppa'] - away_stats['comp_def_ppa']
        rest_advantage = home_stats['rest_days'] - away_stats['rest_days']

        # Build 16 features (matching train_ensemble.py)
        features = np.array([[
            home_stats['pregame_elo'],        # 1. home_pregame_elo
            away_stats['pregame_elo'],        # 2. away_pregame_elo
            home_stats['last5_score_avg'],    # 3. home_last5_score_avg
            away_stats['last5_score_avg'],    # 4. away_last5_score_avg
            home_stats['last5_defense_avg'],  # 5. home_last5_defense_avg
            away_stats['last5_defense_avg'],  # 6. away_last5_defense_avg
            home_stats['comp_off_ppa'],       # 7. home_comp_off_ppa
            away_stats['comp_off_ppa'],       # 8. away_comp_off_ppa
            home_stats['comp_def_ppa'],       # 9. home_comp_def_ppa
            away_stats['comp_def_ppa'],       # 10. away_comp_def_ppa
            net_epa,                          # 11. net_epa
            hfa_lookup.get(home, 2.0),        # 12. home_team_hfa
            hfa_lookup.get(away, 0.0),        # 13. away_team_hfa
            home_stats['rest_days'],          # 14. home_rest
            away_stats['rest_days'],          # 15. away_rest
            rest_advantage                    # 16. rest_advantage
        ]])

        # Sanitize features - replace None/NaN with 0.0
        features = sanitize_features(features)

        # Predict margin using ensemble model (with error handling)
        try:
            model_margin = model.predict(features)[0]
        except Exception as e:
            # Skip this game if prediction fails
            continue
        model_line = -model_margin
        edge = vegas_line - model_line

        # Calculate model variance (Confusion Filter)
        model_std, individual_preds = get_model_variance(model, features)
        high_variance = model_std > 7.0  # Models heavily disagree

        # Monte Carlo Simulation
        mc_result = simulate_game(model_margin, vegas_line, std_dev=14.0, simulations=10000)
        cover_prob = mc_result['cover_probability']
        signal, signal_label = get_bet_signal(cover_prob)

        # Override signal if high variance (Confusion Filter)
        if high_variance:
            signal_label = "PASS"
            signal = "PASS"

        # Kelly
        kelly_result = kelly_recommendation(model_margin, vegas_line, bankroll=1000)
        bet_size = kelly_result['bet_size']

        # Override bet size if high variance
        if high_variance:
            bet_size = 0

        # Context (no wind data available from API, use line move)
        context = get_context(None, line_move)

        # Add variance warning to context
        if high_variance:
            context = "HIGH VAR" if not context else f"{context}, HIGH VAR"

        predictions.append({
            'Game': f"{away} @ {home}",
            'Home': home,
            'Away': away,
            'Bet': "Pass" if high_variance else get_recommended_bet(edge, home, away, vegas_line),
            'Vegas': f"{vegas_line:+.1f}",
            'Model': f"{model_line:+.1f}",
            'Edge': f"{edge:+.1f}",
            'Win Prob': f"{cover_prob*100:.0f}%",
            'Signal': signal_label,
            'Kelly Size': format_bet_size(bet_size),
            'Context': context,
            'Variance': f"{model_std:.1f}",
            'Injury': get_injury_url(home),
            'edge_value': edge,
            'bet_size_value': bet_size,
            'cover_prob_value': cover_prob,
            'variance_value': model_std,
            'high_variance': high_variance
        })

    return pd.DataFrame(predictions)

# ============================================================
# STYLING FOR WIN PROB
# ============================================================
def style_signal(val):
    """Style the Signal column."""
    if val == 'BUY':
        return 'background-color: #90EE90; color: black; font-weight: bold'
    elif val == 'FADE':
        return 'background-color: #FFB6C1; color: black; font-weight: bold'
    else:
        return ''

def style_win_prob(val):
    """Style the Win Prob column."""
    try:
        prob = int(val.replace('%', ''))
        if prob >= 55:
            return 'background-color: #90EE90; color: black; font-weight: bold'
        elif prob <= 45:
            return 'background-color: #FFB6C1; color: black; font-weight: bold'
        else:
            return ''
    except:
        return ''

def style_variance(val):
    """Style the Variance column - yellow for high variance."""
    try:
        variance = float(val)
        if variance > 7.0:
            return 'background-color: #FFFF00; color: black; font-weight: bold'
        else:
            return ''
    except:
        return ''

def style_row_by_variance(row):
    """Style entire row yellow if high variance."""
    if row.get('high_variance', False):
        return ['background-color: #FFFF99'] * len(row)
    return [''] * len(row)

# ============================================================
# MAIN APP
# ============================================================

st.title("Sharp Betting Terminal V7")
st.markdown("*College Football | Ensemble Model + Monte Carlo Simulation*")

# Sidebar
st.sidebar.header("Settings")
season = st.sidebar.selectbox("Season", [2025, 2024, 2023, 2022], index=0)
week = st.sidebar.number_input("Week", min_value=1, max_value=20, value=15)

if st.sidebar.button("Refresh", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Load resources
model = load_model()
history_df = load_history()
hfa_lookup = build_hfa_lookup()

if model is None:
    st.error("Failed to load model. Please run train_ensemble.py first.")
    st.stop()

# Fetch data
with st.spinner("Fetching schedule and lines..."):
    games = fetch_schedule(season, week)
    betting_lines = fetch_lines(season, week)

if not games:
    st.warning(f"No games found for Week {week}, {season}")
    st.stop()

lines_dict = build_lines_dict(betting_lines)

st.info(f"Found {len(games)} games, {len(lines_dict)} with betting lines")

# Generate predictions
with st.spinner("Generating predictions with Monte Carlo..."):
    df_predictions = generate_predictions(games, lines_dict, model, history_df, hfa_lookup, season, week)

if df_predictions.empty:
    st.warning("No predictions generated. Check if games have betting lines.")
    st.stop()

# ============================================================
# DISPLAY RESULTS
# ============================================================

# Filter to actionable bets (Win Prob > 55% or < 45%) AND NOT high variance
actionable = df_predictions[
    ((df_predictions['cover_prob_value'] >= 0.55) | (df_predictions['cover_prob_value'] <= 0.45)) &
    (df_predictions['high_variance'] == False)
].copy()
high_var_games = df_predictions[df_predictions['high_variance'] == True].copy()
all_games = df_predictions.copy()

# Tabs
tab1, tab2, tab3 = st.tabs(["Actionable Bets", "All Games", "High Variance (PASS)"])

with tab1:
    st.subheader(f"Actionable Bets ({len(actionable)} games)")

    if actionable.empty:
        st.info("No actionable bets this week (need Win Prob >55% or <45% AND low model variance)")
    else:
        # Sort by cover probability (most confident first)
        actionable = actionable.sort_values('cover_prob_value', key=lambda x: abs(x - 0.5), ascending=False)

        # Display table with styling
        display_cols = ['Game', 'Signal', 'Win Prob', 'Vegas', 'Model', 'Edge', 'Variance', 'Kelly Size', 'Context']
        styled_df = actionable[display_cols].style.applymap(
            style_signal, subset=['Signal']
        ).applymap(
            style_win_prob, subset=['Win Prob']
        ).applymap(
            style_variance, subset=['Variance']
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

        # Summary
        buy_count = len(actionable[actionable['Signal'] == 'BUY'])
        fade_count = len(actionable[actionable['Signal'] == 'FADE'])
        total_bet_size = actionable['bet_size_value'].sum()
        st.markdown(f"**BUY:** {buy_count} | **FADE:** {fade_count} | **Total Wagered:** ${total_bet_size:.0f}")

with tab2:
    st.subheader(f"All Games ({len(all_games)} games)")

    # Sort by cover probability deviation from 50%
    all_games = all_games.sort_values('cover_prob_value', key=lambda x: abs(x - 0.5), ascending=False)

    display_cols = ['Game', 'Signal', 'Win Prob', 'Vegas', 'Model', 'Edge', 'Variance', 'Kelly Size', 'Context']
    styled_df = all_games[display_cols].style.applymap(
        style_signal, subset=['Signal']
    ).applymap(
        style_win_prob, subset=['Win Prob']
    ).applymap(
        style_variance, subset=['Variance']
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

with tab3:
    st.subheader(f"High Variance Games ({len(high_var_games)} games)")
    st.warning("These games have high model disagreement (Variance > 7.0). Models are confused - PASS recommended.")

    if high_var_games.empty:
        st.info("No high variance games this week. Models are in agreement!")
    else:
        # Sort by variance (highest first)
        high_var_games = high_var_games.sort_values('variance_value', ascending=False)

        display_cols = ['Game', 'Signal', 'Win Prob', 'Vegas', 'Model', 'Edge', 'Variance', 'Context']
        styled_df = high_var_games[display_cols].style.applymap(
            style_variance, subset=['Variance']
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

# Injury Links
with st.expander("Injury Reports"):
    for _, row in df_predictions.iterrows():
        st.markdown(f"- [{row['Home']}]({row['Injury']}) | [{row['Away']}]({get_injury_url(row['Away'])})")

# Footer
st.divider()
st.caption("""
**Model:** Stacking (Gradient + Forest + Linear) with RidgeCV Meta-Learner
**Confusion Filter:** Variance > 7.0 = PASS (models disagree)
**Monte Carlo:** 10,000 simulations | Std Dev: 14 pts
**Signal:** BUY (>55%) | FADE (<45%) | PASS (45-55% or High Variance)
**Kelly:** 0.25x Fractional | Bankroll: $1,000
""")
