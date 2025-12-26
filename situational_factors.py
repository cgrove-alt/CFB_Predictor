"""
Situational Factors Analysis for CFB Betting.

Features:
- Rest Days calculation
- Travel Disadvantage (West Coast traveling East for early games)
- Sandwich Spot detection (weak opponent before big game)
- Sharp Action signals (Reverse Line Movement)
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# TIME ZONE MAPPING
# ============================================================
TEAM_TIMEZONES = {
    # Eastern Time (ET)
    'Alabama': 'CT', 'Auburn': 'CT', 'LSU': 'CT', 'Ole Miss': 'CT', 'Mississippi State': 'CT',
    'Georgia': 'ET', 'Florida': 'ET', 'South Carolina': 'ET', 'Tennessee': 'ET', 'Kentucky': 'ET',
    'Vanderbilt': 'CT', 'Texas A&M': 'CT', 'Arkansas': 'CT', 'Missouri': 'CT',
    'Ohio State': 'ET', 'Michigan': 'ET', 'Penn State': 'ET', 'Michigan State': 'ET',
    'Indiana': 'ET', 'Purdue': 'ET', 'Rutgers': 'ET', 'Maryland': 'ET',
    'Miami': 'ET', 'Florida State': 'ET', 'Clemson': 'ET', 'NC State': 'ET',
    'North Carolina': 'ET', 'Duke': 'ET', 'Wake Forest': 'ET', 'Virginia': 'ET',
    'Virginia Tech': 'ET', 'Boston College': 'ET', 'Syracuse': 'ET', 'Pittsburgh': 'ET',
    'Louisville': 'ET', 'Notre Dame': 'ET', 'Georgia Tech': 'ET',

    # Central Time (CT)
    'Texas': 'CT', 'Oklahoma': 'CT', 'Texas Tech': 'CT', 'Baylor': 'CT', 'TCU': 'CT',
    'Kansas': 'CT', 'Kansas State': 'CT', 'Iowa State': 'CT', 'Oklahoma State': 'CT',
    'Wisconsin': 'CT', 'Minnesota': 'CT', 'Iowa': 'CT', 'Nebraska': 'CT',
    'Northwestern': 'CT', 'Illinois': 'CT', 'SMU': 'CT', 'Houston': 'CT',
    'Tulane': 'CT', 'Memphis': 'CT', 'UCF': 'ET', 'Cincinnati': 'ET',

    # Mountain Time (MT)
    'Colorado': 'MT', 'Utah': 'MT', 'BYU': 'MT', 'Colorado State': 'MT',
    'Air Force': 'MT', 'Wyoming': 'MT', 'New Mexico': 'MT', 'Utah State': 'MT',
    'Boise State': 'MT', 'Arizona': 'MT', 'Arizona State': 'MT',

    # Pacific Time (PT)
    'USC': 'PT', 'UCLA': 'PT', 'Stanford': 'PT', 'California': 'PT',
    'Oregon': 'PT', 'Oregon State': 'PT', 'Washington': 'PT', 'Washington State': 'PT',
    'San Diego State': 'PT', 'Fresno State': 'PT', 'Nevada': 'PT', 'UNLV': 'PT',
    "San José State": 'PT', 'San Jose State': 'PT', "Hawai'i": 'HT', 'Hawaii': 'HT',
}

TIMEZONE_OFFSET = {'ET': 0, 'CT': 1, 'MT': 2, 'PT': 3, 'HT': 5}

# ============================================================
# SITUATIONAL ANALYSIS FUNCTIONS
# ============================================================

def get_timezone_diff(away_team, home_team):
    """Calculate timezone difference (positive = traveling East)."""
    away_tz = TEAM_TIMEZONES.get(away_team, 'CT')
    home_tz = TEAM_TIMEZONES.get(home_team, 'CT')

    away_offset = TIMEZONE_OFFSET.get(away_tz, 1)
    home_offset = TIMEZONE_OFFSET.get(home_tz, 1)

    # Positive = traveling East (earlier body clock)
    return away_offset - home_offset

def is_travel_disadvantage(away_team, home_team, threshold=2):
    """
    Flag if away team travels 2+ time zones East.
    This is a significant disadvantage for early kickoffs.
    """
    tz_diff = get_timezone_diff(away_team, home_team)
    return tz_diff >= threshold

def is_sandwich_spot(team, current_opponent_elo, next_opponent_elo,
                     weak_threshold=1400, strong_threshold=1700):
    """
    Detect 'Sandwich Spot' - team may overlook weak opponent before big game.

    Conditions:
    - Current opponent is weak (Elo < 1400)
    - Next opponent is strong (Elo > 1700)
    """
    if current_opponent_elo is None or next_opponent_elo is None:
        return False

    return (current_opponent_elo < weak_threshold) and (next_opponent_elo > strong_threshold)


# ============================================================
# V21: LETDOWN, REVENGE, AND TRAP SPOT DETECTION
# ============================================================

def is_letdown_spot(team_elo, prev_opponent_elo, prev_result,
                    current_opponent_elo, elite_threshold=1700, weak_threshold=1500):
    """
    Detect 'Letdown Spot' - team likely to underperform after emotional win.

    Conditions:
    - Previous game was a WIN against ranked/elite opponent (Elo > 1700)
    - Current opponent is unranked/weaker (Elo < 1500)

    Research: Favorites in letdown spots underperform by 2-3 points.

    Args:
        team_elo: Current team's Elo rating
        prev_opponent_elo: Previous opponent's Elo rating
        prev_result: 'win' or 'loss' in previous game
        current_opponent_elo: Current opponent's Elo rating
        elite_threshold: Elo threshold for "elite" opponent (default 1700)
        weak_threshold: Elo threshold for "weak" opponent (default 1500)

    Returns:
        bool: True if letdown spot detected
    """
    if prev_opponent_elo is None or current_opponent_elo is None or prev_result is None:
        return False

    # Team won previous game against elite opponent
    prev_was_big_win = (prev_result.lower() == 'win' and prev_opponent_elo >= elite_threshold)

    # Current opponent is weaker
    current_is_weak = current_opponent_elo < weak_threshold

    # Team is good (potential favorite)
    team_is_good = team_elo >= 1600

    return prev_was_big_win and current_is_weak and team_is_good


def is_revenge_spot(team, opponent, team_lost_to_opponent_last_meeting):
    """
    Detect 'Revenge Spot' - team has extra motivation after previous loss.

    Research: Teams in revenge spots cover at ~53-54% rate.

    Args:
        team: Team name
        opponent: Opponent name
        team_lost_to_opponent_last_meeting: True if team lost previous matchup

    Returns:
        bool: True if revenge spot detected
    """
    return team_lost_to_opponent_last_meeting is True


def is_trap_game(team_elo, team_ranking, current_opponent_elo, next_opponent_elo,
                 ranked_threshold=25, weak_opp_threshold=1400, strong_next_threshold=1650):
    """
    Detect 'Trap Game' - ranked team faces weak opponent before big game.

    More strict than sandwich spot - focuses on ranked teams.

    Conditions:
    - Team is ranked (top 25)
    - Current opponent is weak (Elo < 1400)
    - Next opponent is strong/ranked (Elo > 1650)

    Research: Ranked teams in trap games underperform by 2-4 points.

    Args:
        team_elo: Current team's Elo rating
        team_ranking: Current AP/CFP ranking (1-25, or None if unranked)
        current_opponent_elo: Current opponent's Elo rating
        next_opponent_elo: Next week's opponent Elo rating
        ranked_threshold: Ranking threshold (default top 25)
        weak_opp_threshold: Elo for "weak" opponent
        strong_next_threshold: Elo for "strong" next opponent

    Returns:
        bool: True if trap game detected
    """
    if current_opponent_elo is None or next_opponent_elo is None:
        return False

    # Team is ranked
    team_is_ranked = team_ranking is not None and 1 <= team_ranking <= ranked_threshold

    # Or team is elite by Elo
    team_is_elite = team_elo >= 1700

    # Current opponent is weak
    current_is_weak = current_opponent_elo < weak_opp_threshold

    # Next opponent is strong
    next_is_strong = next_opponent_elo >= strong_next_threshold

    return (team_is_ranked or team_is_elite) and current_is_weak and next_is_strong


def calculate_situational_adjustment(letdown=False, revenge=False, trap=False,
                                     travel_disadvantage=False, rest_edge=0,
                                     home_team=True):
    """
    Calculate point adjustment based on situational factors.

    Returns adjustment to add to spread prediction (positive favors home).

    Research-based adjustments:
    - Letdown: -2.5 to -3 points for the affected team
    - Revenge: +1.5 to +2 points for the revengeful team
    - Trap: -2 to -3 points for the trapped team
    - Travel disadvantage: -1 to -2 points for traveling team
    - Rest edge: ~0.3-0.5 points per day
    """
    adjustment = 0.0

    if letdown:
        # Team in letdown likely to underperform
        if home_team:
            adjustment -= 2.5  # Home team in letdown = reduce home advantage
        else:
            adjustment += 2.5  # Away team in letdown = increase home advantage

    if revenge:
        # Team with revenge motivation likely to overperform
        if home_team:
            adjustment += 1.5  # Home team with revenge
        else:
            adjustment -= 1.5  # Away team with revenge

    if trap:
        # Ranked/elite team likely to overlook opponent
        if home_team:
            adjustment -= 2.0  # Home team in trap
        else:
            adjustment += 2.0  # Away team in trap

    if travel_disadvantage:
        # West coast traveling east for early game
        adjustment += 1.5  # Benefits home team

    # Rest edge: ~0.4 points per day
    adjustment += rest_edge * 0.4

    return adjustment

def calculate_rest_edge(home_rest_days, away_rest_days):
    """
    Calculate rest advantage.
    Positive = home team has more rest.
    """
    return home_rest_days - away_rest_days

def detect_reverse_line_move(opening_spread, current_spread, public_pct_home=None):
    """
    Detect Reverse Line Movement (Sharp Action Signal).

    Classic RLM:
    - Public is heavy on one side (>70%)
    - Line moves AGAINST the public side
    - This indicates sharp money on the other side

    Without public %, we detect significant line movement direction.
    """
    if opening_spread is None or current_spread is None:
        return None, ""

    line_move = current_spread - opening_spread

    # Significant move = 1.5+ points
    if abs(line_move) < 1.5:
        return None, ""

    # Line moved toward home (more negative = home more favored)
    if line_move < -1.5:
        signal = "Sharp on HOME"
        if public_pct_home and public_pct_home < 40:
            # Public was on away, line moved to home = classic RLM
            signal = "RLM: SHARP HOME 📉"
        return "home", signal

    # Line moved toward away (more positive = away more favored)
    if line_move > 1.5:
        signal = "Sharp on AWAY"
        if public_pct_home and public_pct_home > 60:
            # Public was on home, line moved to away = classic RLM
            signal = "RLM: SHARP AWAY 📉"
        return "away", signal

    return None, ""

def get_situation_alert(home_team, away_team, home_rest, away_rest,
                        home_lookahead, away_lookahead, west_coast_early,
                        current_opp_elo_home=None, next_opp_elo_home=None,
                        current_opp_elo_away=None, next_opp_elo_away=None):
    """
    Generate situational alert string for a game.
    """
    alerts = []

    # Rest advantage
    rest_edge = calculate_rest_edge(home_rest, away_rest)
    if rest_edge >= 7:
        alerts.append(f"Home Rested (+{rest_edge}d)")
    elif rest_edge <= -7:
        alerts.append(f"Away Rested ({rest_edge}d)")

    # Travel disadvantage
    if is_travel_disadvantage(away_team, home_team):
        alerts.append("Travel Trap 🛫")

    # West Coast Early
    if west_coast_early:
        alerts.append("West Coast Early 🌅")

    # Sandwich spots
    if home_lookahead and current_opp_elo_away and next_opp_elo_home:
        if is_sandwich_spot(home_team, current_opp_elo_away, next_opp_elo_home):
            alerts.append("Sandwich Game 🥪")
    elif home_lookahead:
        alerts.append("Home Lookahead 👀")

    if away_lookahead:
        alerts.append("Away Lookahead 👀")

    return " | ".join(alerts) if alerts else "—"

# ============================================================
# PUBLIC/SHARP DATA (Placeholder - requires Odds API subscription)
# ============================================================
# Note: Real public betting % requires The Odds API or similar service
# This is a placeholder structure for when data is available

def get_public_betting_pct(home_team, away_team):
    """
    Get public betting percentages.
    Requires The Odds API or similar service.

    Returns: (public_pct_home, public_pct_away)
    """
    # Placeholder - in production, would fetch from:
    # - The Odds API (https://the-odds-api.com/)
    # - Action Network
    # - VegasInsider
    return None, None

def analyze_sharp_action(opening_spread, current_spread, public_pct_home=None):
    """
    Analyze for sharp action signals.

    Returns: dict with signal info
    """
    result = {
        'signal': '',
        'side': None,
        'confidence': 'low'
    }

    if opening_spread is None or current_spread is None:
        return result

    line_move = current_spread - opening_spread

    # Significant move
    if abs(line_move) >= 1.5:
        if line_move < 0:
            result['side'] = 'home'
            result['signal'] = f"Steam HOME ({line_move:+.1f})"
        else:
            result['side'] = 'away'
            result['signal'] = f"Steam AWAY ({line_move:+.1f})"
        result['confidence'] = 'medium'

    # Very significant move
    if abs(line_move) >= 2.5:
        result['confidence'] = 'high'
        if line_move < 0:
            result['signal'] = f"SHARP HOME 📉 ({line_move:+.1f})"
        else:
            result['signal'] = f"SHARP AWAY 📉 ({line_move:+.1f})"

    # Classic Reverse Line Movement (if public data available)
    if public_pct_home is not None:
        if public_pct_home > 70 and line_move > 1.5:
            result['signal'] = "RLM: FADE PUBLIC 📉"
            result['side'] = 'away'
            result['confidence'] = 'high'
        elif public_pct_home < 30 and line_move < -1.5:
            result['signal'] = "RLM: FADE PUBLIC 📉"
            result['side'] = 'home'
            result['confidence'] = 'high'

    return result

# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_game_situation(game_data):
    """
    Full situational analysis for a game.

    Input: dict with game info
    Returns: dict with all situational factors
    """
    analysis = {
        'rest_edge': 0,
        'travel_disadvantage': False,
        'west_coast_early': False,
        'sandwich_spot': False,
        'lookahead': None,
        'situation_alert': '',
        'sharp_signal': '',
        'sharp_side': None
    }

    # Rest edge
    home_rest = game_data.get('home_rest_days', 7)
    away_rest = game_data.get('away_rest_days', 7)
    analysis['rest_edge'] = calculate_rest_edge(home_rest, away_rest)

    # Travel
    home_team = game_data.get('home_team', '')
    away_team = game_data.get('away_team', '')
    analysis['travel_disadvantage'] = is_travel_disadvantage(away_team, home_team)

    # West Coast Early
    analysis['west_coast_early'] = game_data.get('west_coast_early', 0) == 1

    # Lookahead
    if game_data.get('home_lookahead'):
        analysis['lookahead'] = 'home'
    elif game_data.get('away_lookahead'):
        analysis['lookahead'] = 'away'

    # Build situation alert
    analysis['situation_alert'] = get_situation_alert(
        home_team, away_team, home_rest, away_rest,
        game_data.get('home_lookahead', 0),
        game_data.get('away_lookahead', 0),
        game_data.get('west_coast_early', 0)
    )

    # Sharp action
    sharp_result = analyze_sharp_action(
        game_data.get('opening_spread'),
        game_data.get('current_spread'),
        game_data.get('public_pct_home')
    )
    analysis['sharp_signal'] = sharp_result['signal']
    analysis['sharp_side'] = sharp_result['side']

    return analysis

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("SITUATIONAL FACTORS ANALYSIS TEST")
    print("="*60)

    # Test timezone differences
    print("\nTimezone Differences:")
    test_matchups = [
        ('USC', 'Ohio State'),
        ('Oregon', 'Georgia'),
        ('Hawaii', 'Alabama'),
        ('Michigan', 'Penn State'),
    ]

    for away, home in test_matchups:
        tz_diff = get_timezone_diff(away, home)
        travel_flag = is_travel_disadvantage(away, home)
        print(f"  {away} @ {home}: TZ diff = {tz_diff}, Travel Trap = {travel_flag}")

    # Test sandwich spot
    print("\nSandwich Spot Test:")
    print(f"  Weak opp (1350) + Strong next (1750): {is_sandwich_spot('Team', 1350, 1750)}")
    print(f"  Weak opp (1350) + Weak next (1400): {is_sandwich_spot('Team', 1350, 1400)}")
    print(f"  Strong opp (1600) + Strong next (1750): {is_sandwich_spot('Team', 1600, 1750)}")

    # Test sharp action
    print("\nSharp Action Test:")
    test_lines = [
        (-7.0, -8.5, None),  # Steam home
        (-7.0, -5.0, None),  # Steam away
        (-7.0, -7.5, None),  # Minor move
        (-7.0, -5.0, 75),    # RLM (public on home, line to away)
    ]

    for open_line, curr_line, public in test_lines:
        result = analyze_sharp_action(open_line, curr_line, public)
        print(f"  Open: {open_line}, Current: {curr_line}, Public: {public}")
        print(f"    -> {result}")

    print("\n" + "="*60)
    print("ANALYSIS MODULE READY")
    print("="*60)
