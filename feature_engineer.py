"""
Feature engineering: Calculate rolling stats, rest days, situational flags.
"""

import pandas as pd
import numpy as np

# ============================================================
# WEST COAST TEAMS (Pacific/Mountain Time)
# ============================================================
WEST_COAST_TEAMS = [
    # Pac-12 / Big Ten West Coast
    'USC', 'UCLA', 'Oregon', 'Oregon State', 'Washington', 'Washington State',
    'California', 'Stanford', 'Arizona', 'Arizona State', 'Colorado', 'Utah',
    # Mountain West
    'Boise State', 'San Diego State', 'Fresno State', 'Nevada', 'UNLV',
    'San José State', 'Hawai\'i', "Hawai'i", 'Hawaii',
    'Colorado State', 'Wyoming', 'Air Force', 'New Mexico', 'Utah State',
    # Other West Coast
    'BYU', 'San Jose State'
]

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv('cfb_data.csv')

# Sort by season and week
df = df.sort_values(['season', 'week', 'id']).reset_index(drop=True)
print(f"Total games: {len(df)}")

# ============================================================
# BUILD TEAM GAME HISTORY
# ============================================================
def build_team_game_history(df):
    """Build a dictionary mapping team -> list of game records"""
    team_games = {}

    for _, row in df.iterrows():
        season = row['season']
        week = row['week']

        # Record home team's game
        home_team = row['home_team']
        if home_team not in team_games:
            team_games[home_team] = []
        team_games[home_team].append({
            'season': season,
            'week': week,
            'points_scored': row['home_points'],
            'points_allowed': row['away_points'],
            'opponent': row['away_team'],
            'opponent_elo': row.get('away_pregame_elo', 1500),
            'is_home': True
        })

        # Record away team's game
        away_team = row['away_team']
        if away_team not in team_games:
            team_games[away_team] = []
        team_games[away_team].append({
            'season': season,
            'week': week,
            'points_scored': row['away_points'],
            'points_allowed': row['home_points'],
            'opponent': row['home_team'],
            'opponent_elo': row.get('home_pregame_elo', 1500),
            'is_home': False
        })

    return team_games

print("Building team game histories...")
team_games = build_team_game_history(df)

# ============================================================
# ROLLING STATS FUNCTIONS
# ============================================================
def get_last_n_games(team, season, week, team_games, n=5):
    """Get team's last n games before given season/week"""
    if team not in team_games:
        return []

    previous_games = []
    for game in team_games[team]:
        if game['season'] < season or (game['season'] == season and game['week'] < week):
            previous_games.append(game)

    previous_games.sort(key=lambda x: (x['season'], x['week']), reverse=True)
    return previous_games[:n]

def calculate_rolling_stats(team, season, week, team_games, n=5, weighted=False):
    """Calculate rolling average stats for a team's last n games"""
    last_games = get_last_n_games(team, season, week, team_games, n)

    if len(last_games) == 0:
        return np.nan, np.nan

    if weighted and len(last_games) >= 2:
        # V18 IMPROVEMENT: Exponential recency weighting
        # More recent games matter more
        # Weights: [0.35, 0.25, 0.20, 0.12, 0.08] for 5 games
        base_weights = [0.35, 0.25, 0.20, 0.12, 0.08]
        weights = base_weights[:len(last_games)]
        # Normalize weights to sum to 1
        weights = [w / sum(weights) for w in weights]

        avg_scored = sum(g['points_scored'] * w for g, w in zip(last_games, weights))
        avg_allowed = sum(g['points_allowed'] * w for g, w in zip(last_games, weights))
    else:
        # Simple average
        avg_scored = np.mean([g['points_scored'] for g in last_games])
        avg_allowed = np.mean([g['points_allowed'] for g in last_games])

    return avg_scored, avg_allowed


def calculate_weighted_rolling_stats(team, season, week, team_games, n=5):
    """V18: Calculate recency-weighted rolling stats"""
    return calculate_rolling_stats(team, season, week, team_games, n, weighted=True)

# ============================================================
# REST DAYS CALCULATION
# ============================================================
def calculate_rest_days(team, season, week, team_games):
    """
    Calculate days since last game.
    Assumes ~7 days between weeks, with bye weeks = 14+ days.
    """
    if team not in team_games:
        return 7  # Default to standard rest

    # Find most recent game before this one
    previous_games = [
        g for g in team_games[team]
        if g['season'] < season or (g['season'] == season and g['week'] < week)
    ]

    if len(previous_games) == 0:
        return 14  # Season opener, assume extra rest

    # Sort to get most recent
    previous_games.sort(key=lambda x: (x['season'], x['week']), reverse=True)
    last_game = previous_games[0]

    # Calculate week difference
    if last_game['season'] == season:
        week_diff = week - last_game['week']
    else:
        # Previous season - assume long rest (bowl to opener)
        week_diff = 30  # ~30 weeks between seasons

    # Convert to approximate days (7 days per week)
    rest_days = week_diff * 7

    return rest_days

# ============================================================
# WEST COAST EARLY GAME FLAG
# ============================================================
def is_west_coast_early(away_team, home_team):
    """
    Flag games where West Coast team travels East for early kickoff.
    This is a known disadvantage (body clock = 9am for noon ET kickoff).
    """
    # Check if away team is West Coast
    away_is_west = any(wc.lower() in away_team.lower() for wc in WEST_COAST_TEAMS)

    # Check if home team is NOT West Coast (i.e., Eastern/Central time)
    home_is_east = not any(wc.lower() in home_team.lower() for wc in WEST_COAST_TEAMS)

    # West Coast team traveling East = disadvantage
    return 1 if (away_is_west and home_is_east) else 0

# ============================================================
# LOOKAHEAD SPOT FLAG
# ============================================================
def get_next_opponent_elo(team, season, week, team_games):
    """Get the Elo of team's next opponent (for lookahead spot detection)"""
    if team not in team_games:
        return None

    # Find games after this week in same season
    future_games = [
        g for g in team_games[team]
        if g['season'] == season and g['week'] > week
    ]

    if len(future_games) == 0:
        return None

    # Sort to get next game
    future_games.sort(key=lambda x: x['week'])
    next_game = future_games[0]

    return next_game.get('opponent_elo', 1500)

def is_lookahead_spot(team, season, week, team_games, elo_threshold=1700):
    """
    Flag if team has a big game next week (opponent Elo > threshold).
    Teams may overlook current opponent = betting opportunity.
    """
    next_elo = get_next_opponent_elo(team, season, week, team_games)

    if next_elo is None:
        return 0

    return 1 if next_elo > elo_threshold else 0

# ============================================================
# CALCULATE ALL FEATURES
# ============================================================
print("Calculating features (this may take a moment)...")

# Rolling stats (simple average)
home_last5_score = []
home_last5_defense = []
away_last5_score = []
away_last5_defense = []

# V18: Weighted rolling stats (recency weighted)
home_weighted_score = []
home_weighted_defense = []
away_weighted_score = []
away_weighted_defense = []

# Rest days
home_rest_days = []
away_rest_days = []

# Situational flags
west_coast_early = []
home_lookahead = []
away_lookahead = []

for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(f"  Processing game {idx}/{len(df)}...")

    season = row['season']
    week = row['week']
    home_team = row['home_team']
    away_team = row['away_team']

    # Rolling stats (simple average)
    h_score, h_defense = calculate_rolling_stats(home_team, season, week, team_games)
    a_score, a_defense = calculate_rolling_stats(away_team, season, week, team_games)
    home_last5_score.append(h_score)
    home_last5_defense.append(h_defense)
    away_last5_score.append(a_score)
    away_last5_defense.append(a_defense)

    # V18: Weighted rolling stats (recency weighted)
    h_w_score, h_w_defense = calculate_weighted_rolling_stats(home_team, season, week, team_games)
    a_w_score, a_w_defense = calculate_weighted_rolling_stats(away_team, season, week, team_games)
    home_weighted_score.append(h_w_score)
    home_weighted_defense.append(h_w_defense)
    away_weighted_score.append(a_w_score)
    away_weighted_defense.append(a_w_defense)

    # Rest days
    h_rest = calculate_rest_days(home_team, season, week, team_games)
    a_rest = calculate_rest_days(away_team, season, week, team_games)
    home_rest_days.append(h_rest)
    away_rest_days.append(a_rest)

    # West Coast Early flag (away team disadvantage)
    wc_early = is_west_coast_early(away_team, home_team)
    west_coast_early.append(wc_early)

    # Lookahead spots
    h_lookahead = is_lookahead_spot(home_team, season, week, team_games)
    a_lookahead = is_lookahead_spot(away_team, season, week, team_games)
    home_lookahead.append(h_lookahead)
    away_lookahead.append(a_lookahead)

# ============================================================
# ADD COLUMNS TO DATAFRAME
# ============================================================
print("\nAdding new columns...")

# Rolling stats (simple average)
df['home_last5_score_avg'] = home_last5_score
df['home_last5_defense_avg'] = home_last5_defense
df['away_last5_score_avg'] = away_last5_score
df['away_last5_defense_avg'] = away_last5_defense

# V18: Weighted rolling stats (recency weighted - more recent games matter more)
df['home_weighted_score'] = home_weighted_score
df['home_weighted_defense'] = home_weighted_defense
df['away_weighted_score'] = away_weighted_score
df['away_weighted_defense'] = away_weighted_defense

# V18: Scoring momentum (weighted vs simple - shows if team is trending up/down)
df['home_scoring_momentum'] = df['home_weighted_score'] - df['home_last5_score_avg']
df['away_scoring_momentum'] = df['away_weighted_score'] - df['away_last5_score_avg']
df['scoring_momentum_diff'] = df['home_scoring_momentum'] - df['away_scoring_momentum']

# Rest days
df['home_rest_days'] = home_rest_days
df['away_rest_days'] = away_rest_days
df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']  # Positive = home rested more

# Situational flags
df['west_coast_early'] = west_coast_early  # 1 = away team at disadvantage
df['home_lookahead'] = home_lookahead  # 1 = home may overlook this game
df['away_lookahead'] = away_lookahead  # 1 = away may overlook this game

# Drop any encoded columns
cols_to_drop = [col for col in df.columns if 'encoded' in col.lower()]
if cols_to_drop:
    print(f"Dropping encoded columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

# ============================================================
# SUMMARY STATS
# ============================================================
print("\n" + "="*60)
print("FEATURE SUMMARY")
print("="*60)

print(f"\nRolling Stats:")
print(f"  Games with home rolling stats: {df['home_last5_score_avg'].notna().sum()}")
print(f"  Games missing home rolling stats: {df['home_last5_score_avg'].isna().sum()}")

print(f"\nRest Days:")
print(f"  Average home rest: {df['home_rest_days'].mean():.1f} days")
print(f"  Average away rest: {df['away_rest_days'].mean():.1f} days")
print(f"  Games with rest advantage > 7 days: {(df['rest_advantage'].abs() > 7).sum()}")

print(f"\nWest Coast Early Games:")
print(f"  Total flagged: {df['west_coast_early'].sum()} ({df['west_coast_early'].mean()*100:.1f}%)")

print(f"\nLookahead Spots (next opponent Elo > 1700):")
print(f"  Home team lookahead: {df['home_lookahead'].sum()} ({df['home_lookahead'].mean()*100:.1f}%)")
print(f"  Away team lookahead: {df['away_lookahead'].sum()} ({df['away_lookahead'].mean()*100:.1f}%)")

# ============================================================
# SAVE
# ============================================================
df.to_csv('cfb_data_smart.csv', index=False)
print(f"\nSaved {len(df)} games to 'cfb_data_smart.csv'")
print(f"Total columns: {len(df.columns)}")

# Print sample
print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)
display_cols = ['season', 'week', 'home_team', 'away_team',
                'home_rest_days', 'away_rest_days', 'rest_advantage',
                'west_coast_early', 'home_lookahead', 'away_lookahead']
print(df[display_cols].head(10).to_string())

# Show some interesting examples
print("\n" + "="*60)
print("WEST COAST EARLY EXAMPLES")
print("="*60)
wc_games = df[df['west_coast_early'] == 1][['season', 'week', 'away_team', 'home_team', 'Margin']].head(5)
print(wc_games.to_string())

print("\n" + "="*60)
print("LOOKAHEAD SPOT EXAMPLES (Home team)")
print("="*60)
lookahead_games = df[df['home_lookahead'] == 1][['season', 'week', 'home_team', 'away_team', 'Margin']].head(5)
print(lookahead_games.to_string())
