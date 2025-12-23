"""
Unified Data Refresh Script for Sharp Sports Predictor.

This script fetches ALL required data from the CFBD API using direct HTTP requests
(bypassing the broken cfbd library) and generates a fresh cfb_data_smart.csv.

Usage:
    python3 refresh_all_data.py

API Endpoints Used:
    - /games - Game results and Elo ratings
    - /ppa/games - Predicted Points Added (PPA) stats
    - /stats/season/advanced - Advanced team stats
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import CFBD_API_KEY

# =============================================================================
# CONFIGURATION
# =============================================================================
CFBD_BASE_URL = "https://api.collegefootballdata.com"
YEARS = [2022, 2023, 2024, 2025]
BLOWOUT_THRESHOLD = 28

def get_headers():
    """Get authorization headers for CFBD API."""
    return {'Authorization': f'Bearer {CFBD_API_KEY}'}

# =============================================================================
# API FETCH FUNCTIONS
# =============================================================================
def fetch_games(year):
    """Fetch all games for a year."""
    url = f"{CFBD_BASE_URL}/games?year={year}"
    resp = requests.get(url, headers=get_headers())
    if resp.status_code == 200:
        return resp.json()
    print(f"  Error fetching games {year}: {resp.status_code}")
    return []

def fetch_ppa_games(year, exclude_garbage_time=True):
    """Fetch PPA (Predicted Points Added) by game."""
    url = f"{CFBD_BASE_URL}/ppa/games?year={year}&excludeGarbageTime={str(exclude_garbage_time).lower()}"
    resp = requests.get(url, headers=get_headers())
    if resp.status_code == 200:
        return resp.json()
    print(f"  Error fetching PPA {year}: {resp.status_code}")
    return []

def fetch_advanced_stats(year):
    """Fetch advanced season stats."""
    url = f"{CFBD_BASE_URL}/stats/season/advanced?year={year}"
    resp = requests.get(url, headers=get_headers())
    if resp.status_code == 200:
        return resp.json()
    return []

def fetch_team_records(year):
    """Fetch team records for calculating HFA."""
    url = f"{CFBD_BASE_URL}/records?year={year}"
    resp = requests.get(url, headers=get_headers())
    if resp.status_code == 200:
        return resp.json()
    return []

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================
def process_games(all_games):
    """Process raw games into DataFrame."""
    records = []
    for game in all_games:
        # Skip games without scores (not yet played)
        if game.get('homePoints') is None or game.get('awayPoints') is None:
            continue

        records.append({
            'id': game.get('id'),
            'season': game.get('season'),
            'week': game.get('week'),
            'home_team': game.get('homeTeam'),
            'away_team': game.get('awayTeam'),
            'home_points': game.get('homePoints'),
            'away_points': game.get('awayPoints'),
            'home_pregame_elo': game.get('homePregameElo'),
            'away_pregame_elo': game.get('awayPregameElo'),
            'start_date': game.get('startDate'),
        })

    df = pd.DataFrame(records)
    df['Margin'] = df['home_points'] - df['away_points']
    df['score_diff'] = abs(df['Margin'])
    df['is_blowout'] = df['score_diff'] > BLOWOUT_THRESHOLD
    return df

def calculate_rest_days(df):
    """Calculate rest days for each team."""
    df = df.sort_values(['season', 'week'])

    # Track last game date for each team
    team_last_game = {}
    home_rest = []
    away_rest = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        season = row['season']
        week = row['week']
        game_key = (season, week)

        # Calculate rest days (default 7)
        if home in team_last_game and team_last_game[home][0] == season:
            home_rest.append((week - team_last_game[home][1]) * 7)
        else:
            home_rest.append(7)

        if away in team_last_game and team_last_game[away][0] == season:
            away_rest.append((week - team_last_game[away][1]) * 7)
        else:
            away_rest.append(7)

        # Update last game
        team_last_game[home] = (season, week)
        team_last_game[away] = (season, week)

    df['home_rest_days'] = home_rest
    df['away_rest_days'] = away_rest
    df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
    df['rest_diff'] = df['rest_advantage']
    return df

def calculate_rolling_stats(df, window=5):
    """Calculate rolling averages for each team."""
    df = df.sort_values(['season', 'week'])

    # Build team game history
    team_scores = {}  # team -> list of (season, week, points_for, points_against)

    home_last5_score = []
    home_last5_defense = []
    away_last5_score = []
    away_last5_defense = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']

        # Calculate averages from history
        if home in team_scores and len(team_scores[home]) > 0:
            recent = team_scores[home][-window:]
            home_last5_score.append(np.mean([g[2] for g in recent]))
            home_last5_defense.append(np.mean([g[3] for g in recent]))
        else:
            home_last5_score.append(28.0)  # Default
            home_last5_defense.append(24.0)

        if away in team_scores and len(team_scores[away]) > 0:
            recent = team_scores[away][-window:]
            away_last5_score.append(np.mean([g[2] for g in recent]))
            away_last5_defense.append(np.mean([g[3] for g in recent]))
        else:
            away_last5_score.append(28.0)
            away_last5_defense.append(24.0)

        # Update history (after using it for prediction)
        if home not in team_scores:
            team_scores[home] = []
        if away not in team_scores:
            team_scores[away] = []

        team_scores[home].append((row['season'], row['week'], row['home_points'], row['away_points']))
        team_scores[away].append((row['season'], row['week'], row['away_points'], row['home_points']))

    df['home_last5_score_avg'] = home_last5_score
    df['home_last5_defense_avg'] = home_last5_defense
    df['away_last5_score_avg'] = away_last5_score
    df['away_last5_defense_avg'] = away_last5_defense
    return df

def process_ppa_data(all_ppa, df):
    """Process PPA data and merge with games."""
    # Build PPA lookup by game_id and team
    ppa_lookup = {}
    for ppa in all_ppa:
        game_id = ppa.get('gameId')
        team = ppa.get('team')
        if game_id and team:
            key = (game_id, team)
            offense = ppa.get('offense', {}) or {}
            defense = ppa.get('defense', {}) or {}
            ppa_lookup[key] = {
                'off_ppa': offense.get('overall'),
                'def_ppa': defense.get('overall'),
                'pass_ppa': offense.get('passing'),
                'rush_ppa': offense.get('rushing'),
                'success': offense.get('successRate'),
            }

    # Add PPA columns to df
    home_off_ppa = []
    home_def_ppa = []
    home_pass_ppa = []
    home_rush_ppa = []
    home_success = []
    away_off_ppa = []
    away_def_ppa = []
    away_pass_ppa = []
    away_rush_ppa = []
    away_success = []

    for _, row in df.iterrows():
        game_id = row['id']
        home = row['home_team']
        away = row['away_team']

        home_ppa = ppa_lookup.get((game_id, home), {})
        away_ppa = ppa_lookup.get((game_id, away), {})

        home_off_ppa.append(home_ppa.get('off_ppa'))
        home_def_ppa.append(home_ppa.get('def_ppa'))
        home_pass_ppa.append(home_ppa.get('pass_ppa'))
        home_rush_ppa.append(home_ppa.get('rush_ppa'))
        home_success.append(home_ppa.get('success'))

        away_off_ppa.append(away_ppa.get('off_ppa'))
        away_def_ppa.append(away_ppa.get('def_ppa'))
        away_pass_ppa.append(away_ppa.get('pass_ppa'))
        away_rush_ppa.append(away_ppa.get('rush_ppa'))
        away_success.append(away_ppa.get('success'))

    df['home_comp_off_ppa'] = home_off_ppa
    df['home_comp_def_ppa'] = home_def_ppa
    df['home_comp_pass_ppa'] = home_pass_ppa
    df['home_comp_rush_ppa'] = home_rush_ppa
    df['home_comp_success'] = home_success

    df['away_comp_off_ppa'] = away_off_ppa
    df['away_comp_def_ppa'] = away_def_ppa
    df['away_comp_pass_ppa'] = away_pass_ppa
    df['away_comp_rush_ppa'] = away_rush_ppa
    df['away_comp_success'] = away_success

    return df

def calculate_hfa(df):
    """Calculate Home Field Advantage for each team."""
    # Calculate historical HFA from game results
    team_home_margin = {}

    for _, row in df.iterrows():
        home = row['home_team']
        if home not in team_home_margin:
            team_home_margin[home] = []
        team_home_margin[home].append(row['Margin'])

    # Calculate average HFA per team
    team_hfa = {}
    for team, margins in team_home_margin.items():
        if len(margins) >= 3:
            team_hfa[team] = min(max(np.mean(margins), -5), 10)  # Clamp to reasonable range
        else:
            team_hfa[team] = 2.5  # Default HFA

    # Add HFA columns
    df['home_team_hfa'] = df['home_team'].map(team_hfa).fillna(2.5)
    df['away_team_hfa'] = df['away_team'].map(team_hfa).fillna(2.5)
    df['hfa_diff'] = df['home_team_hfa'] - df['away_team_hfa']

    return df

def calculate_interaction_features(df):
    """Calculate interaction features for the model."""
    # Fill NaN values with defaults
    df['home_pregame_elo'] = df['home_pregame_elo'].fillna(1500)
    df['away_pregame_elo'] = df['away_pregame_elo'].fillna(1500)
    df['home_comp_off_ppa'] = df['home_comp_off_ppa'].fillna(0)
    df['away_comp_def_ppa'] = df['away_comp_def_ppa'].fillna(0)
    df['home_comp_pass_ppa'] = df['home_comp_pass_ppa'].fillna(0)
    df['away_comp_pass_ppa'] = df['away_comp_pass_ppa'].fillna(0)
    df['home_comp_success'] = df['home_comp_success'].fillna(0.4)
    df['away_comp_success'] = df['away_comp_success'].fillna(0.4)

    # Interaction features
    df['elo_diff'] = df['home_pregame_elo'] - df['away_pregame_elo']
    df['net_epa'] = df['home_comp_off_ppa'] - df['away_comp_def_ppa']
    df['pass_efficiency_diff'] = df['home_comp_pass_ppa'] - df['away_comp_pass_ppa']
    df['epa_elo_interaction'] = df['net_epa'] * df['elo_diff']
    df['success_diff'] = df['home_comp_success'] - df['away_comp_success']

    return df

def add_placeholder_columns(df):
    """Add placeholder columns that may be expected by the model."""
    # These columns exist in the original but are complex to calculate
    # Using placeholders that won't break the model
    placeholder_cols = [
        'west_coast_early', 'home_lookahead', 'away_lookahead',
        'home_off_rush_success', 'home_off_pass_success', 'home_off_std_downs_ppa',
        'home_def_rush_success', 'home_def_pass_success', 'home_def_pass_downs_ppa',
        'away_off_rush_success', 'away_off_pass_success', 'away_off_std_downs_ppa',
        'away_def_rush_success', 'away_def_pass_success', 'away_def_pass_downs_ppa',
        'home_comp_epa', 'home_comp_ypp', 'away_comp_epa', 'away_comp_ypp',
        'home_clean_off_ppa', 'home_clean_def_ppa', 'home_raw_off_ppa', 'home_raw_def_ppa',
        'home_garbage_adj_off', 'home_garbage_adj_def',
        'away_clean_off_ppa', 'away_clean_def_ppa', 'away_raw_off_ppa', 'away_raw_def_ppa',
        'away_garbage_adj_off', 'away_garbage_adj_def',
        'home_adj_off_epa', 'away_adj_off_epa', 'home_adj_def_epa', 'away_adj_def_epa',
        'adj_net_epa', 'matchup_advantage',
    ]

    for col in placeholder_cols:
        if col not in df.columns:
            if 'success' in col:
                df[col] = 0.4
            elif 'ppa' in col or 'epa' in col:
                df[col] = 0.0
            else:
                df[col] = 0

    return df

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 60)
    print("SHARP SPORTS PREDICTOR - DATA REFRESH")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Step 1: Fetch all games
    print("Step 1: Fetching games...")
    all_games = []
    for year in YEARS:
        print(f"  Fetching {year}...")
        games = fetch_games(year)
        all_games.extend(games)
        print(f"    Got {len(games)} games")
    print(f"  Total games: {len(all_games)}")

    # Step 2: Process games
    print("\nStep 2: Processing games...")
    df = process_games(all_games)
    print(f"  Games with scores: {len(df)}")

    # Step 3: Calculate rest days
    print("\nStep 3: Calculating rest days...")
    df = calculate_rest_days(df)

    # Step 4: Calculate rolling stats
    print("\nStep 4: Calculating rolling stats...")
    df = calculate_rolling_stats(df)

    # Step 5: Fetch and process PPA data
    print("\nStep 5: Fetching PPA data...")
    all_ppa = []
    for year in YEARS:
        print(f"  Fetching PPA {year}...")
        ppa = fetch_ppa_games(year, exclude_garbage_time=True)
        all_ppa.extend(ppa)
        print(f"    Got {len(ppa)} records")
    print(f"  Total PPA records: {len(all_ppa)}")

    print("\nStep 6: Processing PPA data...")
    df = process_ppa_data(all_ppa, df)

    # Step 7: Calculate HFA
    print("\nStep 7: Calculating Home Field Advantage...")
    df = calculate_hfa(df)

    # Step 8: Calculate interaction features
    print("\nStep 8: Calculating interaction features...")
    df = calculate_interaction_features(df)

    # Step 9: Add placeholder columns
    print("\nStep 9: Adding placeholder columns...")
    df = add_placeholder_columns(df)

    # Step 10: Save to CSV
    print("\nStep 10: Saving to cfb_data_smart.csv...")
    df.to_csv('cfb_data_smart.csv', index=False)
    print(f"  Saved {len(df)} games")

    # Summary
    print("\n" + "=" * 60)
    print("DATA REFRESH COMPLETE")
    print("=" * 60)
    print(f"Total games: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['season'].min()} - {df['season'].max()}")
    print(f"Latest week: Season {df['season'].max()}, Week {df[df['season'] == df['season'].max()]['week'].max()}")
    print(f"\nFile saved: cfb_data_smart.csv")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
