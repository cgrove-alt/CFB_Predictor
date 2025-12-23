"""
Fetch and Analyze Line Movement Data.

Line movement is one of the most valuable signals for sharp betting:
- Lines moving TOWARD a team despite public betting AGAINST = sharp money
- Reverse line movement (RLM) is a strong indicator of smart money
- Opening lines are set by market makers, closing lines by the market

Sources:
1. CFBD API - Opening and closing lines
2. Future: Add The Odds API for real-time line comparison

Usage:
    python fetch_line_movement.py
"""

import requests
import pandas as pd
import numpy as np
from config import CFBD_API_KEY
from datetime import datetime
import json

CFBD_BASE_URL = "https://api.collegefootballdata.com"
YEARS = [2022, 2023, 2024, 2025]


def get_headers():
    """Get authorization headers for CFBD API."""
    return {'Authorization': f'Bearer {CFBD_API_KEY}'}


def fetch_betting_lines_detailed(year):
    """
    Fetch detailed betting lines for all games in a year.

    Returns all available lines from all providers, not just consensus.
    """
    url = f"{CFBD_BASE_URL}/lines?year={year}"
    resp = requests.get(url, headers=get_headers())
    if resp.status_code == 200:
        return resp.json()
    print(f"  Error fetching lines {year}: {resp.status_code}")
    return []


def process_lines_with_movement(all_lines):
    """
    Process lines with enhanced line movement analysis.

    Returns dict with:
    - vegas_spread: Closing spread
    - spread_open: Opening spread
    - line_movement: Change from open to close
    - movement_direction: 'home_steam', 'away_steam', or 'stable'
    - sharp_indicator: Probability of sharp money
    - num_providers: Number of books with lines
    - spread_variance: Variance across books (disagreement = uncertainty)
    """
    lines_lookup = {}

    for game in all_lines:
        game_id = game.get('id')
        if not game_id:
            continue

        lines = game.get('lines', [])
        if not lines:
            continue

        # Collect all spreads from all providers
        all_spreads = []
        opening_spreads = []
        closing_spreads = []

        for line in lines:
            spread = line.get('spread')
            spread_open = line.get('spreadOpen')

            if spread is not None:
                all_spreads.append(spread)
                closing_spreads.append(spread)

            if spread_open is not None:
                opening_spreads.append(spread_open)

        if not closing_spreads:
            continue

        # Use consensus or average for final values
        consensus_line = None
        for line in lines:
            if 'consensus' in line.get('provider', '').lower():
                consensus_line = line
                break

        if consensus_line:
            closing_spread = consensus_line.get('spread')
            opening_spread = consensus_line.get('spreadOpen', closing_spread)
            over_under = consensus_line.get('overUnder')
        else:
            closing_spread = np.median(closing_spreads)
            opening_spread = np.median(opening_spreads) if opening_spreads else closing_spread
            over_under = np.median([l.get('overUnder', 0) for l in lines if l.get('overUnder')])

        # Calculate line movement
        if opening_spread is not None and closing_spread is not None:
            movement = closing_spread - opening_spread
        else:
            movement = 0

        # Determine movement direction
        if movement < -1.5:  # Line moved toward home team (home became more favored)
            movement_direction = 'home_steam'
        elif movement > 1.5:  # Line moved toward away team
            movement_direction = 'away_steam'
        else:
            movement_direction = 'stable'

        # Calculate spread variance (disagreement between books)
        spread_variance = np.var(closing_spreads) if len(closing_spreads) > 1 else 0

        # Sharp indicator based on:
        # 1. Large movement (>2 points)
        # 2. Low variance (books agree)
        # 3. Movement opposite to expected public bias
        abs_movement = abs(movement)
        sharp_score = 0

        if abs_movement >= 3:
            sharp_score += 0.4
        elif abs_movement >= 2:
            sharp_score += 0.25
        elif abs_movement >= 1:
            sharp_score += 0.1

        if spread_variance < 0.5:  # Books strongly agree
            sharp_score += 0.2
        elif spread_variance < 1.0:
            sharp_score += 0.1

        # Movement toward road team is often sharp (public bets home)
        if movement > 1.5:  # Away team steam
            sharp_score += 0.15

        lines_lookup[game_id] = {
            'vegas_spread': closing_spread,
            'spread_open': opening_spread,
            'over_under': over_under,
            'line_movement': movement,
            'abs_line_movement': abs_movement,
            'movement_direction': movement_direction,
            'spread_variance': spread_variance,
            'num_providers': len(lines),
            'sharp_indicator': min(sharp_score, 1.0),  # Cap at 1.0
        }

    return lines_lookup


def identify_steam_moves(lines_lookup, threshold=2.0):
    """
    Identify games with significant steam moves.

    Steam move = Line moves significantly in one direction due to sharp money.
    """
    steam_games = []

    for game_id, data in lines_lookup.items():
        movement = data.get('line_movement', 0)

        if abs(movement) >= threshold:
            steam_games.append({
                'game_id': game_id,
                'movement': movement,
                'direction': data['movement_direction'],
                'closing_spread': data['vegas_spread'],
                'opening_spread': data['spread_open'],
            })

    return steam_games


def analyze_reverse_line_movement(df, lines_lookup):
    """
    Identify games with reverse line movement.

    RLM = Line moves OPPOSITE to what public betting would suggest.
    Classic example: Public hammers favorite, but line moves toward dog.
    """
    # This requires public betting percentage data which we don't have
    # For now, use heuristics based on team popularity and line movement

    rlm_games = []

    for _, row in df.iterrows():
        game_id = row.get('id')
        if game_id not in lines_lookup:
            continue

        line_data = lines_lookup[game_id]
        movement = line_data.get('line_movement', 0)

        # Heuristic: If home team is big favorite (-10+) but line moves AWAY from them
        # This is potential RLM (public would bet the big home fave)
        opening = line_data.get('spread_open', 0)
        if opening is not None and opening < -10 and movement > 1.5:
            rlm_games.append({
                'game_id': game_id,
                'home_team': row.get('home_team'),
                'away_team': row.get('away_team'),
                'opening': opening,
                'closing': line_data['vegas_spread'],
                'movement': movement,
                'rlm_type': 'big_home_fave_faded'
            })

        # Heuristic: If away team is small dog (+3 or less) but line moves toward them
        # Could indicate sharp money on away team
        if opening is not None and 0 < opening <= 3 and movement < -1.5:
            rlm_games.append({
                'game_id': game_id,
                'home_team': row.get('home_team'),
                'away_team': row.get('away_team'),
                'opening': opening,
                'closing': line_data['vegas_spread'],
                'movement': movement,
                'rlm_type': 'small_road_dog_steamed'
            })

    return rlm_games


def generate_line_movement_features(game_id, lines_lookup):
    """
    Generate line movement features for a specific game.

    Returns dict of features to add to prediction.
    """
    if game_id not in lines_lookup:
        return {
            'line_movement': 0,
            'abs_line_movement': 0,
            'movement_home_steam': 0,
            'movement_away_steam': 0,
            'sharp_indicator': 0,
            'spread_variance': 0,
            'num_providers': 0,
        }

    data = lines_lookup[game_id]

    return {
        'line_movement': data.get('line_movement', 0),
        'abs_line_movement': data.get('abs_line_movement', 0),
        'movement_home_steam': 1 if data.get('movement_direction') == 'home_steam' else 0,
        'movement_away_steam': 1 if data.get('movement_direction') == 'away_steam' else 0,
        'sharp_indicator': data.get('sharp_indicator', 0),
        'spread_variance': data.get('spread_variance', 0),
        'num_providers': data.get('num_providers', 0),
    }


def main():
    print("=" * 70)
    print("FETCHING DETAILED LINE MOVEMENT DATA")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Fetch all betting lines
    print("\nFetching betting lines from CFBD API...")
    all_lines = []
    for year in YEARS:
        print(f"  Fetching {year}...")
        lines = fetch_betting_lines_detailed(year)
        all_lines.extend(lines)
        print(f"    Got {len(lines)} games")

    print(f"\nTotal games: {len(all_lines)}")

    # Process with enhanced movement analysis
    print("\nProcessing line movement...")
    lines_lookup = process_lines_with_movement(all_lines)
    print(f"Games with valid line data: {len(lines_lookup)}")

    # Identify steam moves
    steam_games = identify_steam_moves(lines_lookup, threshold=2.0)
    print(f"\nGames with steam moves (>2pt): {len(steam_games)}")

    # Load existing data
    print("\nLoading cfb_data_smart.csv...")
    try:
        df = pd.read_csv('cfb_data_smart.csv')
        print(f"Loaded {len(df)} games")
    except FileNotFoundError:
        print("cfb_data_smart.csv not found!")
        df = pd.DataFrame()

    # Analyze RLM if we have data
    if not df.empty:
        rlm_games = analyze_reverse_line_movement(df, lines_lookup)
        print(f"Games with potential RLM: {len(rlm_games)}")

        # Add line movement features
        print("\nAdding enhanced line movement features...")

        line_movements = []
        abs_movements = []
        home_steam = []
        away_steam = []
        sharp_scores = []
        variances = []

        for _, row in df.iterrows():
            features = generate_line_movement_features(row.get('id'), lines_lookup)
            line_movements.append(features['line_movement'])
            abs_movements.append(features['abs_line_movement'])
            home_steam.append(features['movement_home_steam'])
            away_steam.append(features['movement_away_steam'])
            sharp_scores.append(features['sharp_indicator'])
            variances.append(features['spread_variance'])

        df['line_movement_v2'] = line_movements
        df['abs_line_movement'] = abs_movements
        df['movement_home_steam'] = home_steam
        df['movement_away_steam'] = away_steam
        df['sharp_indicator'] = sharp_scores
        df['spread_variance'] = variances

        # Save
        print("Saving enhanced data...")
        df.to_csv('cfb_data_smart.csv', index=False)

    # Save line movement summary
    summary = {
        'last_update': datetime.now().isoformat(),
        'total_games': len(lines_lookup),
        'steam_moves': len(steam_games),
        'avg_movement': np.mean([d['line_movement'] for d in lines_lookup.values()]),
        'movement_stats': {
            'home_steam': sum(1 for d in lines_lookup.values() if d['movement_direction'] == 'home_steam'),
            'away_steam': sum(1 for d in lines_lookup.values() if d['movement_direction'] == 'away_steam'),
            'stable': sum(1 for d in lines_lookup.values() if d['movement_direction'] == 'stable'),
        }
    }

    with open('.line_movement_status.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("LINE MOVEMENT SUMMARY")
    print("=" * 70)
    print(f"\nMovement distribution:")
    print(f"  Home steam: {summary['movement_stats']['home_steam']}")
    print(f"  Away steam: {summary['movement_stats']['away_steam']}")
    print(f"  Stable: {summary['movement_stats']['stable']}")
    print(f"\nAverage movement: {summary['avg_movement']:.2f} points")

    if steam_games:
        print(f"\nTop 10 biggest line moves:")
        sorted_steam = sorted(steam_games, key=lambda x: abs(x['movement']), reverse=True)[:10]
        for game in sorted_steam:
            print(f"  {game['direction']}: {game['opening']} -> {game['closing']} ({game['movement']:+.1f})")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
