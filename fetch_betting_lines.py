"""
Fetch historical betting lines from CFBD API.

This adds Vegas spreads to our training data, which is crucial because:
1. Vegas lines are highly predictive of actual margins
2. Models can learn to find edges vs Vegas
3. We can train on spread error instead of raw margin
"""

import requests
import pandas as pd
import numpy as np
from config import CFBD_API_KEY
from datetime import datetime

CFBD_BASE_URL = "https://api.collegefootballdata.com"
YEARS = [2022, 2023, 2024, 2025]

def get_headers():
    """Get authorization headers for CFBD API."""
    return {'Authorization': f'Bearer {CFBD_API_KEY}'}

def fetch_betting_lines(year):
    """Fetch betting lines for all games in a year."""
    url = f"{CFBD_BASE_URL}/lines?year={year}"
    resp = requests.get(url, headers=get_headers())
    if resp.status_code == 200:
        return resp.json()
    print(f"  Error fetching lines {year}: {resp.status_code}")
    return []

def process_betting_lines(all_lines):
    """Process betting lines into a lookup dictionary by game_id."""
    lines_lookup = {}

    for game in all_lines:
        game_id = game.get('id')
        if not game_id:
            continue

        lines = game.get('lines', [])
        if not lines:
            continue

        # Get consensus line (prefer 'consensus' provider, else first available)
        best_line = None
        for line in lines:
            provider = line.get('provider', '').lower()
            if 'consensus' in provider:
                best_line = line
                break

        if best_line is None and lines:
            # Use first available line
            best_line = lines[0]

        if best_line:
            spread = best_line.get('spread')
            over_under = best_line.get('overUnder')
            spread_open = best_line.get('spreadOpen')

            lines_lookup[game_id] = {
                'vegas_spread': spread,  # Home team spread (negative = favored)
                'over_under': over_under,
                'spread_open': spread_open,
                'line_provider': best_line.get('provider', 'unknown')
            }

    return lines_lookup

def main():
    print("=" * 60)
    print("FETCHING HISTORICAL BETTING LINES")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Fetch all betting lines
    print("Fetching betting lines from CFBD API...")
    all_lines = []
    for year in YEARS:
        print(f"  Fetching {year}...")
        lines = fetch_betting_lines(year)
        all_lines.extend(lines)
        print(f"    Got {len(lines)} games with lines")

    print(f"\nTotal games with betting data: {len(all_lines)}")

    # Process into lookup
    lines_lookup = process_betting_lines(all_lines)
    print(f"Games with valid spreads: {len(lines_lookup)}")

    # Load existing data
    print("\nLoading cfb_data_smart.csv...")
    df = pd.read_csv('cfb_data_smart.csv')
    print(f"Games in data: {len(df)}")

    # Add betting lines to data
    print("\nMerging betting lines...")
    vegas_spreads = []
    over_unders = []
    spread_opens = []

    matched = 0
    for _, row in df.iterrows():
        game_id = row['id']
        if game_id in lines_lookup:
            line_data = lines_lookup[game_id]
            vegas_spreads.append(line_data['vegas_spread'])
            over_unders.append(line_data['over_under'])
            spread_opens.append(line_data['spread_open'])
            matched += 1
        else:
            vegas_spreads.append(None)
            over_unders.append(None)
            spread_opens.append(None)

    df['vegas_spread'] = vegas_spreads
    df['over_under'] = over_unders
    df['spread_open'] = spread_opens

    print(f"Games matched with betting lines: {matched} ({matched/len(df)*100:.1f}%)")

    # Calculate derived features
    print("\nCalculating derived betting features...")

    # Line movement (how much the spread moved from open to close)
    df['line_movement'] = df['vegas_spread'] - df['spread_open']

    # Spread error (how wrong Vegas was - positive = home beat spread)
    df['spread_error'] = df['Margin'] - (-df['vegas_spread'])  # Vegas spread is typically shown as home spread

    # Did home team cover?
    df['home_covered'] = (df['Margin'] > -df['vegas_spread']).astype(int)

    # Total points vs over/under
    df['total_points'] = df['home_points'] + df['away_points']
    df['over_under_error'] = df['total_points'] - df['over_under']

    # Save updated data
    print("\nSaving enhanced data...")
    df.to_csv('cfb_data_smart.csv', index=False)

    # Summary stats
    print("\n" + "=" * 60)
    print("BETTING DATA SUMMARY")
    print("=" * 60)

    df_with_lines = df[df['vegas_spread'].notna()]
    print(f"\nGames with Vegas lines: {len(df_with_lines)}")

    if len(df_with_lines) > 0:
        print(f"\nVegas Spread Stats:")
        print(f"  Mean spread: {df_with_lines['vegas_spread'].mean():.2f}")
        print(f"  Std spread: {df_with_lines['vegas_spread'].std():.2f}")

        print(f"\nVegas Accuracy:")
        print(f"  Mean Absolute Error: {df_with_lines['spread_error'].abs().mean():.2f} points")
        print(f"  Home cover rate: {df_with_lines['home_covered'].mean()*100:.1f}%")

        print(f"\nOver/Under Stats:")
        print(f"  Mean O/U: {df_with_lines['over_under'].mean():.1f}")
        print(f"  Mean total points: {df_with_lines['total_points'].mean():.1f}")
        print(f"  O/U MAE: {df_with_lines['over_under_error'].abs().mean():.2f} points")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
