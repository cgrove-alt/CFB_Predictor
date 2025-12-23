"""
Fetch College Football Data using direct API calls.
"""

import requests
import pandas as pd
from config import CFBD_API_KEY

# API Configuration
CFBD_BASE_URL = "https://api.collegefootballdata.com"

def get_headers():
    """Get authorization headers for CFBD API."""
    return {'Authorization': f'Bearer {CFBD_API_KEY}'}

def fetch_games(year):
    """Fetch all games for a given year."""
    url = f"{CFBD_BASE_URL}/games?year={year}"
    resp = requests.get(url, headers=get_headers())
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Error fetching {year}: {resp.status_code} - {resp.text}")
        return []

# Fetch games for 2022, 2023, 2024, 2025
all_games = []
years = [2022, 2023, 2024, 2025]

for year in years:
    print(f"Fetching games for {year}...")
    games = fetch_games(year)
    all_games.extend(games)
    print(f"  Found {len(games)} games")

print(f"\nTotal games fetched: {len(all_games)}")

# Extract relevant fields into a DataFrame (API uses camelCase)
data = []
for game in all_games:
    data.append({
        'id': game.get('id'),
        'season': game.get('season'),
        'week': game.get('week'),
        'home_team': game.get('homeTeam'),
        'away_team': game.get('awayTeam'),
        'home_points': game.get('homePoints'),
        'away_points': game.get('awayPoints'),
        'home_pregame_elo': game.get('homePregameElo'),
        'away_pregame_elo': game.get('awayPregameElo')
    })

df = pd.DataFrame(data)

# Filter out games where home_points is null (games that haven't happened yet)
df = df[df['home_points'].notna()]

# Calculate Margin (Home Points - Away Points)
df['Margin'] = df['home_points'] - df['away_points']

# Save to CSV
df.to_csv('cfb_data.csv', index=False)
print(f"\nSaved {len(df)} games to cfb_data.csv")

# Print first 5 rows
print("\nFirst 5 rows:")
print(df.head().to_string())
