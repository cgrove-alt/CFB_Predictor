"""
Fetch Weather Data from CFBD API (V2).

Uses direct API calls instead of cfbd library to avoid pydantic version conflicts.
Requires Patreon Tier 1+ subscription.
"""

import requests
import pandas as pd
import numpy as np
from config import CFBD_API_KEY

BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {
    "Authorization": f"Bearer {CFBD_API_KEY}",
    "Accept": "application/json"
}

years = [2022, 2023, 2024, 2025]

print("=" * 70)
print("FETCH WEATHER DATA (V2 - Direct API)")
print("=" * 70)

# ============================================================
# FETCH WEATHER DATA
# ============================================================
print("\nFetching weather data from CFBD API...")
weather_data = []
weather_available = False

try:
    # Test if weather API is accessible
    test_url = f"{BASE_URL}/games/weather?year=2024&week=1"
    test_resp = requests.get(test_url, headers=HEADERS)

    if test_resp.status_code == 200:
        weather_available = True
        print("  Weather API accessible! Fetching all data...")

        for year in years:
            print(f"  Fetching weather for {year}...")
            for week in range(1, 20):
                try:
                    url = f"{BASE_URL}/games/weather?year={year}&week={week}"
                    resp = requests.get(url, headers=HEADERS)
                    if resp.status_code == 200:
                        data = resp.json()
                        for w in data:
                            weather_data.append({
                                'game_id': w.get('id'),
                                'wind_speed': w.get('windSpeed'),
                                'temperature': w.get('temperature'),
                                'weather_condition': w.get('weatherCondition')
                            })
                except Exception as e:
                    continue
        print(f"  Total weather records: {len(weather_data)}")
    else:
        print(f"  Weather API returned status {test_resp.status_code}")
        print(f"  Weather API requires Patreon Tier 1+ subscription")
        print("  Will use wind_speed = 0 as fallback")
except Exception as e:
    print(f"  Error accessing Weather API: {str(e)[:100]}")
    print("  Will use wind_speed = 0 as fallback")

# ============================================================
# LOAD EXISTING GAME DATA
# ============================================================
print("\nLoading cfb_data_smart.csv...")
games_df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games: {len(games_df)}")

# ============================================================
# MERGE WEATHER DATA
# ============================================================
print("\nMerging weather data...")
if weather_data:
    weather_df = pd.DataFrame(weather_data)

    # Check for existing weather columns
    if 'wind_speed' in games_df.columns:
        games_df = games_df.drop(columns=['wind_speed'], errors='ignore')
    if 'temperature' in games_df.columns:
        games_df = games_df.drop(columns=['temperature'], errors='ignore')
    if 'weather_condition' in games_df.columns:
        games_df = games_df.drop(columns=['weather_condition'], errors='ignore')

    games_df = games_df.merge(weather_df, left_on='id', right_on='game_id', how='left')
    games_df = games_df.drop(columns=['game_id'], errors='ignore')
else:
    # No weather data available - set wind_speed to 0
    games_df['wind_speed'] = 0
    games_df['temperature'] = np.nan
    games_df['weather_condition'] = None

# Fill missing wind_speed with 0 (indoor/dome games or missing data)
games_df['wind_speed'] = games_df['wind_speed'].fillna(0)
games_with_wind = (games_df['wind_speed'] > 0).sum()
print(f"  Games with wind_speed > 0: {games_with_wind}")

# ============================================================
# SAVE UPDATED DATASET
# ============================================================
games_df.to_csv('cfb_data_smart.csv', index=False)
print(f"\nSaved updated dataset to 'cfb_data_smart.csv'")
print(f"Total columns: {len(games_df.columns)}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total games: {len(games_df)}")
print(f"Games with wind_speed data: {games_with_wind}")
print(f"Weather API status: {'Available' if weather_available else 'Requires Patreon Tier 1+'}")

if weather_available:
    # Show sample data
    print("\nSample weather data:")
    sample = games_df[games_df['wind_speed'] > 0][['season', 'week', 'home_team', 'away_team', 'wind_speed', 'temperature']].head(5)
    print(sample.to_string())
