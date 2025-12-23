"""
Fetch Talent Composite rankings and merge with game data.
"""

import cfbd
import pandas as pd
from config import CFBD_API_KEY

# Configure the API client
configuration = cfbd.Configuration()
configuration.access_token = CFBD_API_KEY
api_client = cfbd.ApiClient(configuration)
teams_api = cfbd.TeamsApi(api_client)

# Fetch talent data for 2022-2025
print("Fetching Talent Composite rankings...")
all_talent = []

for year in [2022, 2023, 2024, 2025]:
    print(f"  Fetching {year}...")
    try:
        talent = teams_api.get_talent(year=year)
        for t in talent:
            all_talent.append({
                'season': year,
                'team': t.team,
                'talent_score': t.talent
            })
        print(f"    Found {len(talent)} teams")
    except Exception as e:
        print(f"  Error fetching {year}: {e}")

talent_df = pd.DataFrame(all_talent)
print(f"Total talent records: {len(talent_df)}")

# Load existing game data
print("\nLoading cfb_data_smart.csv...")
games_df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games: {len(games_df)}")

# Create talent lookup by (season, team)
talent_lookup = talent_df.set_index(['season', 'team'])['talent_score'].to_dict()

# Find minimum talent score for fallback
min_talent = talent_df['talent_score'].min()
print(f"Minimum talent score (for fallback): {min_talent:.2f}")

# Merge talent for home teams
def get_talent(row, team_col):
    key = (row['season'], row[team_col])
    return talent_lookup.get(key, min_talent)

print("\nMerging talent data...")
games_df['home_talent_score'] = games_df.apply(lambda row: get_talent(row, 'home_team'), axis=1)
games_df['away_talent_score'] = games_df.apply(lambda row: get_talent(row, 'away_team'), axis=1)

# Check how many got the fallback
home_missing = (games_df['home_talent_score'] == min_talent).sum()
away_missing = (games_df['away_talent_score'] == min_talent).sum()
print(f"Games with home team using fallback: {home_missing}")
print(f"Games with away team using fallback: {away_missing}")

# Save the updated dataset
games_df.to_csv('cfb_data_smart.csv', index=False)
print(f"\nSaved updated dataset to 'cfb_data_smart.csv'")

# Print first 5 rows to prove talent columns exist
print("\nFirst 5 rows with talent columns:")
display_cols = ['season', 'week', 'home_team', 'away_team', 'home_talent_score', 'away_talent_score', 'Margin']
print(games_df[display_cols].head().to_string())
