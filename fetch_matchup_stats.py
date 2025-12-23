"""
Fetch Advanced Team Game Stats from CFBD API.

Extracts:
- Rushing success rates (offense/defense)
- Passing success rates (offense/defense)
- Standard downs PPA (offense)
- Passing downs PPA (defense)
"""

import cfbd
import pandas as pd
import numpy as np
from config import CFBD_API_KEY

# ============================================================
# API SETUP
# ============================================================
configuration = cfbd.Configuration()
configuration.access_token = CFBD_API_KEY
api_client = cfbd.ApiClient(configuration)
stats_api = cfbd.StatsApi(api_client)

# ============================================================
# FETCH ADVANCED GAME STATS
# ============================================================
print("=" * 60)
print("FETCHING ADVANCED TEAM GAME STATS")
print("=" * 60)

all_stats = []
years = [2022, 2023, 2024, 2025]

for year in years:
    print(f"\nFetching {year}...")
    try:
        # Get advanced game stats for the year (fetch all weeks)
        stats = stats_api.get_advanced_game_stats(year=year)
        print(f"  Found {len(stats)} game stats records")

        for game_stat in stats:
            game_id = game_stat.game_id

            # Process each team's stats in this game
            offense = game_stat.offense
            defense = game_stat.defense

            record = {
                'game_id': game_id,
                'team': game_stat.team,
                'opponent': game_stat.opponent,
            }

            # Extract offense rushing success rate
            if offense and offense.rushing_plays:
                record['off_rush_success'] = offense.rushing_plays.success_rate
            else:
                record['off_rush_success'] = None

            # Extract offense passing success rate
            if offense and offense.passing_plays:
                record['off_pass_success'] = offense.passing_plays.success_rate
            else:
                record['off_pass_success'] = None

            # Extract offense standard downs PPA
            if offense and offense.standard_downs:
                record['off_std_downs_ppa'] = offense.standard_downs.ppa
            else:
                record['off_std_downs_ppa'] = None

            # Extract defense rushing success rate
            if defense and defense.rushing_plays:
                record['def_rush_success'] = defense.rushing_plays.success_rate
            else:
                record['def_rush_success'] = None

            # Extract defense passing success rate
            if defense and defense.passing_plays:
                record['def_pass_success'] = defense.passing_plays.success_rate
            else:
                record['def_pass_success'] = None

            # Extract defense passing downs PPA
            if defense and defense.passing_downs:
                record['def_pass_downs_ppa'] = defense.passing_downs.ppa
            else:
                record['def_pass_downs_ppa'] = None

            all_stats.append(record)

    except Exception as e:
        print(f"  Error fetching {year}: {e}")

print(f"\nTotal stats records: {len(all_stats)}")

# ============================================================
# CREATE DATAFRAME
# ============================================================
stats_df = pd.DataFrame(all_stats)
print(f"\nStats DataFrame shape: {stats_df.shape}")
print(f"Columns: {list(stats_df.columns)}")

# Check for data
print("\nSample data:")
print(stats_df.head())

# ============================================================
# LOAD EXISTING GAME DATA
# ============================================================
print("\n" + "=" * 60)
print("MERGING WITH GAME DATA")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Games loaded: {len(df)}")

# ============================================================
# PREPARE STATS FOR MERGE
# ============================================================
# We need to match stats to games - stats have game_id, team, opponent
# Our game data has id, home_team, away_team

if len(stats_df) == 0:
    print("\nNo stats fetched - cannot merge.")
    exit(1)

# Create home team stats (where team = home_team)
home_stats = stats_df.copy()
home_stats = home_stats.rename(columns={
    'game_id': 'id',
    'off_rush_success': 'home_off_rush_success',
    'off_pass_success': 'home_off_pass_success',
    'off_std_downs_ppa': 'home_off_std_downs_ppa',
    'def_rush_success': 'home_def_rush_success',
    'def_pass_success': 'home_def_pass_success',
    'def_pass_downs_ppa': 'home_def_pass_downs_ppa',
    'team': 'home_team'
})
home_stats = home_stats[['id', 'home_team', 'home_off_rush_success', 'home_off_pass_success',
                          'home_off_std_downs_ppa', 'home_def_rush_success',
                          'home_def_pass_success', 'home_def_pass_downs_ppa']]

# Create away team stats (where team = away_team)
away_stats = stats_df.copy()
away_stats = away_stats.rename(columns={
    'game_id': 'id',
    'off_rush_success': 'away_off_rush_success',
    'off_pass_success': 'away_off_pass_success',
    'off_std_downs_ppa': 'away_off_std_downs_ppa',
    'def_rush_success': 'away_def_rush_success',
    'def_pass_success': 'away_def_pass_success',
    'def_pass_downs_ppa': 'away_def_pass_downs_ppa',
    'team': 'away_team'
})
away_stats = away_stats[['id', 'away_team', 'away_off_rush_success', 'away_off_pass_success',
                          'away_off_std_downs_ppa', 'away_def_rush_success',
                          'away_def_pass_success', 'away_def_pass_downs_ppa']]

# ============================================================
# MERGE WITH GAME DATA
# ============================================================
print("\nMerging home team stats...")
# Drop existing columns if they exist
cols_to_drop = [col for col in df.columns if 'rush_success' in col or 'pass_success' in col
                or 'std_downs_ppa' in col or 'pass_downs_ppa' in col]
if cols_to_drop:
    print(f"  Dropping existing columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

df = df.merge(home_stats, on=['id', 'home_team'], how='left')
print(f"  After home merge: {len(df)} games")

print("Merging away team stats...")
df = df.merge(away_stats, on=['id', 'away_team'], how='left')
print(f"  After away merge: {len(df)} games")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("MERGE SUMMARY")
print("=" * 60)

new_cols = ['home_off_rush_success', 'home_off_pass_success', 'home_off_std_downs_ppa',
            'home_def_rush_success', 'home_def_pass_success', 'home_def_pass_downs_ppa',
            'away_off_rush_success', 'away_off_pass_success', 'away_off_std_downs_ppa',
            'away_def_rush_success', 'away_def_pass_success', 'away_def_pass_downs_ppa']

for col in new_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null} values ({non_null/len(df)*100:.1f}%)")

# ============================================================
# SAVE
# ============================================================
print("\n" + "=" * 60)
print("SAVING DATA")
print("=" * 60)

df.to_csv('cfb_data_smart.csv', index=False)
print(f"\nSaved {len(df)} games to 'cfb_data_smart.csv'")
print(f"Total columns: {len(df.columns)}")

# Print all columns
print("\nAll columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
