"""
Fetch Advanced Team Game Stats and Weather data, merge with game data.
Note: Weather data requires a paid Patreon subscription (Tier 1+).
"""

import cfbd
import pandas as pd
import numpy as np
from config import CFBD_API_KEY

# Configure the API client
configuration = cfbd.Configuration()
configuration.access_token = CFBD_API_KEY
api_client = cfbd.ApiClient(configuration)
games_api = cfbd.GamesApi(api_client)
stats_api = cfbd.StatsApi(api_client)

years = [2022, 2023, 2024, 2025]

# ============================================================
# FETCH WEATHER DATA (Requires Patreon Tier 1+)
# ============================================================
print("Attempting to fetch Weather data...")
weather_data = []
weather_available = False

try:
    # Test if weather API is accessible
    test_weather = games_api.get_weather(year=2024, week=1)
    weather_available = True
    print("  Weather API accessible! Fetching all data...")

    for year in years:
        print(f"  Fetching weather for {year}...")
        for week in range(1, 20):
            try:
                weather = games_api.get_weather(year=year, week=week)
                if weather:
                    for w in weather:
                        weather_data.append({
                            'game_id': w.id,
                            'wind_speed': w.wind_speed if hasattr(w, 'wind_speed') else None,
                            'temperature': w.temperature if hasattr(w, 'temperature') else None,
                            'weather_condition': w.weather_condition if hasattr(w, 'weather_condition') else None
                        })
            except:
                continue
    print(f"  Total weather records: {len(weather_data)}")
except Exception as e:
    print(f"  Weather API requires Patreon Tier 1+ subscription")
    print(f"  Error: {str(e)[:100]}...")
    print("  Will use wind_speed = 0 as fallback")

# ============================================================
# FETCH ADVANCED GAME STATS
# ============================================================
print("\nFetching Advanced Team Game Stats...")
all_stats = []

for year in years:
    print(f"  Fetching {year}...")
    try:
        for week in range(1, 20):
            try:
                stats = stats_api.get_advanced_game_stats(year=year, week=week)
                if stats:
                    for s in stats:
                        record = {
                            'game_id': s.game_id,
                            'season': s.season,
                            'week': s.week,
                            'team': s.team,
                            'opponent': s.opponent,
                        }

                        # Offense stats (EPA = PPA in CFBD terminology)
                        if s.offense:
                            record['offense_epa'] = s.offense.ppa  # EPA/PPA
                            record['offense_success_rate'] = s.offense.success_rate
                            record['offense_explosiveness'] = s.offense.explosiveness
                            record['offense_plays'] = s.offense.plays

                        # Defense stats
                        if s.defense:
                            record['defense_epa'] = s.defense.ppa  # EPA/PPA
                            record['defense_success_rate'] = s.defense.success_rate
                            record['defense_explosiveness'] = s.defense.explosiveness
                            record['defense_plays'] = s.defense.plays

                        all_stats.append(record)
            except:
                continue
    except Exception as e:
        print(f"  Error fetching {year}: {e}")

print(f"Total advanced stat records: {len(all_stats)}")

# ============================================================
# LOAD EXISTING GAME DATA
# ============================================================
print("\nLoading cfb_data_smart.csv...")
games_df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games: {len(games_df)}")

# Remove old EPA columns if they exist (to avoid duplicates)
epa_cols_to_remove = [col for col in games_df.columns if 'offense_ppa' in col or 'defense_ppa' in col
                      or 'offense_epa' in col or 'defense_epa' in col
                      or 'success_rate' in col or 'explosiveness' in col or 'plays' in col.lower()]
if epa_cols_to_remove:
    print(f"Removing old columns: {epa_cols_to_remove}")
    games_df = games_df.drop(columns=epa_cols_to_remove, errors='ignore')

# ============================================================
# MERGE WEATHER DATA
# ============================================================
print("\nMerging weather data...")
if weather_data:
    weather_df = pd.DataFrame(weather_data)
    games_df = games_df.merge(weather_df, left_on='id', right_on='game_id', how='left')
    games_df = games_df.drop(columns=['game_id'], errors='ignore')
else:
    # No weather data available - set wind_speed to 0
    games_df['wind_speed'] = 0
    games_df['temperature'] = np.nan
    games_df['weather_condition'] = None

# Fill missing wind_speed with 0 (indoor/dome games or missing data)
games_df['wind_speed'] = games_df['wind_speed'].fillna(0)
print(f"  Games with wind_speed > 0: {(games_df['wind_speed'] > 0).sum()}")

# ============================================================
# MERGE ADVANCED STATS
# ============================================================
print("\nPreparing advanced stats for merge...")
stats_df = pd.DataFrame(all_stats)

# Create home team stats
home_stats = stats_df.copy()
home_stats = home_stats.rename(columns={
    'offense_epa': 'home_offense_epa',
    'offense_success_rate': 'home_offense_success_rate',
    'offense_explosiveness': 'home_offense_explosiveness',
    'offense_plays': 'home_offense_plays',
    'defense_epa': 'home_defense_epa',
    'defense_success_rate': 'home_defense_success_rate',
    'defense_explosiveness': 'home_defense_explosiveness',
    'defense_plays': 'home_defense_plays',
})
home_merge_cols = ['game_id', 'home_offense_epa', 'home_offense_success_rate',
                   'home_offense_explosiveness', 'home_offense_plays',
                   'home_defense_epa', 'home_defense_success_rate',
                   'home_defense_explosiveness', 'home_defense_plays']

# Create away team stats
away_stats = stats_df.copy()
away_stats = away_stats.rename(columns={
    'offense_epa': 'away_offense_epa',
    'offense_success_rate': 'away_offense_success_rate',
    'offense_explosiveness': 'away_offense_explosiveness',
    'offense_plays': 'away_offense_plays',
    'defense_epa': 'away_defense_epa',
    'defense_success_rate': 'away_defense_success_rate',
    'defense_explosiveness': 'away_defense_explosiveness',
    'defense_plays': 'away_defense_plays',
})
away_merge_cols = ['game_id', 'away_offense_epa', 'away_offense_success_rate',
                   'away_offense_explosiveness', 'away_offense_plays',
                   'away_defense_epa', 'away_defense_success_rate',
                   'away_defense_explosiveness', 'away_defense_plays']

# Merge home stats
games_df = games_df.merge(
    home_stats[home_merge_cols + ['team']],
    left_on=['id', 'home_team'],
    right_on=['game_id', 'team'],
    how='left'
).drop(columns=['game_id', 'team'], errors='ignore')

# Merge away stats
games_df = games_df.merge(
    away_stats[away_merge_cols + ['team']],
    left_on=['id', 'away_team'],
    right_on=['game_id', 'team'],
    how='left'
).drop(columns=['game_id', 'team'], errors='ignore')

# ============================================================
# FILL MISSING EPA WITH SEASON AVERAGES
# ============================================================
print("\nFilling missing EPA with season averages...")

# Calculate season averages for EPA
for season in games_df['season'].unique():
    season_mask = games_df['season'] == season

    # Home offense EPA
    season_avg = games_df.loc[season_mask, 'home_offense_epa'].mean()
    if pd.notna(season_avg):
        games_df.loc[season_mask, 'home_offense_epa'] = games_df.loc[season_mask, 'home_offense_epa'].fillna(season_avg)

    # Away offense EPA
    season_avg = games_df.loc[season_mask, 'away_offense_epa'].mean()
    if pd.notna(season_avg):
        games_df.loc[season_mask, 'away_offense_epa'] = games_df.loc[season_mask, 'away_offense_epa'].fillna(season_avg)

    # Home defense EPA
    season_avg = games_df.loc[season_mask, 'home_defense_epa'].mean()
    if pd.notna(season_avg):
        games_df.loc[season_mask, 'home_defense_epa'] = games_df.loc[season_mask, 'home_defense_epa'].fillna(season_avg)

    # Away defense EPA
    season_avg = games_df.loc[season_mask, 'away_defense_epa'].mean()
    if pd.notna(season_avg):
        games_df.loc[season_mask, 'away_defense_epa'] = games_df.loc[season_mask, 'away_defense_epa'].fillna(season_avg)

# Fill any remaining with overall average
games_df['home_offense_epa'] = games_df['home_offense_epa'].fillna(games_df['home_offense_epa'].mean())
games_df['away_offense_epa'] = games_df['away_offense_epa'].fillna(games_df['away_offense_epa'].mean())
games_df['home_defense_epa'] = games_df['home_defense_epa'].fillna(games_df['home_defense_epa'].mean())
games_df['away_defense_epa'] = games_df['away_defense_epa'].fillna(games_df['away_defense_epa'].mean())

# Count stats
home_epa_count = games_df['home_offense_epa'].notna().sum()
away_epa_count = games_df['away_offense_epa'].notna().sum()
print(f"  Games with home EPA: {home_epa_count}")
print(f"  Games with away EPA: {away_epa_count}")

# ============================================================
# SAVE UPDATED DATASET
# ============================================================
games_df.to_csv('cfb_data_smart.csv', index=False)
print(f"\nSaved updated dataset to 'cfb_data_smart.csv'")
print(f"Total columns: {len(games_df.columns)}")

# ============================================================
# PRINT SAMPLE DATA
# ============================================================
print("\n" + "="*70)
print("SAMPLE DATA (First 5 rows with wind_speed and offense_epa):")
print("="*70)
display_cols = ['season', 'week', 'home_team', 'away_team',
                'wind_speed', 'home_offense_epa', 'away_offense_epa']
print(games_df[display_cols].head().to_string())

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)
print(f"Total games: {len(games_df)}")
print(f"Games with wind_speed data: {(games_df['wind_speed'] > 0).sum()} (rest set to 0)")
print(f"Games with EPA data: {home_epa_count}")
print(f"Weather API status: {'Available' if weather_available else 'Requires Patreon Tier 1+'}")
