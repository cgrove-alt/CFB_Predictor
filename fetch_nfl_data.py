"""
Fetch NFL Data using nfl_data_py library.

Fetches schedule data and play-by-play advanced stats,
aggregates EPA/success metrics, and creates nfl_data_smart.csv.
"""

import nfl_data_py as nfl
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
YEARS = [2022, 2023, 2024, 2025]

# ============================================================
# FETCH SCHEDULES
# ============================================================
print("=" * 60)
print("FETCH NFL DATA")
print("=" * 60)

print("\nFetching schedule data for 2022-2025...")
schedules = nfl.import_schedules(YEARS)
print(f"Total games in schedule: {len(schedules)}")

# Keep relevant columns
schedule_cols = [
    'game_id', 'season', 'week', 'game_type',
    'home_team', 'away_team', 'home_score', 'away_score',
    'spread_line', 'total_line', 'home_rest', 'away_rest',
    'home_moneyline', 'away_moneyline', 'roof', 'surface', 'temp', 'wind'
]

# Filter to columns that exist
schedule_cols = [c for c in schedule_cols if c in schedules.columns]
games_df = schedules[schedule_cols].copy()

# Calculate margin (home perspective)
games_df['home_margin'] = games_df['home_score'] - games_df['away_score']
games_df['total_points'] = games_df['home_score'] + games_df['away_score']

print(f"Games with scores: {games_df['home_score'].notna().sum()}")

# ============================================================
# FETCH PLAY-BY-PLAY DATA
# ============================================================
print("\n" + "=" * 60)
print("FETCHING PLAY-BY-PLAY DATA (This may take a few minutes...)")
print("=" * 60)

# Only fetch years that have completed games
pbp_years = [y for y in YEARS if y <= 2024]
print(f"\nFetching PBP for: {pbp_years}")

try:
    pbp = nfl.import_pbp_data(pbp_years, downcast=True)
    print(f"Total plays loaded: {len(pbp)}")
except Exception as e:
    print(f"Error fetching PBP: {e}")
    print("Proceeding with schedule data only...")
    pbp = None

# ============================================================
# AGGREGATE ADVANCED STATS
# ============================================================
if pbp is not None and len(pbp) > 0:
    print("\n" + "=" * 60)
    print("AGGREGATING ADVANCED STATS")
    print("=" * 60)

    # Filter to real plays (not penalties, timeouts, etc.)
    plays = pbp[
        (pbp['play_type'].isin(['pass', 'run'])) &
        (pbp['epa'].notna()) &
        (pbp['posteam'].notna())
    ].copy()

    print(f"Valid offensive plays: {len(plays)}")

    # Aggregate by game_id and possession team (offense)
    print("\nCalculating offensive stats per team per game...")

    offense_stats = plays.groupby(['game_id', 'posteam']).agg({
        'epa': ['mean', 'sum', 'count'],
        'success': 'mean',  # success rate
        'yards_gained': 'mean',
        'air_yards': 'mean',
        'yards_after_catch': 'mean'
    }).reset_index()

    # Flatten column names
    offense_stats.columns = [
        'game_id', 'team',
        'avg_epa', 'total_epa', 'play_count',
        'success_rate', 'avg_yards', 'avg_air_yards', 'avg_yac'
    ]

    # Calculate rush vs pass EPA separately
    print("Calculating rush vs pass splits...")

    rush_stats = plays[plays['play_type'] == 'run'].groupby(['game_id', 'posteam']).agg({
        'epa': 'mean',
        'success': 'mean'
    }).reset_index()
    rush_stats.columns = ['game_id', 'team', 'rush_epa', 'rush_success']

    pass_stats = plays[plays['play_type'] == 'pass'].groupby(['game_id', 'posteam']).agg({
        'epa': 'mean',
        'success': 'mean'
    }).reset_index()
    pass_stats.columns = ['game_id', 'team', 'pass_epa', 'pass_success']

    # Merge rush/pass into offense stats
    offense_stats = offense_stats.merge(rush_stats, on=['game_id', 'team'], how='left')
    offense_stats = offense_stats.merge(pass_stats, on=['game_id', 'team'], how='left')

    print(f"Team-game stats created: {len(offense_stats)}")

    # ============================================================
    # CALCULATE DEFENSIVE STATS (opponent's offense against this team)
    # ============================================================
    print("\nCalculating defensive stats (opponent offense)...")

    defense_stats = plays.groupby(['game_id', 'defteam']).agg({
        'epa': 'mean',
        'success': 'mean'
    }).reset_index()
    defense_stats.columns = ['game_id', 'team', 'def_epa_allowed', 'def_success_allowed']

    # Merge defense into offense stats
    team_stats = offense_stats.merge(defense_stats, on=['game_id', 'team'], how='left')

    print(f"Complete team-game stats: {len(team_stats)}")

    # ============================================================
    # CREATE HOME/AWAY SPLITS
    # ============================================================
    print("\n" + "=" * 60)
    print("CREATING HOME/AWAY STATS")
    print("=" * 60)

    # Home team stats
    home_stats = team_stats.copy()
    home_stats = home_stats.rename(columns={
        'team': 'home_team',
        'avg_epa': 'home_avg_epa',
        'total_epa': 'home_total_epa',
        'play_count': 'home_plays',
        'success_rate': 'home_success_rate',
        'avg_yards': 'home_avg_yards',
        'rush_epa': 'home_rush_epa',
        'pass_epa': 'home_pass_epa',
        'rush_success': 'home_rush_success',
        'pass_success': 'home_pass_success',
        'def_epa_allowed': 'home_def_epa_allowed',
        'def_success_allowed': 'home_def_success_allowed'
    })

    # Away team stats
    away_stats = team_stats.copy()
    away_stats = away_stats.rename(columns={
        'team': 'away_team',
        'avg_epa': 'away_avg_epa',
        'total_epa': 'away_total_epa',
        'play_count': 'away_plays',
        'success_rate': 'away_success_rate',
        'avg_yards': 'away_avg_yards',
        'rush_epa': 'away_rush_epa',
        'pass_epa': 'away_pass_epa',
        'rush_success': 'away_rush_success',
        'pass_success': 'away_pass_success',
        'def_epa_allowed': 'away_def_epa_allowed',
        'def_success_allowed': 'away_def_success_allowed'
    })

    # Drop columns we don't need for merge
    home_cols = [c for c in home_stats.columns if c.startswith('home_') or c == 'game_id']
    away_cols = [c for c in away_stats.columns if c.startswith('away_') or c == 'game_id']

    home_stats = home_stats[home_cols]
    away_stats = away_stats[away_cols]

    # ============================================================
    # MERGE STATS INTO SCHEDULE
    # ============================================================
    print("\nMerging stats into schedule...")

    # Merge home stats
    games_df = games_df.merge(home_stats, on=['game_id', 'home_team'], how='left')

    # Merge away stats
    games_df = games_df.merge(away_stats, on=['game_id', 'away_team'], how='left')

    print(f"Games with home EPA: {games_df['home_avg_epa'].notna().sum()}")
    print(f"Games with away EPA: {games_df['away_avg_epa'].notna().sum()}")

else:
    print("\nNo PBP data available. Schedule-only dataset created.")

# ============================================================
# SAVE DATASET
# ============================================================
print("\n" + "=" * 60)
print("SAVING DATASET")
print("=" * 60)

games_df.to_csv('nfl_data_smart.csv', index=False)
print(f"\nSaved to 'nfl_data_smart.csv'")
print(f"Total games: {len(games_df)}")
print(f"Total columns: {len(games_df.columns)}")

# ============================================================
# PRINT SAMPLE DATA
# ============================================================
print("\n" + "=" * 60)
print("SAMPLE DATA (5 rows)")
print("=" * 60)

# Show key columns
display_cols = ['home_team', 'away_team', 'spread_line', 'home_avg_epa', 'away_avg_epa', 'home_margin']
display_cols = [c for c in display_cols if c in games_df.columns]

sample = games_df[games_df['home_avg_epa'].notna()][display_cols].head()
print(sample.to_string())

# ============================================================
# COLUMN SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ALL COLUMNS")
print("=" * 60)

for col in games_df.columns:
    non_null = games_df[col].notna().sum()
    print(f"  {col}: {non_null} values")

print("\n" + "=" * 60)
print("NFL DATA FETCH COMPLETE")
print("=" * 60)
