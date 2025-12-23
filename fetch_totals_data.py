"""
Fetch Totals Data for CFB Betting Model.

Fetches pace, PPA, and over/under lines for totals betting analysis.
Creates cfb_totals_data.csv with merged stats and betting lines.
"""

import pandas as pd
import numpy as np
import cfbd
from config import CFBD_API_KEY
import time

# ============================================================
# API SETUP
# ============================================================
configuration = cfbd.Configuration()
configuration.access_token = CFBD_API_KEY
api_client = cfbd.ApiClient(configuration)
stats_api = cfbd.StatsApi(api_client)
betting_api = cfbd.BettingApi(api_client)
games_api = cfbd.GamesApi(api_client)

# ============================================================
# FETCH ADVANCED TEAM GAME STATS
# ============================================================
print("=" * 60)
print("FETCH TOTALS DATA")
print("=" * 60)

print("\n" + "=" * 60)
print("FETCHING ADVANCED GAME STATS")
print("=" * 60)

all_stats = []
years = [2022, 2023, 2024, 2025]

for year in years:
    print(f"\nFetching stats for {year}...")
    try:
        stats = stats_api.get_advanced_game_stats(year=year)
        print(f"  Found {len(stats)} game stat records")

        for game_stat in stats:
            game_id = game_stat.game_id
            team = game_stat.team
            opponent = game_stat.opponent

            offense = game_stat.offense
            defense = game_stat.defense

            record = {
                'game_id': game_id,
                'team': team,
                'opponent': opponent,
            }

            # Extract offense plays and PPA
            if offense:
                record['offense_plays'] = offense.plays if hasattr(offense, 'plays') else None
                record['offense_ppa'] = offense.ppa if hasattr(offense, 'ppa') else None
                record['offense_total_ppa'] = offense.total_ppa if hasattr(offense, 'total_ppa') else None
            else:
                record['offense_plays'] = None
                record['offense_ppa'] = None
                record['offense_total_ppa'] = None

            # Extract defense plays and PPA
            if defense:
                record['defense_plays'] = defense.plays if hasattr(defense, 'plays') else None
                record['defense_ppa'] = defense.ppa if hasattr(defense, 'ppa') else None
                record['defense_total_ppa'] = defense.total_ppa if hasattr(defense, 'total_ppa') else None
            else:
                record['defense_plays'] = None
                record['defense_ppa'] = None
                record['defense_total_ppa'] = None

            record['season'] = year
            all_stats.append(record)

    except Exception as e:
        print(f"  Error fetching {year}: {e}")

    time.sleep(0.5)

print(f"\nTotal stats records: {len(all_stats)}")

# Create DataFrame
stats_df = pd.DataFrame(all_stats)

# Calculate Pace = (offense_plays + defense_plays) / 2
stats_df['pace'] = (stats_df['offense_plays'].fillna(0) + stats_df['defense_plays'].fillna(0)) / 2

print(f"\nStats DataFrame shape: {stats_df.shape}")
print(f"Pace stats: Mean={stats_df['pace'].mean():.1f}, Min={stats_df['pace'].min():.0f}, Max={stats_df['pace'].max():.0f}")

# ============================================================
# FETCH BETTING LINES (Over/Under)
# ============================================================
print("\n" + "=" * 60)
print("FETCHING BETTING LINES (OVER/UNDER)")
print("=" * 60)

all_lines = []

for year in years:
    print(f"\nFetching betting lines for {year}...")
    try:
        lines = betting_api.get_lines(year=year)
        print(f"  Found {len(lines)} games with lines")

        for game in lines:
            game_id = game.id
            home_team = game.home_team
            away_team = game.away_team
            home_score = game.home_score
            away_score = game.away_score

            # Calculate actual total
            if home_score is not None and away_score is not None:
                actual_total = home_score + away_score
            else:
                actual_total = None

            # Get consensus line (or first available)
            over_under = None
            spread = None
            provider = None

            if game.lines:
                # Look for consensus first
                for line in game.lines:
                    if line.provider and 'consensus' in line.provider.lower():
                        over_under = line.over_under
                        spread = line.spread
                        provider = line.provider
                        break

                # If no consensus, use first available
                if over_under is None and game.lines:
                    line = game.lines[0]
                    over_under = line.over_under
                    spread = line.spread
                    provider = line.provider

            all_lines.append({
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'actual_total': actual_total,
                'over_under': over_under,
                'spread': spread,
                'provider': provider,
                'season': year
            })

    except Exception as e:
        print(f"  Error fetching {year}: {e}")

    time.sleep(0.5)

print(f"\nTotal betting records: {len(all_lines)}")

lines_df = pd.DataFrame(all_lines)

# Filter to games with over/under lines
lines_with_ou = lines_df[lines_df['over_under'].notna()]
print(f"Games with Over/Under lines: {len(lines_with_ou)}")

# ============================================================
# FETCH GAME DATA (for week info)
# ============================================================
print("\n" + "=" * 60)
print("FETCHING GAME DATA")
print("=" * 60)

all_games = []

for year in years:
    print(f"\nFetching games for {year}...")
    try:
        games = games_api.get_games(year=year)
        print(f"  Found {len(games)} games")

        for game in games:
            all_games.append({
                'game_id': game.id,
                'week': game.week,
                'season_type': game.season_type,
                'start_date': game.start_date
            })

    except Exception as e:
        print(f"  Error: {e}")

    time.sleep(0.5)

games_df = pd.DataFrame(all_games)
print(f"\nTotal games: {len(games_df)}")

# ============================================================
# AGGREGATE STATS BY GAME
# ============================================================
print("\n" + "=" * 60)
print("AGGREGATING STATS BY GAME")
print("=" * 60)

# We have stats per team per game - need to aggregate to game level
# Group by game_id and aggregate

game_stats = stats_df.groupby('game_id').agg({
    'offense_plays': 'sum',  # Total plays in game
    'defense_plays': 'sum',
    'offense_ppa': 'mean',   # Average PPA
    'defense_ppa': 'mean',
    'offense_total_ppa': 'sum',
    'defense_total_ppa': 'sum',
    'pace': 'mean',          # Average pace
    'season': 'first'
}).reset_index()

game_stats.columns = ['game_id', 'total_offense_plays', 'total_defense_plays',
                      'avg_offense_ppa', 'avg_defense_ppa',
                      'total_offense_ppa', 'total_defense_ppa',
                      'game_pace', 'season']

# Recalculate game pace as total plays / 2 (each team's plays)
game_stats['game_pace'] = game_stats['total_offense_plays'] / 2

print(f"Aggregated game stats: {len(game_stats)} games")

# ============================================================
# MERGE ALL DATA
# ============================================================
print("\n" + "=" * 60)
print("MERGING DATA")
print("=" * 60)

# Merge lines with game stats
merged = lines_df.merge(game_stats, on='game_id', how='left', suffixes=('', '_stats'))

# Merge with game info (week)
merged = merged.merge(games_df, on='game_id', how='left')

# Use the season from lines_df (drop duplicate)
if 'season_stats' in merged.columns:
    merged = merged.drop(columns=['season_stats'])

print(f"Merged records: {len(merged)}")

# Calculate over/under result
merged['ou_result'] = merged.apply(
    lambda row: 'OVER' if row['actual_total'] and row['over_under'] and row['actual_total'] > row['over_under']
    else ('UNDER' if row['actual_total'] and row['over_under'] and row['actual_total'] < row['over_under']
          else ('PUSH' if row['actual_total'] and row['over_under'] and row['actual_total'] == row['over_under']
                else None)),
    axis=1
)

# Calculate over/under margin
merged['ou_margin'] = merged['actual_total'] - merged['over_under']

# ============================================================
# CREATE TEAM-LEVEL PACE STATS
# ============================================================
print("\n" + "=" * 60)
print("CALCULATING TEAM PACE AVERAGES")
print("=" * 60)

# Calculate rolling pace for each team
team_pace = stats_df.groupby('team').agg({
    'pace': 'mean',
    'offense_ppa': 'mean',
    'defense_ppa': 'mean',
    'offense_plays': 'mean'
}).reset_index()

team_pace.columns = ['team', 'avg_pace', 'avg_off_ppa', 'avg_def_ppa', 'avg_plays']
team_pace = team_pace.sort_values('avg_pace', ascending=False)

print(f"\nTop 5 Fastest Teams (by Pace):")
print(f"{'Team':<25} {'Pace':<10} {'Off PPA':<10} {'Plays/Game':<10}")
print("-" * 55)
for _, row in team_pace.head(5).iterrows():
    print(f"{row['team']:<25} {row['avg_pace']:<10.1f} {row['avg_off_ppa']:<10.3f} {row['avg_plays']:<10.1f}")

print(f"\nBottom 5 Slowest Teams (by Pace):")
print(f"{'Team':<25} {'Pace':<10} {'Off PPA':<10} {'Plays/Game':<10}")
print("-" * 55)
for _, row in team_pace.tail(5).iterrows():
    print(f"{row['team']:<25} {row['avg_pace']:<10.1f} {row['avg_off_ppa']:<10.3f} {row['avg_plays']:<10.1f}")

# ============================================================
# ADD HOME/AWAY PACE TO MERGED DATA
# ============================================================
# Create pace lookup
pace_lookup = dict(zip(team_pace['team'], team_pace['avg_pace']))
ppa_off_lookup = dict(zip(team_pace['team'], team_pace['avg_off_ppa']))
ppa_def_lookup = dict(zip(team_pace['team'], team_pace['avg_def_ppa']))

merged['home_avg_pace'] = merged['home_team'].map(pace_lookup)
merged['away_avg_pace'] = merged['away_team'].map(pace_lookup)
merged['combined_pace'] = (merged['home_avg_pace'].fillna(65) + merged['away_avg_pace'].fillna(65)) / 2

merged['home_off_ppa'] = merged['home_team'].map(ppa_off_lookup)
merged['away_off_ppa'] = merged['away_team'].map(ppa_off_lookup)
merged['home_def_ppa'] = merged['home_team'].map(ppa_def_lookup)
merged['away_def_ppa'] = merged['away_team'].map(ppa_def_lookup)

# ============================================================
# SAVE DATA
# ============================================================
print("\n" + "=" * 60)
print("SAVING DATA")
print("=" * 60)

# Select columns for export
export_cols = [
    'game_id', 'season', 'week', 'home_team', 'away_team',
    'home_score', 'away_score', 'actual_total',
    'over_under', 'spread', 'provider',
    'ou_result', 'ou_margin',
    'game_pace', 'total_offense_plays',
    'avg_offense_ppa', 'avg_defense_ppa',
    'home_avg_pace', 'away_avg_pace', 'combined_pace',
    'home_off_ppa', 'away_off_ppa', 'home_def_ppa', 'away_def_ppa'
]

# Filter to columns that exist
export_cols = [c for c in export_cols if c in merged.columns]
export_df = merged[export_cols].copy()

export_df.to_csv('cfb_totals_data.csv', index=False)
print(f"\nSaved to 'cfb_totals_data.csv' ({len(export_df)} records)")

# Also save team pace lookup
team_pace.to_csv('team_pace.csv', index=False)
print(f"Saved to 'team_pace.csv' ({len(team_pace)} teams)")

# ============================================================
# PRINT SAMPLE DATA
# ============================================================
print("\n" + "=" * 60)
print("FIRST 5 ROWS WITH PACE AND OVER/UNDER")
print("=" * 60)

sample = export_df[export_df['over_under'].notna()].head(5)

print(f"\n{'Game':<40} {'O/U':<8} {'Actual':<8} {'Pace':<8} {'Result':<8}")
print("-" * 72)

for _, row in sample.iterrows():
    game = f"{row['away_team'][:15]} @ {row['home_team'][:15]}"
    ou = row['over_under'] if pd.notna(row['over_under']) else 'N/A'
    actual = row['actual_total'] if pd.notna(row['actual_total']) else 'N/A'
    pace = row['combined_pace'] if pd.notna(row['combined_pace']) else 'N/A'
    result = row['ou_result'] if pd.notna(row['ou_result']) else 'N/A'

    ou_str = f"{ou:.1f}" if isinstance(ou, (int, float)) else ou
    actual_str = f"{actual:.0f}" if isinstance(actual, (int, float)) else actual
    pace_str = f"{pace:.1f}" if isinstance(pace, (int, float)) else pace

    print(f"{game:<40} {ou_str:<8} {actual_str:<8} {pace_str:<8} {result:<8}")

# ============================================================
# OVER/UNDER ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("OVER/UNDER ANALYSIS")
print("=" * 60)

valid_ou = export_df[export_df['ou_result'].notna()]

if len(valid_ou) > 0:
    over_count = (valid_ou['ou_result'] == 'OVER').sum()
    under_count = (valid_ou['ou_result'] == 'UNDER').sum()
    push_count = (valid_ou['ou_result'] == 'PUSH').sum()
    total = len(valid_ou)

    print(f"\nOver/Under Results ({total} games):")
    print(f"  OVER:  {over_count} ({over_count/total*100:.1f}%)")
    print(f"  UNDER: {under_count} ({under_count/total*100:.1f}%)")
    print(f"  PUSH:  {push_count} ({push_count/total*100:.1f}%)")

    # Average margin
    avg_margin = valid_ou['ou_margin'].mean()
    print(f"\nAverage O/U Margin: {avg_margin:+.1f} points")

    if avg_margin > 0:
        print("  (Games trending OVER on average)")
    else:
        print("  (Games trending UNDER on average)")

    # Pace correlation
    valid_pace = valid_ou[valid_ou['combined_pace'].notna() & valid_ou['actual_total'].notna()]
    if len(valid_pace) > 10:
        correlation = valid_pace['combined_pace'].corr(valid_pace['actual_total'])
        print(f"\nPace vs Total Correlation: {correlation:.3f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
Data Fetched:
  - Advanced game stats: {len(stats_df)} records
  - Betting lines: {len(lines_df)} games
  - Games with O/U lines: {len(lines_with_ou)}

Files Created:
  - cfb_totals_data.csv: Game-level totals data
  - team_pace.csv: Team average pace lookup

Key Columns:
  - game_pace: Plays per team in game
  - combined_pace: Average of both teams' season pace
  - over_under: Vegas O/U line
  - actual_total: Actual combined score
  - ou_result: OVER/UNDER/PUSH
  - ou_margin: Actual - Line
""")

print("=" * 60)
print("TOTALS DATA FETCH COMPLETE")
print("=" * 60)
