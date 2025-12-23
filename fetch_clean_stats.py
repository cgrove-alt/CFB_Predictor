"""
Fetch Clean Stats for CFB Betting Model.

Calculates 'Clean EPA' by adjusting for garbage time using:
1. API's built-in exclude_garbage_time parameter for PPA
2. Blowout flag (Score Diff > 28) for manual adjustment
3. Comparison of raw vs clean stats

Also shows the difference for blowout games like Georgia vs FSU 2023.
"""

import pandas as pd
import numpy as np
import cfbd
from config import CFBD_API_KEY
import time

# ============================================================
# CONFIGURATION
# ============================================================
BLOWOUT_THRESHOLD = 28  # Score difference for blowout flag

# ============================================================
# API SETUP
# ============================================================
configuration = cfbd.Configuration()
configuration.access_token = CFBD_API_KEY
api_client = cfbd.ApiClient(configuration)
metrics_api = cfbd.MetricsApi(api_client)
games_api = cfbd.GamesApi(api_client)

# ============================================================
# FETCH RAW AND CLEAN PPA
# ============================================================
def fetch_ppa_both(year):
    """
    Fetch PPA both with and without garbage time filter.
    Returns comparison data.
    """
    print(f"\nFetching PPA for {year}...")

    try:
        # Raw PPA (includes garbage time)
        print(f"  Fetching RAW PPA (with garbage time)...")
        raw_ppa = metrics_api.get_predicted_points_added_by_game(
            year=year,
            exclude_garbage_time=False
        )
        print(f"    Got {len(raw_ppa)} records")

        # Clean PPA (excludes garbage time)
        print(f"  Fetching CLEAN PPA (without garbage time)...")
        clean_ppa = metrics_api.get_predicted_points_added_by_game(
            year=year,
            exclude_garbage_time=True
        )
        print(f"    Got {len(clean_ppa)} records")

        # Convert to DataFrames
        raw_records = []
        for game in raw_ppa:
            raw_records.append({
                'game_id': game.game_id,
                'team': game.team,
                'opponent': game.opponent,
                'raw_off_ppa': game.offense.overall if game.offense else None,
                'raw_def_ppa': game.defense.overall if game.defense else None,
                'raw_pass_ppa': game.offense.passing if game.offense else None,
                'raw_rush_ppa': game.offense.rushing if game.offense else None,
            })

        clean_records = []
        for game in clean_ppa:
            clean_records.append({
                'game_id': game.game_id,
                'team': game.team,
                'clean_off_ppa': game.offense.overall if game.offense else None,
                'clean_def_ppa': game.defense.overall if game.defense else None,
                'clean_pass_ppa': game.offense.passing if game.offense else None,
                'clean_rush_ppa': game.offense.rushing if game.offense else None,
            })

        raw_df = pd.DataFrame(raw_records)
        clean_df = pd.DataFrame(clean_records)

        # Merge raw and clean
        merged = raw_df.merge(
            clean_df,
            on=['game_id', 'team'],
            how='outer'
        )

        # Calculate garbage time adjustment
        merged['garbage_adj_off'] = merged['raw_off_ppa'] - merged['clean_off_ppa']
        merged['garbage_adj_def'] = merged['raw_def_ppa'] - merged['clean_def_ppa']

        merged['season'] = year

        return merged

    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()


# ============================================================
# FETCH GAME SCORES FOR BLOWOUT FLAG
# ============================================================
def fetch_game_scores(year):
    """Fetch final scores to calculate blowout flag."""
    print(f"\nFetching game scores for {year}...")

    try:
        games = games_api.get_games(year=year)

        records = []
        for game in games:
            if game.home_points is not None and game.away_points is not None:
                score_diff = abs(game.home_points - game.away_points)
                records.append({
                    'game_id': game.id,
                    'home_team': game.home_team,
                    'away_team': game.away_team,
                    'home_points': game.home_points,
                    'away_points': game.away_points,
                    'score_diff': score_diff,
                    'is_blowout': score_diff > BLOWOUT_THRESHOLD
                })

        print(f"  Got {len(records)} games with scores")
        return pd.DataFrame(records)

    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()


# ============================================================
# MAIN EXECUTION
# ============================================================
print("=" * 60)
print("FETCH CLEAN STATS")
print("=" * 60)

print(f"\nBlowout Threshold: Score Diff > {BLOWOUT_THRESHOLD}")

# ============================================================
# FETCH DATA FOR ALL YEARS
# ============================================================
all_ppa = []
all_scores = []
years = [2022, 2023, 2024]

for year in years:
    # Fetch PPA (raw and clean)
    ppa_df = fetch_ppa_both(year)
    if len(ppa_df) > 0:
        all_ppa.append(ppa_df)

    # Fetch game scores
    scores_df = fetch_game_scores(year)
    if len(scores_df) > 0:
        scores_df['season'] = year
        all_scores.append(scores_df)

    time.sleep(1)  # Rate limiting

# Combine all years
if all_ppa:
    ppa_combined = pd.concat(all_ppa, ignore_index=True)
    print(f"\nTotal PPA records: {len(ppa_combined)}")
else:
    ppa_combined = pd.DataFrame()
    print("\nNo PPA data fetched")

if all_scores:
    scores_combined = pd.concat(all_scores, ignore_index=True)
    print(f"Total game scores: {len(scores_combined)}")
else:
    scores_combined = pd.DataFrame()
    print("No score data fetched")

# ============================================================
# MERGE PPA WITH BLOWOUT FLAGS
# ============================================================
print("\n" + "=" * 60)
print("MERGING DATA")
print("=" * 60)

if len(ppa_combined) > 0 and len(scores_combined) > 0:
    # Merge PPA with scores (for home team)
    ppa_with_scores = ppa_combined.merge(
        scores_combined[['game_id', 'home_team', 'away_team', 'score_diff', 'is_blowout']],
        on='game_id',
        how='left'
    )

    # Identify if team is home or away
    ppa_with_scores['is_home'] = ppa_with_scores['team'] == ppa_with_scores['home_team']

    print(f"\nMerged records: {len(ppa_with_scores)}")
    print(f"Blowout games: {ppa_with_scores['is_blowout'].sum()}")

# ============================================================
# LOAD AND UPDATE cfb_data_smart.csv
# ============================================================
print("\n" + "=" * 60)
print("UPDATING cfb_data_smart.csv")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Games loaded: {len(df)}")
original_cols = len(df.columns)

# Calculate blowout flag from existing data
df['score_diff'] = abs(df['home_points'] - df['away_points'])
df['is_blowout'] = df['score_diff'] > BLOWOUT_THRESHOLD

blowout_count = df['is_blowout'].sum()
print(f"\nBlowout games identified: {blowout_count} ({blowout_count/len(df)*100:.1f}%)")

# Prepare PPA data for merge
if len(ppa_combined) > 0:
    # Home team clean stats
    home_ppa = ppa_combined[['game_id', 'team', 'clean_off_ppa', 'clean_def_ppa',
                             'raw_off_ppa', 'raw_def_ppa', 'garbage_adj_off', 'garbage_adj_def']].copy()
    home_ppa = home_ppa.rename(columns={
        'game_id': 'id',
        'team': 'home_team',
        'clean_off_ppa': 'home_clean_off_ppa',
        'clean_def_ppa': 'home_clean_def_ppa',
        'raw_off_ppa': 'home_raw_off_ppa',
        'raw_def_ppa': 'home_raw_def_ppa',
        'garbage_adj_off': 'home_garbage_adj_off',
        'garbage_adj_def': 'home_garbage_adj_def'
    })

    # Away team clean stats
    away_ppa = ppa_combined[['game_id', 'team', 'clean_off_ppa', 'clean_def_ppa',
                             'raw_off_ppa', 'raw_def_ppa', 'garbage_adj_off', 'garbage_adj_def']].copy()
    away_ppa = away_ppa.rename(columns={
        'game_id': 'id',
        'team': 'away_team',
        'clean_off_ppa': 'away_clean_off_ppa',
        'clean_def_ppa': 'away_clean_def_ppa',
        'raw_off_ppa': 'away_raw_off_ppa',
        'raw_def_ppa': 'away_raw_def_ppa',
        'garbage_adj_off': 'away_garbage_adj_off',
        'garbage_adj_def': 'away_garbage_adj_def'
    })

    # Merge
    df = df.merge(home_ppa, on=['id', 'home_team'], how='left')
    df = df.merge(away_ppa, on=['id', 'away_team'], how='left')

    coverage = df['home_clean_off_ppa'].notna().sum()
    print(f"\nClean PPA coverage: {coverage} games ({coverage/len(df)*100:.1f}%)")

# ============================================================
# SAVE UPDATED DATA
# ============================================================
print("\n" + "=" * 60)
print("SAVING")
print("=" * 60)

new_cols = len(df.columns) - original_cols
print(f"\nNew columns added: {new_cols}")

# List new columns
new_col_names = [c for c in df.columns if 'clean' in c.lower() or 'raw' in c.lower() or 'garbage' in c.lower() or 'blowout' in c.lower()]
print(f"New columns: {new_col_names}")

df.to_csv('cfb_data_smart.csv', index=False)
print(f"\nSaved to 'cfb_data_smart.csv'")

# ============================================================
# EXAMPLE: GEORGIA vs FSU 2023
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE: BLOWOUT GAME ANALYSIS")
print("=" * 60)

# Find Georgia vs FSU 2023 (Georgia won 63-3)
example = df[(df['season'] == 2023) &
             (((df['home_team'] == 'Georgia') & (df['away_team'] == 'Florida State')) |
              ((df['home_team'] == 'Florida State') & (df['away_team'] == 'Georgia')))]

if len(example) > 0:
    game = example.iloc[0]
    print(f"\nGame: {game['away_team']} @ {game['home_team']}")
    print(f"Score: {game['home_team']} {game['home_points']:.0f} - {game['away_team']} {game['away_points']:.0f}")
    print(f"Score Diff: {game['score_diff']:.0f}")
    print(f"Is Blowout: {game['is_blowout']}")

    print(f"\n{game['home_team']} (Home) EPA:")
    if pd.notna(game.get('home_raw_off_ppa')):
        print(f"  Raw Offense PPA:   {game['home_raw_off_ppa']:.3f}")
        print(f"  Clean Offense PPA: {game['home_clean_off_ppa']:.3f}")
        print(f"  Garbage Adjustment: {game['home_garbage_adj_off']:.3f}")
    else:
        print("  (PPA data not available)")

    print(f"\n{game['away_team']} (Away) EPA:")
    if pd.notna(game.get('away_raw_off_ppa')):
        print(f"  Raw Offense PPA:   {game['away_raw_off_ppa']:.3f}")
        print(f"  Clean Offense PPA: {game['away_clean_off_ppa']:.3f}")
        print(f"  Garbage Adjustment: {game['away_garbage_adj_off']:.3f}")
    else:
        print("  (PPA data not available)")
else:
    print("\nGeorgia vs FSU 2023 not found in data.")
    print("Looking for other blowout examples...")

    # Find any blowout game with PPA data
    blowouts = df[(df['is_blowout'] == True) &
                  (df['home_raw_off_ppa'].notna()) &
                  (df['season'] == 2023)].head(3)

    if len(blowouts) > 0:
        print(f"\nFound {len(blowouts)} blowout examples:")
        for _, game in blowouts.iterrows():
            print(f"\n{'='*50}")
            print(f"Game: {game['away_team']} @ {game['home_team']}")
            print(f"Score: {game['home_points']:.0f} - {game['away_points']:.0f} (Diff: {game['score_diff']:.0f})")

            print(f"\n{game['home_team']} EPA:")
            print(f"  Raw Offense:   {game['home_raw_off_ppa']:.3f}")
            print(f"  Clean Offense: {game['home_clean_off_ppa']:.3f}")
            print(f"  Adjustment:    {game['home_garbage_adj_off']:.3f}")

# ============================================================
# OVERALL STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("OVERALL GARBAGE TIME IMPACT")
print("=" * 60)

if 'home_garbage_adj_off' in df.columns:
    valid_data = df[df['home_garbage_adj_off'].notna()]

    if len(valid_data) > 0:
        print(f"\nAnalyzing {len(valid_data)} games with PPA data:")

        # Overall adjustment stats
        print(f"\nGarbage Time Adjustment (Offense PPA):")
        print(f"  Mean: {valid_data['home_garbage_adj_off'].mean():.4f}")
        print(f"  Std:  {valid_data['home_garbage_adj_off'].std():.4f}")
        print(f"  Max:  {valid_data['home_garbage_adj_off'].max():.4f}")
        print(f"  Min:  {valid_data['home_garbage_adj_off'].min():.4f}")

        # Blowout vs close games
        blowout_adj = valid_data[valid_data['is_blowout']]['home_garbage_adj_off'].mean()
        close_adj = valid_data[~valid_data['is_blowout']]['home_garbage_adj_off'].mean()

        print(f"\nAverage Adjustment by Game Type:")
        print(f"  Blowout games (>28 pts): {blowout_adj:.4f}")
        print(f"  Close games (<=28 pts):  {close_adj:.4f}")
        print(f"  Difference:              {blowout_adj - close_adj:.4f}")

print("\n" + "=" * 60)
print("CLEAN STATS FETCH COMPLETE")
print("=" * 60)
