"""
Filter Garbage Time Plays for CFB Betting Model.

Fetches play-level data and calculates "Competitive" stats by filtering out:
- Plays where score difference > 28 points in the 2nd half (3rd/4th quarter)
- OR Plays where Win Probability > 95%

Creates new 'clean' EPA and Success Rate columns.
"""

import pandas as pd
import numpy as np
import cfbd
from config import CFBD_API_KEY
import time

# ============================================================
# CONFIGURATION
# ============================================================
GARBAGE_TIME_SCORE_DIFF = 28  # Points difference threshold
GARBAGE_TIME_PERIOD = 3  # 3rd quarter or later (periods 3 & 4)
WIN_PROB_THRESHOLD = 0.95  # 95% win probability threshold

# ============================================================
# API SETUP
# ============================================================
configuration = cfbd.Configuration()
configuration.access_token = CFBD_API_KEY
api_client = cfbd.ApiClient(configuration)
plays_api = cfbd.PlaysApi(api_client)
metrics_api = cfbd.MetricsApi(api_client)

# ============================================================
# GARBAGE TIME DEFINITIONS
# ============================================================
def is_garbage_time(period, offense_score, defense_score, win_prob=None):
    """
    Determine if a play is in garbage time.

    Garbage time criteria:
    1. Score difference > 28 points in 2nd half (period >= 3)
    2. OR Win probability > 95%
    """
    score_diff = abs(offense_score - defense_score)

    # Second half with large lead
    if period >= GARBAGE_TIME_PERIOD and score_diff > GARBAGE_TIME_SCORE_DIFF:
        return True

    # Win probability check (if available)
    if win_prob is not None:
        if win_prob > WIN_PROB_THRESHOLD or win_prob < (1 - WIN_PROB_THRESHOLD):
            return True

    return False


# ============================================================
# FETCH PLAYS AND CALCULATE COMPETITIVE STATS
# ============================================================
def fetch_competitive_stats(year, week=None):
    """
    Fetch play-by-play data and calculate competitive-only stats.

    Returns dict of {game_id: {team: {stats}}}
    """
    print(f"  Fetching plays for {year}" + (f" week {week}" if week else "..."))

    try:
        # Fetch plays (may need to loop through weeks)
        all_plays = []

        if week:
            plays = plays_api.get_plays(year=year, week=week)
            all_plays.extend(plays)
        else:
            # Fetch all regular season weeks (1-15)
            for w in range(1, 16):
                try:
                    plays = plays_api.get_plays(year=year, week=w)
                    all_plays.extend(plays)
                    print(f"    Week {w}: {len(plays)} plays")
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print(f"    Week {w}: Error - {e}")
                    continue

        print(f"  Total plays fetched: {len(all_plays)}")

        if not all_plays:
            return {}

        # Convert to records
        plays_data = []
        for p in all_plays:
            score_diff = abs((p.offense_score or 0) - (p.defense_score or 0))

            plays_data.append({
                'game_id': p.game_id,
                'offense': p.offense,
                'defense': p.defense,
                'period': p.period,
                'offense_score': p.offense_score or 0,
                'defense_score': p.defense_score or 0,
                'score_diff': score_diff,
                'ppa': p.ppa,
                'play_type': p.play_type,
                'yards_gained': p.yards_gained,
                'down': p.down,
                'distance': p.distance,
            })

        plays_df = pd.DataFrame(plays_data)

        # Filter out garbage time
        plays_df['is_garbage'] = plays_df.apply(
            lambda row: is_garbage_time(
                row['period'],
                row['offense_score'],
                row['defense_score']
            ), axis=1
        )

        total_plays = len(plays_df)
        garbage_plays = plays_df['is_garbage'].sum()
        competitive_plays = total_plays - garbage_plays

        print(f"  Total plays: {total_plays}")
        print(f"  Garbage time plays: {garbage_plays} ({garbage_plays/total_plays*100:.1f}%)")
        print(f"  Competitive plays: {competitive_plays} ({competitive_plays/total_plays*100:.1f}%)")

        # Filter to competitive plays only
        comp_df = plays_df[~plays_df['is_garbage']].copy()

        # Calculate stats per team per game
        # Success = gained 50% of needed yards on 1st down,
        #           70% on 2nd down, 100% on 3rd/4th down
        def is_successful_play(row):
            if row['down'] == 1:
                return row['yards_gained'] >= 0.5 * row['distance']
            elif row['down'] == 2:
                return row['yards_gained'] >= 0.7 * row['distance']
            else:  # 3rd or 4th down
                return row['yards_gained'] >= row['distance']

        comp_df['is_success'] = comp_df.apply(is_successful_play, axis=1)

        # Aggregate by game and team (offense perspective)
        game_stats = comp_df.groupby(['game_id', 'offense']).agg({
            'ppa': 'mean',
            'is_success': 'mean',
            'yards_gained': 'mean'
        }).reset_index()

        game_stats.columns = ['game_id', 'team', 'comp_epa', 'comp_success_rate', 'comp_ypp']

        return game_stats

    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()


# ============================================================
# ALTERNATIVE: USE API's BUILT-IN GARBAGE TIME FILTER
# ============================================================
def fetch_ppa_excluding_garbage(year):
    """
    Use the API's built-in exclude_garbage_time parameter for PPA.
    This is a backup if play-by-play fails.
    """
    print(f"  Fetching PPA (excluding garbage time) for {year}...")

    try:
        # Get PPA by game with garbage time excluded
        ppa_data = metrics_api.get_predicted_points_added_by_game(
            year=year,
            exclude_garbage_time=True
        )

        records = []
        for game in ppa_data:
            records.append({
                'game_id': game.game_id,
                'team': game.team,
                'opponent': game.opponent,
                'comp_off_ppa': game.offense.overall if game.offense else None,
                'comp_def_ppa': game.defense.overall if game.defense else None,
                'comp_off_passing_ppa': game.offense.passing if game.offense else None,
                'comp_off_rushing_ppa': game.offense.rushing if game.offense else None,
            })

        print(f"    Found {len(records)} game records")
        return pd.DataFrame(records)

    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()


# ============================================================
# MAIN EXECUTION
# ============================================================
print("=" * 60)
print("GARBAGE TIME FILTER")
print("=" * 60)

print(f"\nGarbage Time Definition:")
print(f"  - Score diff > {GARBAGE_TIME_SCORE_DIFF} points in 2nd half (Period >= {GARBAGE_TIME_PERIOD})")
print(f"  - OR Win Probability > {WIN_PROB_THRESHOLD*100:.0f}%")

# ============================================================
# APPROACH 1: API's Built-in Filter (PPA only)
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 1: API PPA with exclude_garbage_time=True")
print("=" * 60)

all_ppa = []
years = [2022, 2023, 2024]

for year in years:
    ppa_df = fetch_ppa_excluding_garbage(year)
    if len(ppa_df) > 0:
        ppa_df['season'] = year
        all_ppa.append(ppa_df)
    time.sleep(1)  # Rate limiting

if all_ppa:
    ppa_combined = pd.concat(all_ppa, ignore_index=True)
    print(f"\nTotal PPA records (garbage-filtered): {len(ppa_combined)}")
else:
    ppa_combined = pd.DataFrame()
    print("\nNo PPA data fetched via API filter")

# ============================================================
# APPROACH 2: Manual Play-by-Play Filtering
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 2: Manual Play-by-Play Filtering")
print("=" * 60)

all_stats = []

for year in years:
    print(f"\n{year}:")
    stats_df = fetch_competitive_stats(year)
    if len(stats_df) > 0:
        stats_df['season'] = year
        all_stats.append(stats_df)
    time.sleep(1)

if all_stats:
    stats_combined = pd.concat(all_stats, ignore_index=True)
    print(f"\nTotal competitive stats records: {len(stats_combined)}")
else:
    stats_combined = pd.DataFrame()
    print("\nNo play-by-play stats calculated")

# ============================================================
# MERGE WITH EXISTING DATA
# ============================================================
print("\n" + "=" * 60)
print("MERGING WITH EXISTING DATA")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Games loaded: {len(df)}")

original_cols = len(df.columns)

# Merge PPA data (API approach)
if len(ppa_combined) > 0:
    # Home team stats
    home_ppa = ppa_combined.rename(columns={
        'game_id': 'id',
        'comp_off_ppa': 'home_comp_off_ppa',
        'comp_def_ppa': 'home_comp_def_ppa',
        'comp_off_passing_ppa': 'home_comp_pass_ppa',
        'comp_off_rushing_ppa': 'home_comp_rush_ppa',
        'team': 'home_team'
    })[['id', 'home_team', 'home_comp_off_ppa', 'home_comp_def_ppa',
        'home_comp_pass_ppa', 'home_comp_rush_ppa']]

    # Away team stats
    away_ppa = ppa_combined.rename(columns={
        'game_id': 'id',
        'comp_off_ppa': 'away_comp_off_ppa',
        'comp_def_ppa': 'away_comp_def_ppa',
        'comp_off_passing_ppa': 'away_comp_pass_ppa',
        'comp_off_rushing_ppa': 'away_comp_rush_ppa',
        'team': 'away_team'
    })[['id', 'away_team', 'away_comp_off_ppa', 'away_comp_def_ppa',
        'away_comp_pass_ppa', 'away_comp_rush_ppa']]

    # Merge
    df = df.merge(home_ppa, on=['id', 'home_team'], how='left')
    df = df.merge(away_ppa, on=['id', 'away_team'], how='left')

    ppa_coverage = df['home_comp_off_ppa'].notna().sum()
    print(f"  PPA stats merged: {ppa_coverage} games ({ppa_coverage/len(df)*100:.1f}%)")

# Merge play-by-play stats (manual approach)
if len(stats_combined) > 0:
    # Home team stats
    home_stats = stats_combined.rename(columns={
        'game_id': 'id',
        'team': 'home_team',
        'comp_epa': 'home_comp_epa',
        'comp_success_rate': 'home_comp_success',
        'comp_ypp': 'home_comp_ypp'
    })[['id', 'home_team', 'home_comp_epa', 'home_comp_success', 'home_comp_ypp']]

    # Away team stats
    away_stats = stats_combined.rename(columns={
        'game_id': 'id',
        'team': 'away_team',
        'comp_epa': 'away_comp_epa',
        'comp_success_rate': 'away_comp_success',
        'comp_ypp': 'away_comp_ypp'
    })[['id', 'away_team', 'away_comp_epa', 'away_comp_success', 'away_comp_ypp']]

    # Merge
    df = df.merge(home_stats, on=['id', 'home_team'], how='left')
    df = df.merge(away_stats, on=['id', 'away_team'], how='left')

    stats_coverage = df['home_comp_epa'].notna().sum()
    print(f"  Play stats merged: {stats_coverage} games ({stats_coverage/len(df)*100:.1f}%)")

# ============================================================
# SAVE UPDATED DATA
# ============================================================
print("\n" + "=" * 60)
print("SAVING UPDATED DATA")
print("=" * 60)

new_cols = len(df.columns) - original_cols
print(f"\nNew columns added: {new_cols}")

# List new columns
new_col_names = [c for c in df.columns if 'comp_' in c]
print(f"New column names: {new_col_names}")

df.to_csv('cfb_data_smart.csv', index=False)
print(f"\nSaved to 'cfb_data_smart.csv'")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
Garbage Time Filter Applied:
  - Score diff > {GARBAGE_TIME_SCORE_DIFF} pts in 2nd half
  - Win Probability > {WIN_PROB_THRESHOLD*100:.0f}%

New Competitive Stats Added:
  - home/away_comp_off_ppa: Offensive PPA (garbage-free)
  - home/away_comp_def_ppa: Defensive PPA (garbage-free)
  - home/away_comp_pass_ppa: Passing PPA (garbage-free)
  - home/away_comp_rush_ppa: Rushing PPA (garbage-free)
  - home/away_comp_epa: EPA from competitive plays only
  - home/away_comp_success: Success rate from competitive plays
  - home/away_comp_ypp: Yards per play from competitive plays

These 'clean' stats exclude garbage time, giving better
predictive signal for close games.
""")

print("=" * 60)
print("GARBAGE TIME FILTER COMPLETE")
print("=" * 60)
