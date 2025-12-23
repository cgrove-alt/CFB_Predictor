"""
Calculate Dynamic Home Field Advantage (HFA) for CFB Teams.

Computes team-specific HFA by:
1. Calculating average margin at home (Home Points - Away Points)
2. Calculating average margin away (Away Points - Home Points)
3. Raw HFA = Home Margin - Away Margin
4. Regressed HFA = (Raw HFA * Games + 2.5 * 10) / (Games + 10)

Saves 'team_hfa.csv' and updates 'cfb_data_smart.csv'.
"""

import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
# Prior for regression (league average HFA is ~2.5-3 points)
PRIOR_HFA = 2.5
PRIOR_WEIGHT = 10  # Equivalent to 10 games of data at the prior

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("CALCULATE DYNAMIC HOME FIELD ADVANTAGE")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")

# ============================================================
# CALCULATE HOME MARGINS
# ============================================================
print("\n" + "=" * 60)
print("CALCULATING HOME MARGINS")
print("=" * 60)

# Home margin = Home Points - Away Points (already stored in 'Margin')
# Group by home_team
home_stats = df.groupby('home_team').agg({
    'Margin': ['mean', 'count']
}).reset_index()
home_stats.columns = ['team', 'home_margin', 'home_games']

print(f"\nTeams with home games: {len(home_stats)}")
print(f"Avg home margin (all teams): {home_stats['home_margin'].mean():.2f}")

# ============================================================
# CALCULATE AWAY MARGINS
# ============================================================
print("\n" + "=" * 60)
print("CALCULATING AWAY MARGINS")
print("=" * 60)

# Away margin from away team's perspective = Away Points - Home Points = -Margin
# Group by away_team
away_stats = df.groupby('away_team').agg({
    'Margin': ['mean', 'count']
}).reset_index()
away_stats.columns = ['team', 'raw_away_margin', 'away_games']

# Convert to away team's perspective (they want positive = winning)
away_stats['away_margin'] = -away_stats['raw_away_margin']

print(f"\nTeams with away games: {len(away_stats)}")
print(f"Avg away margin (from away perspective): {away_stats['away_margin'].mean():.2f}")

# ============================================================
# MERGE AND CALCULATE RAW HFA
# ============================================================
print("\n" + "=" * 60)
print("CALCULATING RAW HFA")
print("=" * 60)

# Merge home and away stats
hfa_df = home_stats.merge(away_stats[['team', 'away_margin', 'away_games']],
                           on='team', how='outer')

# Fill missing values
hfa_df['home_margin'] = hfa_df['home_margin'].fillna(0)
hfa_df['away_margin'] = hfa_df['away_margin'].fillna(0)
hfa_df['home_games'] = hfa_df['home_games'].fillna(0).astype(int)
hfa_df['away_games'] = hfa_df['away_games'].fillna(0).astype(int)
hfa_df['total_games'] = hfa_df['home_games'] + hfa_df['away_games']

# Calculate Raw HFA = Home Margin - Away Margin
# This represents how much better a team does at home vs away
hfa_df['raw_hfa'] = hfa_df['home_margin'] - hfa_df['away_margin']

print(f"\nRaw HFA Statistics:")
print(f"  Mean: {hfa_df['raw_hfa'].mean():.2f}")
print(f"  Std:  {hfa_df['raw_hfa'].std():.2f}")
print(f"  Max:  {hfa_df['raw_hfa'].max():.2f}")
print(f"  Min:  {hfa_df['raw_hfa'].min():.2f}")

# ============================================================
# REGRESS TO MEAN
# ============================================================
print("\n" + "=" * 60)
print("APPLYING REGRESSION TO MEAN")
print("=" * 60)

print(f"\nRegression Formula:")
print(f"  hfa_rating = (Raw_HFA * Games + {PRIOR_HFA} * {PRIOR_WEIGHT}) / (Games + {PRIOR_WEIGHT})")

# Apply regression formula:
# hfa_rating = (raw_hfa * total_games + PRIOR_HFA * PRIOR_WEIGHT) / (total_games + PRIOR_WEIGHT)
hfa_df['hfa_rating'] = (
    (hfa_df['raw_hfa'] * hfa_df['total_games'] + PRIOR_HFA * PRIOR_WEIGHT) /
    (hfa_df['total_games'] + PRIOR_WEIGHT)
)

print(f"\nRegressed HFA Statistics:")
print(f"  Mean: {hfa_df['hfa_rating'].mean():.2f}")
print(f"  Std:  {hfa_df['hfa_rating'].std():.2f}")
print(f"  Max:  {hfa_df['hfa_rating'].max():.2f}")
print(f"  Min:  {hfa_df['hfa_rating'].min():.2f}")

# Show regression effect
print(f"\nRegression Effect (Raw vs Regressed):")
print(f"  Raw range:      {hfa_df['raw_hfa'].min():.1f} to {hfa_df['raw_hfa'].max():.1f}")
print(f"  Regressed range: {hfa_df['hfa_rating'].min():.1f} to {hfa_df['hfa_rating'].max():.1f}")

# ============================================================
# SAVE TEAM HFA FILE
# ============================================================
print("\n" + "=" * 60)
print("SAVING TEAM HFA FILE")
print("=" * 60)

# Create clean export
hfa_export = hfa_df[['team', 'hfa_rating', 'raw_hfa', 'home_margin', 'away_margin',
                      'home_games', 'away_games', 'total_games']].copy()
hfa_export = hfa_export.sort_values('hfa_rating', ascending=False)
hfa_export.to_csv('team_hfa.csv', index=False)
print(f"\nSaved to 'team_hfa.csv' ({len(hfa_export)} teams)")

# ============================================================
# MERGE BACK TO GAME DATA
# ============================================================
print("\n" + "=" * 60)
print("MERGING HFA TO GAME DATA")
print("=" * 60)

# Create lookup dict
hfa_lookup = dict(zip(hfa_df['team'], hfa_df['hfa_rating']))

# Reload fresh copy to avoid duplicate columns
df = pd.read_csv('cfb_data_smart.csv')

# Remove old HFA columns if they exist
cols_to_drop = [c for c in df.columns if 'hfa' in c.lower()]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"Removed old HFA columns: {cols_to_drop}")

# Map HFA ratings
df['home_team_hfa'] = df['home_team'].map(hfa_lookup).fillna(PRIOR_HFA)
df['away_team_hfa'] = df['away_team'].map(hfa_lookup).fillna(PRIOR_HFA)

# Calculate HFA differential (home advantage minus away team's typical HFA)
df['hfa_diff'] = df['home_team_hfa'] - df['away_team_hfa']

print(f"\nHFA columns added:")
print(f"  home_team_hfa: {df['home_team_hfa'].notna().sum()} values")
print(f"  away_team_hfa: {df['away_team_hfa'].notna().sum()} values")
print(f"  hfa_diff: {df['hfa_diff'].notna().sum()} values")

# ============================================================
# SAVE UPDATED GAME DATA
# ============================================================
print("\n" + "=" * 60)
print("SAVING UPDATED GAME DATA")
print("=" * 60)

df.to_csv('cfb_data_smart.csv', index=False)
print(f"\nSaved to 'cfb_data_smart.csv'")

# ============================================================
# TOP 5 HARDEST PLACES TO PLAY
# ============================================================
print("\n" + "=" * 60)
print("TOP 5 HARDEST PLACES TO PLAY")
print("=" * 60)

# Filter to teams with reasonable sample size
qualified = hfa_df[hfa_df['total_games'] >= 10].copy()

top_5 = qualified.nlargest(5, 'hfa_rating')
print(f"\n{'Rank':<6} {'Team':<25} {'HFA':>8} {'Raw':>10} {'Games':>8}")
print("-" * 60)

for rank, (_, row) in enumerate(top_5.iterrows(), 1):
    print(f"{rank:<6} {row['team']:<25} {row['hfa_rating']:>+8.2f} {row['raw_hfa']:>+10.2f} {int(row['total_games']):>8}")

# ============================================================
# BOTTOM 5 (WEAKEST HOME ADVANTAGE)
# ============================================================
print("\n" + "=" * 60)
print("BOTTOM 5 (WEAKEST HOME FIELD ADVANTAGE)")
print("=" * 60)

bottom_5 = qualified.nsmallest(5, 'hfa_rating')
print(f"\n{'Rank':<6} {'Team':<25} {'HFA':>8} {'Raw':>10} {'Games':>8}")
print("-" * 60)

for rank, (_, row) in enumerate(bottom_5.iterrows(), 1):
    print(f"{rank:<6} {row['team']:<25} {row['hfa_rating']:>+8.2f} {row['raw_hfa']:>+10.2f} {int(row['total_games']):>8}")

# ============================================================
# NOTABLE POWER 5 TEAMS
# ============================================================
print("\n" + "=" * 60)
print("NOTABLE POWER 5 TEAMS")
print("=" * 60)

notable_teams = ['Georgia', 'Alabama', 'Ohio State', 'Michigan', 'Texas',
                 'LSU', 'Oregon', 'Clemson', 'Florida State', 'Penn State',
                 'Notre Dame', 'USC', 'Oklahoma', 'Tennessee', 'Florida']

print(f"\n{'Team':<20} {'HFA':>10} {'Raw HFA':>10} {'Games':>8}")
print("-" * 50)

for team in notable_teams:
    team_data = hfa_df[hfa_df['team'] == team]
    if len(team_data) > 0:
        row = team_data.iloc[0]
        print(f"{team:<20} {row['hfa_rating']:>+10.2f} {row['raw_hfa']:>+10.2f} {int(row['total_games']):>8}")
    else:
        print(f"{team:<20} {'N/A':>10}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
Dynamic Home Field Advantage Calculation:
  - Total teams: {len(hfa_df)}
  - Qualified teams (>=10 games): {len(qualified)}
  - Prior HFA: {PRIOR_HFA} points
  - Prior weight: {PRIOR_WEIGHT} games

Regression Formula:
  hfa_rating = (raw_hfa * games + {PRIOR_HFA} * {PRIOR_WEIGHT}) / (games + {PRIOR_WEIGHT})

This formula:
  - Teams with few games regress heavily toward {PRIOR_HFA}
  - Teams with many games keep their raw HFA
  - Prevents extreme outliers from small samples

Files Updated:
  - team_hfa.csv: Team HFA ratings lookup
  - cfb_data_smart.csv: Added home_team_hfa, away_team_hfa, hfa_diff
""")

print("=" * 60)
print("HFA CALCULATION COMPLETE")
print("=" * 60)
