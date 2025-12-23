"""
Opponent-Adjusted EPA Stats Calculator.

Calculates opponent-adjusted offensive EPA by accounting for
the strength of defenses faced. This rewards offenses for
performing well against tough defenses.

Formula:
  Adj_Off_EPA = Raw_Off_EPA - Opponent_Avg_Def_EPA

If opponent defense is above average (negative EPA allowed),
the adjustment adds to the offense's EPA (rewards them).
If opponent defense is below average (positive EPA allowed),
the adjustment subtracts from the offense's EPA.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("OPPONENT-ADJUSTED EPA CALCULATOR")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")
print(f"Seasons: {sorted(df['season'].unique())}")

# ============================================================
# STEP 1: Calculate Average EPA Allowed for Every Defense
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: CALCULATING AVERAGE DEFENSIVE EPA BY TEAM")
print("=" * 60)

# For each team, calculate their average defensive EPA allowed
# (how many EPA points they give up to opposing offenses)

# Home team's defense faces away team's offense
# Away team's defense faces home team's offense

# Collect all defensive performances
def_performances = []

# When team plays at home, their defense faces away offense
home_def = df[['season', 'home_team', 'away_comp_off_ppa']].copy()
home_def.columns = ['season', 'team', 'epa_allowed']

# When team plays away, their defense faces home offense
away_def = df[['season', 'away_team', 'home_comp_off_ppa']].copy()
away_def.columns = ['season', 'team', 'epa_allowed']

# Combine
all_def = pd.concat([home_def, away_def], ignore_index=True)
all_def = all_def.dropna()

# Calculate season average for each team's defense
def_avg = all_def.groupby(['season', 'team'])['epa_allowed'].mean().reset_index()
def_avg.columns = ['season', 'team', 'avg_def_epa_allowed']

print(f"\nDefensive averages calculated for {len(def_avg)} team-seasons")
print(f"\nSample defensive averages (2024):")
sample = def_avg[def_avg['season'] == 2024].sort_values('avg_def_epa_allowed')
print(f"  Best defenses (lowest EPA allowed):")
for _, row in sample.head(5).iterrows():
    print(f"    {row['team']}: {row['avg_def_epa_allowed']:.3f}")
print(f"  Worst defenses (highest EPA allowed):")
for _, row in sample.tail(5).iterrows():
    print(f"    {row['team']}: {row['avg_def_epa_allowed']:.3f}")

# League average by season
league_avg = all_def.groupby('season')['epa_allowed'].mean()
print(f"\nLeague average EPA allowed by season:")
for season, avg in league_avg.items():
    print(f"  {season}: {avg:.3f}")

# ============================================================
# STEP 2: Calculate Opponent-Adjusted EPA for Every Game
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: CALCULATING OPPONENT-ADJUSTED EPA")
print("=" * 60)

# For home offense: adjust based on away team's defensive average
# For away offense: adjust based on home team's defensive average

# Merge away team's defensive average into each game (for home offense adjustment)
df = df.merge(
    def_avg.rename(columns={'team': 'away_team', 'avg_def_epa_allowed': 'away_def_avg'}),
    on=['season', 'away_team'],
    how='left'
)

# Merge home team's defensive average into each game (for away offense adjustment)
df = df.merge(
    def_avg.rename(columns={'team': 'home_team', 'avg_def_epa_allowed': 'home_def_avg'}),
    on=['season', 'home_team'],
    how='left'
)

# Get league average for each season
df['league_avg_def'] = df['season'].map(league_avg)

# Calculate opponent-adjusted EPA
# Adj_Off_EPA = Raw_Off_EPA - (Opponent_Def_Avg - League_Avg)
# This normalizes so average opponent = 0 adjustment

# Home offense adjusted EPA (facing away defense)
df['home_adj_off_epa'] = df['home_comp_off_ppa'] - (df['away_def_avg'] - df['league_avg_def'])

# Away offense adjusted EPA (facing home defense)
df['away_adj_off_epa'] = df['away_comp_off_ppa'] - (df['home_def_avg'] - df['league_avg_def'])

# Also create adjusted defensive EPA (how well defense did vs opponent's adjusted offense)
df['home_adj_def_epa'] = df['home_comp_def_ppa'] - (df['away_def_avg'] - df['league_avg_def'])
df['away_adj_def_epa'] = df['away_comp_def_ppa'] - (df['home_def_avg'] - df['league_avg_def'])

# Net adjusted EPA (schematic mismatch with opponent adjustment)
df['adj_net_epa'] = df['home_adj_off_epa'] - df['away_adj_def_epa']

print(f"\nNew columns created:")
print(f"  - home_adj_off_epa: Home offense adjusted for opponent defense strength")
print(f"  - away_adj_off_epa: Away offense adjusted for opponent defense strength")
print(f"  - home_adj_def_epa: Home defense adjusted")
print(f"  - away_adj_def_epa: Away defense adjusted")
print(f"  - adj_net_epa: Adjusted net EPA (schematic mismatch)")

# Show adjustment impact
print(f"\nAdjustment impact (home offense):")
print(f"  Raw EPA range: {df['home_comp_off_ppa'].min():.3f} to {df['home_comp_off_ppa'].max():.3f}")
print(f"  Adj EPA range: {df['home_adj_off_epa'].min():.3f} to {df['home_adj_off_epa'].max():.3f}")

# Correlation between raw and adjusted
corr = df['home_comp_off_ppa'].corr(df['home_adj_off_epa'])
print(f"  Correlation (raw vs adjusted): {corr:.3f}")

# ============================================================
# STEP 3: Show Example Adjustments
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: EXAMPLE ADJUSTMENTS")
print("=" * 60)

# Find games where adjustment made biggest difference
df['home_adj_diff'] = df['home_adj_off_epa'] - df['home_comp_off_ppa']

print("\nBiggest positive adjustments (offense rewarded for tough opponent):")
top_adj = df.nlargest(5, 'home_adj_diff')[['season', 'week', 'home_team', 'away_team',
                                            'home_comp_off_ppa', 'home_adj_off_epa', 'home_adj_diff']]
for _, row in top_adj.iterrows():
    print(f"  {row['home_team']} vs {row['away_team']} (Week {row['week']}, {row['season']})")
    print(f"    Raw: {row['home_comp_off_ppa']:.3f} -> Adj: {row['home_adj_off_epa']:.3f} (+{row['home_adj_diff']:.3f})")

print("\nBiggest negative adjustments (offense penalized for weak opponent):")
bot_adj = df.nsmallest(5, 'home_adj_diff')[['season', 'week', 'home_team', 'away_team',
                                             'home_comp_off_ppa', 'home_adj_off_epa', 'home_adj_diff']]
for _, row in bot_adj.iterrows():
    print(f"  {row['home_team']} vs {row['away_team']} (Week {row['week']}, {row['season']})")
    print(f"    Raw: {row['home_comp_off_ppa']:.3f} -> Adj: {row['home_adj_off_epa']:.3f} ({row['home_adj_diff']:.3f})")

# ============================================================
# STEP 4: Clean Up and Save
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: SAVING UPDATED DATA")
print("=" * 60)

# Drop helper columns
df = df.drop(columns=['away_def_avg', 'home_def_avg', 'league_avg_def', 'home_adj_diff'], errors='ignore')

# Save
df.to_csv('cfb_data_smart.csv', index=False)
print(f"\nSaved to 'cfb_data_smart.csv'")
print(f"Total columns: {len(df.columns)}")

# List new columns
new_cols = ['home_adj_off_epa', 'away_adj_off_epa', 'home_adj_def_epa', 'away_adj_def_epa', 'adj_net_epa']
print(f"\nNew opponent-adjusted columns added:")
for col in new_cols:
    if col in df.columns:
        print(f"  - {col}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("OPPONENT-ADJUSTED EPA COMPLETE")
print("=" * 60)

print(f"""
Summary:
  - Calculated average defensive EPA allowed for {len(def_avg)} team-seasons
  - Created 5 new opponent-adjusted features
  - Rewards offenses for playing well vs tough defenses
  - Penalizes offenses for padding stats vs weak defenses

New Features:
  1. home_adj_off_epa - Home offense adjusted EPA
  2. away_adj_off_epa - Away offense adjusted EPA
  3. home_adj_def_epa - Home defense adjusted EPA
  4. away_adj_def_epa - Away defense adjusted EPA
  5. adj_net_epa - Adjusted net EPA (schematic mismatch)

Next Step: Run train_ensemble.py to retrain with adjusted stats
""")

print("=" * 60)
print("ADJUSTMENT COMPLETE - READY FOR RETRAINING")
print("=" * 60)
