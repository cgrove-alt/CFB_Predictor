"""
Feature Interactions Calculator for CFB Spread Prediction.

Creates interaction features that capture non-linear relationships:
1. Pass_Efficiency_Diff: Difference in passing efficiency (PPA)
2. Rest_Diff: Home rest advantage (fatigue factor)
3. Elo_Diff: Power rating differential (talent/strength proxy)
4. Matchup_Advantage: Offensive efficiency vs opponent defense
5. EPA_x_Elo: Efficiency weighted by team strength
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("FEATURE INTERACTIONS CALCULATOR")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")
print(f"Total columns before: {len(df.columns)}")

# ============================================================
# STEP 1: REST DIFFERENTIAL (Fatigue Factor)
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: REST_DIFF (Fatigue Factor)")
print("=" * 60)

# Rest_Diff = home_rest_days - away_rest_days
# Positive = Home team is more rested
if 'rest_diff' not in df.columns:
    df['rest_diff'] = df['home_rest_days'] - df['away_rest_days']
    print(f"\nCreated 'rest_diff' = home_rest_days - away_rest_days")
else:
    print("\n'rest_diff' already exists")

print(f"  Range: {df['rest_diff'].min():.1f} to {df['rest_diff'].max():.1f}")
print(f"  Mean: {df['rest_diff'].mean():.2f}")
print(f"  NaN count: {df['rest_diff'].isna().sum()}")

# ============================================================
# STEP 2: ELO DIFFERENTIAL (Talent/Strength Proxy)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: ELO_DIFF (Talent/Strength Proxy)")
print("=" * 60)

# Elo_Diff = home_pregame_elo - away_pregame_elo
# Positive = Home team is stronger
if 'elo_diff' not in df.columns:
    df['elo_diff'] = df['home_pregame_elo'] - df['away_pregame_elo']
    print(f"\nCreated 'elo_diff' = home_pregame_elo - away_pregame_elo")
else:
    print("\n'elo_diff' already exists")

print(f"  Range: {df['elo_diff'].min():.1f} to {df['elo_diff'].max():.1f}")
print(f"  Mean: {df['elo_diff'].mean():.2f}")
print(f"  NaN count: {df['elo_diff'].isna().sum()}")

# ============================================================
# STEP 3: PASS EFFICIENCY DIFFERENTIAL
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: PASS_EFFICIENCY_DIFF (Passing Game Mismatch)")
print("=" * 60)

# Pass_Efficiency_Diff = home_comp_pass_ppa - away_comp_pass_ppa
# Positive = Home team has better passing efficiency
if 'pass_efficiency_diff' not in df.columns:
    df['pass_efficiency_diff'] = df['home_comp_pass_ppa'] - df['away_comp_pass_ppa']
    print(f"\nCreated 'pass_efficiency_diff' = home_comp_pass_ppa - away_comp_pass_ppa")
else:
    print("\n'pass_efficiency_diff' already exists")

print(f"  Range: {df['pass_efficiency_diff'].min():.3f} to {df['pass_efficiency_diff'].max():.3f}")
print(f"  Mean: {df['pass_efficiency_diff'].mean():.4f}")
print(f"  NaN count: {df['pass_efficiency_diff'].isna().sum()}")

# ============================================================
# STEP 4: MATCHUP ADVANTAGE (Offensive Efficiency vs Defense)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: MATCHUP_ADVANTAGE (Offense vs Opponent Defense)")
print("=" * 60)

# Home matchup: home_comp_off_ppa vs away_comp_def_ppa
# Away matchup: away_comp_off_ppa vs home_comp_def_ppa
# Net matchup advantage = (home_matchup - away_matchup) / 2

if 'matchup_advantage' not in df.columns:
    home_matchup = df['home_comp_off_ppa'] - df['away_comp_def_ppa']
    away_matchup = df['away_comp_off_ppa'] - df['home_comp_def_ppa']
    df['matchup_advantage'] = (home_matchup - away_matchup) / 2
    print(f"\nCreated 'matchup_advantage' = (home_matchup - away_matchup) / 2")
    print("  Where home_matchup = home_off_ppa - away_def_ppa")
    print("  Where away_matchup = away_off_ppa - home_def_ppa")
else:
    print("\n'matchup_advantage' already exists")

print(f"  Range: {df['matchup_advantage'].min():.3f} to {df['matchup_advantage'].max():.3f}")
print(f"  Mean: {df['matchup_advantage'].mean():.4f}")
print(f"  NaN count: {df['matchup_advantage'].isna().sum()}")

# ============================================================
# STEP 5: EPA x ELO INTERACTION (Efficiency weighted by strength)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: EPA_ELO_INTERACTION (Efficiency x Strength)")
print("=" * 60)

# Normalized EPA * Normalized Elo
# This captures: efficient teams that are also strong overall
if 'epa_elo_interaction' not in df.columns:
    # Normalize EPA diff and Elo diff to similar scales
    epa_diff = df['home_comp_epa'] - df['away_comp_epa']
    elo_normalized = df['elo_diff'] / 100  # Scale down Elo (ranges in hundreds)

    df['epa_elo_interaction'] = epa_diff * elo_normalized
    print(f"\nCreated 'epa_elo_interaction' = epa_diff * (elo_diff / 100)")
else:
    print("\n'epa_elo_interaction' already exists")

print(f"  Range: {df['epa_elo_interaction'].min():.3f} to {df['epa_elo_interaction'].max():.3f}")
print(f"  Mean: {df['epa_elo_interaction'].mean():.4f}")
print(f"  NaN count: {df['epa_elo_interaction'].isna().sum()}")

# ============================================================
# STEP 6: OFFENSIVE SUCCESS DIFFERENTIAL
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: SUCCESS_DIFF (Play Success Rate)")
print("=" * 60)

# Success rate diff = (home_off_pass_success + home_off_rush_success) / 2
#                   - (away_off_pass_success + away_off_rush_success) / 2
if 'success_diff' not in df.columns:
    # Check if columns exist
    if 'home_off_pass_success' in df.columns and 'home_off_rush_success' in df.columns:
        home_success = (df['home_off_pass_success'] + df.get('home_off_rush_success', 0)) / 2
        away_success = (df['away_off_pass_success'] + df.get('away_off_rush_success', 0)) / 2
        df['success_diff'] = home_success - away_success
        print(f"\nCreated 'success_diff' = avg(home_success) - avg(away_success)")
    else:
        # Fallback: use pass success only
        df['success_diff'] = df['home_off_pass_success'] - df['away_off_pass_success']
        print(f"\nCreated 'success_diff' = home_off_pass_success - away_off_pass_success")
else:
    print("\n'success_diff' already exists")

print(f"  Range: {df['success_diff'].min():.3f} to {df['success_diff'].max():.3f}")
print(f"  Mean: {df['success_diff'].mean():.4f}")
print(f"  NaN count: {df['success_diff'].isna().sum()}")

# ============================================================
# SUMMARY OF NEW FEATURES
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY OF INTERACTION FEATURES")
print("=" * 60)

new_features = [
    'rest_diff',           # Fatigue factor
    'elo_diff',            # Talent/strength proxy
    'pass_efficiency_diff', # Passing game mismatch
    'matchup_advantage',   # Offense vs opponent defense
    'epa_elo_interaction', # Efficiency x strength
    'success_diff'         # Play success rate differential
]

print(f"\nNew interaction features ({len(new_features)}):")
for i, feat in enumerate(new_features, 1):
    if feat in df.columns:
        non_null = df[feat].notna().sum()
        print(f"  {i}. {feat}: {non_null} non-null values")
    else:
        print(f"  {i}. {feat}: NOT CREATED")

# ============================================================
# SAVE DATA
# ============================================================
print("\n" + "=" * 60)
print("SAVING UPDATED DATA")
print("=" * 60)

df.to_csv('cfb_data_smart.csv', index=False)
print(f"\nSaved to 'cfb_data_smart.csv'")
print(f"Total columns after: {len(df.columns)}")

# ============================================================
# CORRELATION WITH MARGIN
# ============================================================
print("\n" + "=" * 60)
print("CORRELATION WITH MARGIN (Predictive Power)")
print("=" * 60)

print(f"\n{'Feature':<25} {'Correlation':>12}")
print("-" * 40)
for feat in new_features:
    if feat in df.columns:
        corr = df[feat].corr(df['Margin'])
        print(f"{feat:<25} {corr:>+12.3f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FEATURE INTERACTIONS COMPLETE")
print("=" * 60)

print(f"""
Summary:
  - Added {len(new_features)} interaction features
  - Total columns: {len(df.columns)}

New Features:
  1. rest_diff: Home rest advantage (fatigue)
  2. elo_diff: Power rating differential (talent)
  3. pass_efficiency_diff: Passing game mismatch
  4. matchup_advantage: Offense vs opponent defense
  5. epa_elo_interaction: Efficiency x strength
  6. success_diff: Play success rate differential

These capture non-linear relationships that single features miss!

Next Step: Run train_pro_stacking.py to learn these interactions
""")

print("=" * 60)
print("INTERACTIONS CALCULATED - READY FOR RETRAINING")
print("=" * 60)
