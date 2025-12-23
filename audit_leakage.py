"""
Data Leakage Audit for CFB Betting Model.

Checks if rolling averages for a game include data from that game itself.
This would be data leakage - using future data to predict.

Test: For a specific team's Week 10 game:
1. Get the rolling average stored in the data
2. Manually calculate rolling average using only games BEFORE Week 10
3. If they differ, we have leakage
"""

import pandas as pd
import numpy as np

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 60)
print("DATA LEAKAGE AUDIT")
print("=" * 60)

print("\nLoading cfb_data_smart.csv...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games: {len(df)}")

# ============================================================
# TEST CONFIGURATION
# ============================================================
TEST_TEAM = "Georgia"
TEST_SEASON = 2024
TEST_WEEK = 10

print(f"\nTest Configuration:")
print(f"  Team: {TEST_TEAM}")
print(f"  Season: {TEST_SEASON}")
print(f"  Week: {TEST_WEEK}")

# ============================================================
# FIND THE TEST GAME
# ============================================================
print("\n" + "=" * 60)
print("FINDING TEST GAME")
print("=" * 60)

# Find the game where test team plays in the specified week
home_game = df[(df['home_team'] == TEST_TEAM) &
               (df['season'] == TEST_SEASON) &
               (df['week'] == TEST_WEEK)]

away_game = df[(df['away_team'] == TEST_TEAM) &
               (df['season'] == TEST_SEASON) &
               (df['week'] == TEST_WEEK)]

if len(home_game) > 0:
    test_game = home_game.iloc[0]
    is_home = True
    stored_avg = test_game['home_last5_score_avg']
    stored_def = test_game['home_last5_defense_avg']
    game_points = test_game['home_points']
    game_allowed = test_game['away_points']
    opponent = test_game['away_team']
elif len(away_game) > 0:
    test_game = away_game.iloc[0]
    is_home = False
    stored_avg = test_game['away_last5_score_avg']
    stored_def = test_game['away_last5_defense_avg']
    game_points = test_game['away_points']
    game_allowed = test_game['home_points']
    opponent = test_game['home_team']
else:
    print(f"ERROR: No game found for {TEST_TEAM} in Week {TEST_WEEK}, {TEST_SEASON}")
    exit(1)

print(f"\nFound game: {TEST_TEAM} vs {opponent}")
print(f"  {TEST_TEAM} scored: {game_points}")
print(f"  {TEST_TEAM} allowed: {game_allowed}")
print(f"  Playing as: {'HOME' if is_home else 'AWAY'}")
print(f"\nStored rolling averages:")
print(f"  Last 5 Score Avg: {stored_avg:.2f}" if pd.notna(stored_avg) else "  Last 5 Score Avg: NaN")
print(f"  Last 5 Defense Avg: {stored_def:.2f}" if pd.notna(stored_def) else "  Last 5 Defense Avg: NaN")

# ============================================================
# MANUALLY CALCULATE ROLLING AVERAGE (GAMES BEFORE WEEK 10)
# ============================================================
print("\n" + "=" * 60)
print("MANUAL CALCULATION (Games BEFORE Week 10)")
print("=" * 60)

# Get all games for this team BEFORE the test week
home_games = df[(df['home_team'] == TEST_TEAM) &
                (df['season'] == TEST_SEASON) &
                (df['week'] < TEST_WEEK)][['week', 'home_points', 'away_points']].copy()
home_games.columns = ['week', 'scored', 'allowed']

away_games = df[(df['away_team'] == TEST_TEAM) &
                (df['season'] == TEST_SEASON) &
                (df['week'] < TEST_WEEK)][['week', 'away_points', 'home_points']].copy()
away_games.columns = ['week', 'scored', 'allowed']

all_prior_games = pd.concat([home_games, away_games])
all_prior_games = all_prior_games.sort_values('week', ascending=False)

print(f"\nGames BEFORE Week {TEST_WEEK}:")
for _, g in all_prior_games.iterrows():
    print(f"  Week {int(g['week'])}: Scored {int(g['scored'])}, Allowed {int(g['allowed'])}")

# Take last 5 games
last_5 = all_prior_games.head(5)
print(f"\nLast 5 games used for average:")
for _, g in last_5.iterrows():
    print(f"  Week {int(g['week'])}: Scored {int(g['scored'])}, Allowed {int(g['allowed'])}")

if len(last_5) > 0:
    manual_score_avg = last_5['scored'].mean()
    manual_def_avg = last_5['allowed'].mean()
else:
    manual_score_avg = np.nan
    manual_def_avg = np.nan

print(f"\nManual calculation (excluding Week {TEST_WEEK}):")
print(f"  Score Avg: {manual_score_avg:.2f}" if pd.notna(manual_score_avg) else "  Score Avg: NaN")
print(f"  Defense Avg: {manual_def_avg:.2f}" if pd.notna(manual_def_avg) else "  Defense Avg: NaN")

# ============================================================
# COMPARE AND DETECT LEAKAGE
# ============================================================
print("\n" + "=" * 60)
print("LEAKAGE DETECTION")
print("=" * 60)

# Check if Week 10 game is included in the calculation
# If stored avg includes Week 10, it would be different from manual calculation

leakage_detected = False

if pd.notna(stored_avg) and pd.notna(manual_score_avg):
    diff = abs(stored_avg - manual_score_avg)
    print(f"\nScore Average:")
    print(f"  Stored in data: {stored_avg:.2f}")
    print(f"  Manual (pre-game): {manual_score_avg:.2f}")
    print(f"  Difference: {diff:.4f}")

    if diff > 0.01:
        print(f"  STATUS: MISMATCH DETECTED")
        leakage_detected = True
    else:
        print(f"  STATUS: CLEAN")

if pd.notna(stored_def) and pd.notna(manual_def_avg):
    diff = abs(stored_def - manual_def_avg)
    print(f"\nDefense Average:")
    print(f"  Stored in data: {stored_def:.2f}")
    print(f"  Manual (pre-game): {manual_def_avg:.2f}")
    print(f"  Difference: {diff:.4f}")

    if diff > 0.01:
        print(f"  STATUS: MISMATCH DETECTED")
        leakage_detected = True
    else:
        print(f"  STATUS: CLEAN")

# ============================================================
# ADDITIONAL CHECK: Does including Week 10 match?
# ============================================================
print("\n" + "=" * 60)
print("ADDITIONAL CHECK: What if Week 10 IS included?")
print("=" * 60)

# Add Week 10 game to the calculation
week_10_game = pd.DataFrame({'week': [TEST_WEEK], 'scored': [game_points], 'allowed': [game_allowed]})
all_with_week_10 = pd.concat([week_10_game, all_prior_games])
all_with_week_10 = all_with_week_10.sort_values('week', ascending=False)

last_5_with_leak = all_with_week_10.head(5)
if len(last_5_with_leak) > 0:
    leaked_score_avg = last_5_with_leak['scored'].mean()
    leaked_def_avg = last_5_with_leak['allowed'].mean()

    print(f"\nIf Week {TEST_WEEK} game WAS included (LEAKY calculation):")
    print(f"  Score Avg: {leaked_score_avg:.2f}")
    print(f"  Defense Avg: {leaked_def_avg:.2f}")

    # Check which one matches the stored value
    if pd.notna(stored_avg):
        if abs(stored_avg - leaked_score_avg) < abs(stored_avg - manual_score_avg):
            print(f"\n  ALERT: Stored value matches LEAKY calculation!")
            leakage_detected = True
        else:
            print(f"\n  GOOD: Stored value matches CLEAN calculation")

# ============================================================
# FINAL VERDICT
# ============================================================
print("\n" + "=" * 60)
print("FINAL VERDICT")
print("=" * 60)

if leakage_detected:
    print("\n🚨 LEAK DETECTED 🚨")
    print("The rolling averages may include data from the current game.")
    print("This is a data leakage issue that could inflate model accuracy.")
else:
    print("\n✅ CLEAN ✅")
    print("Rolling averages correctly exclude the current game's data.")
    print("No data leakage detected.")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
