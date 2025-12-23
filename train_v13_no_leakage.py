"""
Train V13 Model - NO DATA LEAKAGE.

CRITICAL: Previous models used home_comp_off_ppa which is from THAT game.
This version ONLY uses features known BEFORE the game.

Safe features:
- Elo ratings (pre-game)
- Last 5 game averages (rolling, calculated before game)
- Home field advantage (historical)
- Rest days (scheduling data)
- Vegas spread (known before game)
- Momentum/streaks (calculated from past results)
- ATS rates (calculated from past results)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TRAIN V13 NO LEAKAGE MODEL")
print("Only Using Features Known BEFORE the Game")
print("=" * 70)

# Load data
df = pd.read_csv('cfb_data_smart.csv')
df_vegas = df[df['vegas_spread'].notna()].copy()
df_vegas = df_vegas.sort_values(['season', 'week']).reset_index(drop=True)

# Calculate spread error (target)
df_vegas['spread_error'] = df_vegas['Margin'] - (-df_vegas['vegas_spread'])

print(f"\nTotal games with Vegas spreads: {len(df_vegas)}")

# ============================================================
# SAFE FEATURES (Known Before Game)
# ============================================================
print("\n" + "=" * 70)
print("SAFE FEATURES (No Leakage)")
print("=" * 70)

# 1. Elo (pre-game)
print("\n1. Pre-game Elo ratings")

# 2. Rolling averages (last 5 games - these ARE safe)
print("2. Last 5 game averages (rolling)")

# 3. Home field advantage (historical)
print("3. Historical HFA")

# 4. Rest days (known from schedule)
print("4. Rest days")

# 5. Vegas features
print("5. Vegas spread, line movement")
df_vegas['line_movement'] = df_vegas['vegas_spread'] - df_vegas['spread_open'].fillna(df_vegas['vegas_spread'])
df_vegas['large_favorite'] = (df_vegas['vegas_spread'] < -14).astype(int)
df_vegas['large_underdog'] = (df_vegas['vegas_spread'] > 14).astype(int)
df_vegas['close_game'] = (abs(df_vegas['vegas_spread']) < 7).astype(int)

# 6. Momentum (calculated from past results only)
print("6. Win/loss streaks")
team_streaks = {}
home_streak = []
away_streak = []

for idx, row in df_vegas.iterrows():
    home = row['home_team']
    away = row['away_team']
    season = row['season']

    h_streak = team_streaks.get(home, (0, 0))
    a_streak = team_streaks.get(away, (0, 0))

    # Use streak only if same season
    h_val = h_streak[1] if h_streak[0] == season else 0
    a_val = a_streak[1] if a_streak[0] == season else 0

    home_streak.append(h_val)
    away_streak.append(a_val)

    # Update AFTER recording (so we don't use current game)
    if pd.notna(row['Margin']):
        home_won = row['Margin'] > 0
        if home_won:
            new_h = max(1, h_val + 1) if h_val >= 0 else 1
            new_a = min(-1, a_val - 1) if a_val <= 0 else -1
        else:
            new_h = min(-1, h_val - 1) if h_val <= 0 else -1
            new_a = max(1, a_val + 1) if a_val >= 0 else 1
        team_streaks[home] = (season, new_h)
        team_streaks[away] = (season, new_a)

df_vegas['home_streak'] = home_streak
df_vegas['away_streak'] = away_streak
df_vegas['streak_diff'] = df_vegas['home_streak'] - df_vegas['away_streak']

# 7. ATS performance (calculated from past games)
print("7. Historical ATS rates")
team_ats = {}
home_ats = []
away_ats = []

for idx, row in df_vegas.iterrows():
    home = row['home_team']
    away = row['away_team']

    # Get rates BEFORE this game
    if home in team_ats and len(team_ats[home]) >= 3:
        home_ats.append(np.mean(team_ats[home][-10:]))
    else:
        home_ats.append(0.5)

    if away in team_ats and len(team_ats[away]) >= 3:
        away_ats.append(np.mean(team_ats[away][-10:]))
    else:
        away_ats.append(0.5)

    # Update AFTER recording
    if pd.notna(row.get('spread_error')):
        home_covered = row['spread_error'] > 0
        if home not in team_ats:
            team_ats[home] = []
        if away not in team_ats:
            team_ats[away] = []
        team_ats[home].append(1 if home_covered else 0)
        team_ats[away].append(0 if home_covered else 1)

df_vegas['home_ats'] = home_ats
df_vegas['away_ats'] = away_ats
df_vegas['ats_diff'] = df_vegas['home_ats'] - df_vegas['away_ats']

# ============================================================
# DEFINE SAFE FEATURES
# ============================================================
SAFE_FEATURES = [
    # Elo (pre-game) - SAFE
    'home_pregame_elo', 'away_pregame_elo', 'elo_diff',

    # Rolling averages (last 5 games) - SAFE
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',

    # HFA (historical) - SAFE
    'home_team_hfa', 'hfa_diff',

    # Rest (scheduling) - SAFE
    'rest_diff',

    # Vegas features - SAFE
    'line_movement',
    'large_favorite', 'large_underdog',
    'close_game',

    # Momentum (from past results) - SAFE
    'home_streak', 'away_streak', 'streak_diff',

    # ATS (from past results) - SAFE
    'home_ats', 'away_ats', 'ats_diff',
]

available = [f for f in SAFE_FEATURES if f in df_vegas.columns]
print(f"\nUsing {len(available)} safe features:")
for i, f in enumerate(available, 1):
    print(f"  {i:2d}. {f}")

# ============================================================
# WALK-FORWARD VALIDATION (2025)
# ============================================================
print("\n" + "=" * 70)
print("WALK-FORWARD VALIDATION (2025)")
print("=" * 70)

df_valid = df_vegas[df_vegas['spread_error'].notna()].copy()
weeks_2025 = sorted(df_valid[df_valid['season'] == 2025]['week'].unique())

print(f"Testing weeks: {weeks_2025}")

results = []
all_pred = []
all_actual = []
all_vegas = []

for week in weeks_2025:
    # Train on ALL data before this week
    train_mask = (df_valid['season'] < 2025) | ((df_valid['season'] == 2025) & (df_valid['week'] < week))
    test_mask = (df_valid['season'] == 2025) & (df_valid['week'] == week)

    train_data = df_valid[train_mask]
    test_data = df_valid[test_mask]

    if len(train_data) < 100 or len(test_data) == 0:
        continue

    X_train = train_data[available].fillna(0)
    y_train = train_data['spread_error']
    X_test = test_data[available].fillna(0)
    y_test = test_data['spread_error']

    vegas_spread = test_data['vegas_spread']
    actual_margin = test_data['Margin']

    # Train fresh model
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predict spread error
    pred_error = model.predict(X_test)

    # Final margin = Vegas + adjustment
    pred_margin = -vegas_spread + pred_error
    vegas_margin = -vegas_spread

    model_mae = mean_absolute_error(actual_margin, pred_margin)
    vegas_mae = mean_absolute_error(actual_margin, vegas_margin)

    # Betting sim
    bets = 0
    wins = 0
    for pe, ae in zip(pred_error, y_test):
        if abs(pe) < 3.0:
            continue
        bets += 1
        if (pe > 0 and ae > 0) or (pe < 0 and ae < 0):
            wins += 1

    win_rate = wins / bets if bets > 0 else 0

    results.append({
        'week': week,
        'games': len(test_data),
        'model_mae': model_mae,
        'vegas_mae': vegas_mae,
        'bets': bets,
        'wins': wins,
        'win_rate': win_rate
    })

    all_pred.extend(pred_margin.tolist())
    all_actual.extend(actual_margin.tolist())
    all_vegas.extend(vegas_margin.tolist())

    beat = "✓" if model_mae < vegas_mae else "✗"
    print(f"Week {week:2d}: {len(test_data):3d} games | Model: {model_mae:.2f} | Vegas: {vegas_mae:.2f} {beat} | Bets: {bets}, Win: {win_rate:.1%}")

# ============================================================
# AGGREGATE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("AGGREGATE RESULTS (NO LEAKAGE)")
print("=" * 70)

final_model_mae = mean_absolute_error(all_actual, all_pred)
final_vegas_mae = mean_absolute_error(all_actual, all_vegas)

res_df = pd.DataFrame(results)
total_bets = res_df['bets'].sum()
total_wins = res_df['wins'].sum()
overall_win_rate = total_wins / total_bets if total_bets > 0 else 0
weeks_beat_vegas = (res_df['model_mae'] < res_df['vegas_mae']).sum()

print(f"\nMAE Comparison:")
print(f"  Model MAE: {final_model_mae:.2f}")
print(f"  Vegas MAE: {final_vegas_mae:.2f}")
print(f"  Difference: {final_vegas_mae - final_model_mae:+.2f}")

print(f"\nWeeks beating Vegas: {weeks_beat_vegas}/{len(res_df)}")

print(f"\nBetting (threshold=3.0):")
print(f"  Total bets: {total_bets}")
print(f"  Wins: {total_wins}")
print(f"  Win rate: {overall_win_rate:.1%}")
print(f"  Break-even: 52.4%")

if overall_win_rate > 0.524:
    profit = total_wins * 0.91 - (total_bets - total_wins) * 1.0
    print(f"  Profit: {profit:+.1f} units")
    profitable = True
else:
    print(f"  NOT PROFITABLE")
    profitable = False

# ============================================================
# TRAIN FINAL MODEL
# ============================================================
print("\n" + "=" * 70)
print("TRAINING FINAL MODEL")
print("=" * 70)

# Train on all available data
X_all = df_valid[df_valid['season'].isin([2022, 2023, 2024, 2025])][available].fillna(0)
y_all = df_valid[df_valid['season'].isin([2022, 2023, 2024, 2025])]['spread_error']

final_model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_all, y_all)

# Save
joblib.dump(final_model, 'cfb_spread_error_v13.pkl')
print("Saved to 'cfb_spread_error_v13.pkl'")

# Feature importance
importances = final_model.feature_importances_
imp_df = pd.DataFrame({
    'feature': available,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nFeature Importance (no leakage):")
for _, row in imp_df.head(10).iterrows():
    print(f"  {row['importance']:.4f}  {row['feature']}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("V13 NO LEAKAGE MODEL - SUMMARY")
print("=" * 70)

print(f"""
RESULTS (No Data Leakage):
  Model MAE:  {final_model_mae:.2f}
  Vegas MAE:  {final_vegas_mae:.2f}
  Difference: {final_vegas_mae - final_model_mae:+.2f}

BETTING SIMULATION:
  Win Rate: {overall_win_rate:.1%}
  Profitable: {'YES' if profitable else 'NO'}

FEATURES USED ({len(available)}):
  - Pre-game Elo ratings
  - Last 5 game rolling averages
  - Historical HFA
  - Rest days
  - Vegas line movement
  - Momentum (win streaks)
  - ATS history

IMPORTANT NOTE:
  Previous results with 74% win rate were due to DATA LEAKAGE!
  The comp_ppa features were from THAT game, not historical.
""")

if final_model_mae < final_vegas_mae:
    print("✓ Model beats Vegas (no leakage)")
else:
    print("✗ Model does not beat Vegas")

print("\n" + "=" * 70)
print("V13 COMPLETE")
print("=" * 70)
