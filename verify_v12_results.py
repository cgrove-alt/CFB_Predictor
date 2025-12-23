"""
Verify V12 Results with Strict Walk-Forward Testing.

The previous results showed 81% win rate which is suspicious.
Let's verify with proper week-by-week walk-forward testing.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VERIFYING V12 SPREAD ERROR MODEL")
print("Strict Week-by-Week Walk-Forward Testing")
print("=" * 70)

# Load data
df = pd.read_csv('cfb_data_smart.csv')
df_vegas = df[df['vegas_spread'].notna()].copy()
df_vegas = df_vegas.sort_values(['season', 'week']).reset_index(drop=True)

# Calculate spread error
df_vegas['spread_error'] = df_vegas['Margin'] - (-df_vegas['vegas_spread'])

# Calculate features (same as V12)
df_vegas['line_movement'] = df_vegas['vegas_spread'] - df_vegas['spread_open'].fillna(df_vegas['vegas_spread'])
df_vegas['large_favorite'] = (df_vegas['vegas_spread'] < -14).astype(int)
df_vegas['large_underdog'] = (df_vegas['vegas_spread'] > 14).astype(int)
df_vegas['close_game'] = (abs(df_vegas['vegas_spread']) < 7).astype(int)
df_vegas['rest_spread_interaction'] = df_vegas['rest_diff'] * abs(df_vegas['vegas_spread']) / 10

# Momentum features
team_streaks = {}
home_streak = []
away_streak = []

for idx, row in df_vegas.iterrows():
    home = row['home_team']
    away = row['away_team']
    season = row['season']

    h_streak = 0
    a_streak = 0
    if home in team_streaks and team_streaks[home][0] == season:
        h_streak = team_streaks[home][1]
    if away in team_streaks and team_streaks[away][0] == season:
        a_streak = team_streaks[away][1]

    home_streak.append(h_streak)
    away_streak.append(a_streak)

    if pd.notna(row['Margin']):
        home_won = row['Margin'] > 0
        if home_won:
            new_h_streak = max(1, h_streak + 1) if h_streak >= 0 else 1
            new_a_streak = min(-1, a_streak - 1) if a_streak <= 0 else -1
        else:
            new_h_streak = min(-1, h_streak - 1) if h_streak <= 0 else -1
            new_a_streak = max(1, a_streak + 1) if a_streak >= 0 else 1
        team_streaks[home] = (season, new_h_streak)
        team_streaks[away] = (season, new_a_streak)

df_vegas['home_streak'] = home_streak
df_vegas['away_streak'] = away_streak
df_vegas['streak_diff'] = df_vegas['home_streak'] - df_vegas['away_streak']

# ATS features
team_ats = {}
home_ats_rate = []
away_ats_rate = []

for idx, row in df_vegas.iterrows():
    home = row['home_team']
    away = row['away_team']

    if home in team_ats and len(team_ats[home]) >= 3:
        home_ats_rate.append(np.mean(team_ats[home][-10:]))
    else:
        home_ats_rate.append(0.5)

    if away in team_ats and len(team_ats[away]) >= 3:
        away_ats_rate.append(np.mean(team_ats[away][-10:]))
    else:
        away_ats_rate.append(0.5)

    if pd.notna(row.get('spread_error')):
        home_covered = row['spread_error'] > 0
        if home not in team_ats:
            team_ats[home] = []
        if away not in team_ats:
            team_ats[away] = []
        team_ats[home].append(1 if home_covered else 0)
        team_ats[away].append(0 if home_covered else 1)

df_vegas['home_ats_rate'] = home_ats_rate
df_vegas['away_ats_rate'] = away_ats_rate
df_vegas['ats_diff'] = df_vegas['home_ats_rate'] - df_vegas['away_ats_rate']

# Define features (NO LEAKAGE - all features are from BEFORE the game)
FEATURES = [
    'home_comp_off_ppa', 'away_comp_off_ppa',
    'home_comp_def_ppa', 'away_comp_def_ppa',
    'home_comp_pass_ppa', 'away_comp_pass_ppa',
    'home_comp_rush_ppa', 'away_comp_rush_ppa',
    'home_streak', 'away_streak', 'streak_diff',
    'home_ats_rate', 'away_ats_rate', 'ats_diff',
    'line_movement',
    'large_favorite', 'large_underdog',
    'close_game',
    'rest_diff', 'rest_spread_interaction',
    'home_team_hfa', 'hfa_diff',
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',
]

available = [f for f in FEATURES if f in df_vegas.columns]

# Filter to valid data
df_valid = df_vegas[df_vegas['spread_error'].notna()].copy()

print(f"\nTotal games with spreads: {len(df_valid)}")

# ============================================================
# STRICT WALK-FORWARD TEST ON 2025
# ============================================================
print("\n" + "=" * 70)
print("WALK-FORWARD TEST: 2025 (Week by Week)")
print("=" * 70)

# Get 2025 weeks
weeks_2025 = sorted(df_valid[df_valid['season'] == 2025]['week'].unique())
print(f"2025 weeks: {weeks_2025}")

all_predictions = []
all_actuals = []
all_vegas = []
week_results = []

for week in weeks_2025:
    # Training: ALL data strictly BEFORE this week
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

    # Train model (fresh each week - no future data)
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

    # Final margin prediction
    pred_margin = -vegas_spread + pred_error
    vegas_margin = -vegas_spread

    # Calculate MAEs
    model_mae = mean_absolute_error(actual_margin, pred_margin)
    vegas_mae = mean_absolute_error(actual_margin, vegas_margin)

    # Betting simulation for this week
    n_bets = 0
    wins = 0
    for pe, ae in zip(pred_error, y_test):
        if abs(pe) < 3.0:  # Threshold
            continue
        n_bets += 1
        if (pe > 0 and ae > 0) or (pe < 0 and ae < 0):
            wins += 1

    win_rate = wins / n_bets if n_bets > 0 else 0

    week_results.append({
        'week': week,
        'games': len(test_data),
        'model_mae': model_mae,
        'vegas_mae': vegas_mae,
        'bets': n_bets,
        'wins': wins,
        'win_rate': win_rate
    })

    all_predictions.extend(pred_margin.tolist())
    all_actuals.extend(actual_margin.tolist())
    all_vegas.extend(vegas_margin.tolist())

    beat = "✓" if model_mae < vegas_mae else "✗"
    print(f"Week {week:2d}: {len(test_data):3d} games | Model: {model_mae:.2f} | Vegas: {vegas_mae:.2f} {beat} | Bets: {n_bets}, Win: {win_rate:.1%}")

# ============================================================
# AGGREGATE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("AGGREGATE RESULTS (2025 Walk-Forward)")
print("=" * 70)

final_model_mae = mean_absolute_error(all_actuals, all_predictions)
final_vegas_mae = mean_absolute_error(all_actuals, all_vegas)

wf_df = pd.DataFrame(week_results)
total_bets = wf_df['bets'].sum()
total_wins = wf_df['wins'].sum()
overall_win_rate = total_wins / total_bets if total_bets > 0 else 0

weeks_beat_vegas = (wf_df['model_mae'] < wf_df['vegas_mae']).sum()

print(f"\nFinal MAE Comparison:")
print(f"  Model MAE: {final_model_mae:.2f}")
print(f"  Vegas MAE: {final_vegas_mae:.2f}")
print(f"  Difference: {final_vegas_mae - final_model_mae:+.2f}")

print(f"\nWeeks beating Vegas: {weeks_beat_vegas}/{len(wf_df)}")

print(f"\nBetting Simulation (threshold=3.0):")
print(f"  Total bets: {total_bets}")
print(f"  Total wins: {total_wins}")
print(f"  Win rate: {overall_win_rate:.1%}")
print(f"  Break-even: 52.4%")

if overall_win_rate > 0.524:
    profit_units = total_wins * 0.91 - (total_bets - total_wins) * 1.0
    print(f"  Estimated profit: {profit_units:+.1f} units")
else:
    print(f"  NOT PROFITABLE at standard -110 lines")

# ============================================================
# REALITY CHECK
# ============================================================
print("\n" + "=" * 70)
print("REALITY CHECK")
print("=" * 70)

print(f"""
If the model is truly predictive, we should see:
1. Model MAE < Vegas MAE consistently
2. Win rate > 52.4% for profitability
3. Consistency across weeks

Results:
- Model beats Vegas in {weeks_beat_vegas}/{len(wf_df)} weeks
- Overall win rate: {overall_win_rate:.1%}
- Model MAE: {final_model_mae:.2f} vs Vegas: {final_vegas_mae:.2f}
""")

if final_model_mae < final_vegas_mae and overall_win_rate > 0.524:
    print("✓ MODEL APPEARS TO HAVE PREDICTIVE VALUE")
else:
    print("✗ MODEL DOES NOT RELIABLY BEAT VEGAS")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
