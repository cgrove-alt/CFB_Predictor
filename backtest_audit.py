"""
Walk-Forward Backtest Audit for CFB Betting Model.

This script performs a realistic backtest by:
1. Training only on data BEFORE each prediction week
2. Predicting games for each week starting Week 6, 2024
3. Comparing predictions to actual results and Vegas lines
4. Calculating ROI assuming -110 odds
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import cfbd
from config import CFBD_API_KEY

# ============================================================
# CONFIGURATION
# ============================================================
EDGE_THRESHOLD = 4.0  # Minimum edge to place a bet
JUICE = -110  # Standard -110 odds

# Features available in cfb_data_smart.csv
FEATURE_COLS = [
    'home_pregame_elo', 'away_pregame_elo',
    'home_last5_score_avg', 'away_last5_score_avg',
    'home_last5_defense_avg', 'away_last5_defense_avg',
]

# ============================================================
# API SETUP
# ============================================================
configuration = cfbd.Configuration()
configuration.access_token = CFBD_API_KEY
api_client = cfbd.ApiClient(configuration)
betting_api = cfbd.BettingApi(api_client)

# ============================================================
# FETCH HISTORICAL BETTING LINES
# ============================================================
def fetch_betting_lines(year):
    """Fetch betting lines for a given year from CFBD API."""
    print(f"  Fetching betting lines for {year}...")
    try:
        lines = betting_api.get_lines(year=year)
        lines_data = []
        for game in lines:
            if game.lines:
                # Get consensus or first available line
                for line in game.lines:
                    if line.spread:
                        lines_data.append({
                            'id': game.id,
                            'home_team': game.home_team,
                            'away_team': game.away_team,
                            'spread_line': float(line.spread),
                            'provider': line.provider
                        })
                        break  # Use first available spread
        return lines_data
    except Exception as e:
        print(f"    Error fetching {year}: {e}")
        return []

# ============================================================
# BETTING MATH
# ============================================================
def calculate_payout(odds):
    """Calculate profit on a $100 bet at given odds."""
    if odds < 0:
        return 100 / (abs(odds) / 100)  # -110 => $90.91 profit
    else:
        return odds  # +110 => $110 profit

def bet_result(predicted_margin, vegas_spread, actual_margin):
    """
    Determine if a bet won, lost, or pushed.

    predicted_margin: Model's predicted home margin
    vegas_spread: Vegas spread (negative = home favored)
    actual_margin: Actual home margin

    Returns: 'win', 'loss', or 'push'
    """
    # Determine which side we bet
    edge = predicted_margin - (-vegas_spread)  # Model thinks home will do better than Vegas

    if edge > 0:
        # Bet HOME to cover
        result = actual_margin + vegas_spread  # Positive = home covered
    else:
        # Bet AWAY to cover
        result = -(actual_margin + vegas_spread)  # Positive = away covered

    if result > 0:
        return 'win'
    elif result < 0:
        return 'loss'
    else:
        return 'push'

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("WALK-FORWARD BACKTEST AUDIT")
print("=" * 70)

print("\nLoading game data...")
df = pd.read_csv('cfb_data_smart.csv')
print(f"Total games loaded: {len(df)}")

# Fetch betting lines for 2024
print("\nFetching betting lines from API...")
lines_2024 = fetch_betting_lines(2024)
lines_2023 = fetch_betting_lines(2023)
lines_2022 = fetch_betting_lines(2022)

all_lines = lines_2022 + lines_2023 + lines_2024
lines_df = pd.DataFrame(all_lines)
print(f"Total betting lines fetched: {len(lines_df)}")

# Merge betting lines with game data
if len(lines_df) > 0:
    # Merge on id
    df = df.merge(lines_df[['id', 'spread_line']], on='id', how='left')
    print(f"Games with spread data: {df['spread_line'].notna().sum()}")

# Sort by season and week
df = df.sort_values(['season', 'week']).reset_index(drop=True)

# Filter to games with all features and spread data
df_valid = df.dropna(subset=FEATURE_COLS + ['spread_line', 'Margin'])
print(f"Games with complete data: {len(df_valid)}")

# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("WALK-FORWARD VALIDATION (Starting Week 6, 2024)")
print("=" * 70)

# Get 2024 weeks starting from week 6
weeks_2024 = sorted(df_valid[df_valid['season'] == 2024]['week'].unique())
test_weeks = [w for w in weeks_2024 if w >= 6]

print(f"Test weeks: {test_weeks}")

# Track results
weekly_results = []
all_bets = []

for week in test_weeks:
    # Training data: All games BEFORE this week
    train_mask = (df_valid['season'] < 2024) | \
                 ((df_valid['season'] == 2024) & (df_valid['week'] < week))

    # Test data: Games in this week
    test_mask = (df_valid['season'] == 2024) & (df_valid['week'] == week)

    train_df = df_valid[train_mask]
    test_df = df_valid[test_mask]

    if len(test_df) == 0:
        continue

    # Prepare training data
    X_train = train_df[FEATURE_COLS]
    y_train = train_df['Margin']

    # Train model
    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Prepare test data
    X_test = test_df[FEATURE_COLS]

    # Predict
    predictions = model.predict(X_test)

    # Analyze bets for this week
    week_wins = 0
    week_losses = 0
    week_pushes = 0
    week_profit = 0
    week_bets = 0

    for i, (idx, row) in enumerate(test_df.iterrows()):
        predicted_margin = predictions[i]
        vegas_spread = row['spread_line']
        actual_margin = row['Margin']

        # Calculate edge
        edge = predicted_margin - (-vegas_spread)

        # Only bet if edge exceeds threshold
        if abs(edge) >= EDGE_THRESHOLD:
            week_bets += 1

            result = bet_result(predicted_margin, vegas_spread, actual_margin)

            if edge > 0:
                bet_side = 'HOME'
                bet_team = row['home_team']
            else:
                bet_side = 'AWAY'
                bet_team = row['away_team']

            profit = 0
            if result == 'win':
                week_wins += 1
                profit = calculate_payout(JUICE)
                week_profit += profit
            elif result == 'loss':
                week_losses += 1
                profit = -100
                week_profit -= 100
            else:
                week_pushes += 1

            all_bets.append({
                'season': 2024,
                'week': week,
                'game': f"{row['away_team']} @ {row['home_team']}",
                'bet_side': bet_side,
                'bet_team': bet_team,
                'edge': edge,
                'predicted': predicted_margin,
                'vegas': vegas_spread,
                'actual': actual_margin,
                'result': result,
                'profit': profit
            })

    # Record weekly results
    week_record = f"{week_wins}-{week_losses}" + (f"-{week_pushes}" if week_pushes > 0 else "")
    win_rate = week_wins / (week_wins + week_losses) * 100 if (week_wins + week_losses) > 0 else 0

    weekly_results.append({
        'week': week,
        'games': len(test_df),
        'bets': week_bets,
        'wins': week_wins,
        'losses': week_losses,
        'pushes': week_pushes,
        'record': week_record,
        'win_rate': win_rate,
        'profit': week_profit,
        'roi': (week_profit / (week_bets * 100) * 100) if week_bets > 0 else 0
    })

    print(f"Week {week:2d}: {len(test_df):2d} games | {week_bets:2d} bets | "
          f"Record: {week_record:8s} | Win%: {win_rate:5.1f}% | "
          f"Profit: ${week_profit:+7.2f}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

results_df = pd.DataFrame(weekly_results)
bets_df = pd.DataFrame(all_bets)

if len(results_df) == 0:
    print("\nNo bets placed - check data availability.")
else:
    total_bets = results_df['bets'].sum()
    total_wins = results_df['wins'].sum()
    total_losses = results_df['losses'].sum()
    total_pushes = results_df['pushes'].sum()
    total_profit = results_df['profit'].sum()

    overall_win_rate = total_wins / (total_wins + total_losses) * 100 if (total_wins + total_losses) > 0 else 0
    overall_roi = (total_profit / (total_bets * 100) * 100) if total_bets > 0 else 0

    print(f"\nTotal Bets (Edge >= {EDGE_THRESHOLD}): {total_bets}")
    print(f"Record: {total_wins}-{total_losses}" + (f"-{total_pushes}" if total_pushes > 0 else ""))
    print(f"\nWin Rate: {overall_win_rate:.1f}%")
    print(f"Break-Even Rate: 52.4% (at -110 odds)")
    print(f"\nTotal Profit: ${total_profit:+.2f}")
    print(f"ROI: {overall_roi:+.2f}%")

    # Required win rate to break even at -110
    break_even = 110 / (110 + 100) * 100
    print(f"\nEdge Over Break-Even: {overall_win_rate - break_even:+.2f}%")

    # ============================================================
    # EDGE ANALYSIS
    # ============================================================
    if len(bets_df) > 0:
        print("\n" + "=" * 70)
        print("EDGE SIZE ANALYSIS")
        print("=" * 70)

        edge_buckets = [
            (4, 6, "Edge 4-6"),
            (6, 8, "Edge 6-8"),
            (8, 10, "Edge 8-10"),
            (10, float('inf'), "Edge 10+")
        ]

        print(f"\n{'Edge Range':<15} {'Bets':>6} {'Record':>10} {'Win%':>8} {'Profit':>10} {'ROI':>8}")
        print("-" * 60)

        for low, high, label in edge_buckets:
            bucket = bets_df[(bets_df['edge'].abs() >= low) & (bets_df['edge'].abs() < high)]
            if len(bucket) > 0:
                bucket_wins = (bucket['result'] == 'win').sum()
                bucket_losses = (bucket['result'] == 'loss').sum()
                bucket_pushes = (bucket['result'] == 'push').sum()
                bucket_profit = bucket['profit'].sum()
                bucket_win_rate = bucket_wins / (bucket_wins + bucket_losses) * 100 if (bucket_wins + bucket_losses) > 0 else 0
                bucket_roi = (bucket_profit / (len(bucket) * 100) * 100)
                record = f"{bucket_wins}-{bucket_losses}" + (f"-{bucket_pushes}" if bucket_pushes > 0 else "")

                print(f"{label:<15} {len(bucket):>6} {record:>10} {bucket_win_rate:>7.1f}% ${bucket_profit:>8.2f} {bucket_roi:>+7.1f}%")

        # ============================================================
        # BEST & WORST BETS
        # ============================================================
        wins = bets_df[bets_df['result'] == 'win']
        losses = bets_df[bets_df['result'] == 'loss']

        if len(wins) > 0:
            print("\n" + "=" * 70)
            print("TOP 5 BEST BETS (Highest Edge Wins)")
            print("=" * 70)

            top_wins = wins.sort_values('edge', key=abs, ascending=False).head(5)
            for _, bet in top_wins.iterrows():
                print(f"Week {bet['week']}: {bet['game']}")
                print(f"  Bet: {bet['bet_team']} | Edge: {bet['edge']:+.1f}")
                print(f"  Predicted: {bet['predicted']:.1f} | Vegas: {bet['vegas']:.1f} | Actual: {bet['actual']:.0f}")
                print()

        if len(losses) > 0:
            print("=" * 70)
            print("TOP 5 WORST BETS (Highest Edge Losses)")
            print("=" * 70)

            top_losses = losses.sort_values('edge', key=abs, ascending=False).head(5)
            for _, bet in top_losses.iterrows():
                print(f"Week {bet['week']}: {bet['game']}")
                print(f"  Bet: {bet['bet_team']} | Edge: {bet['edge']:+.1f}")
                print(f"  Predicted: {bet['predicted']:.1f} | Vegas: {bet['vegas']:.1f} | Actual: {bet['actual']:.0f}")
                print()

        # ============================================================
        # WEEKLY PROFIT CHART (ASCII)
        # ============================================================
        print("=" * 70)
        print("WEEKLY CUMULATIVE PROFIT CHART")
        print("=" * 70)

        cumulative_profit = 0
        for _, row in results_df.iterrows():
            cumulative_profit += row['profit']
            if cumulative_profit >= 0:
                profit_bar = '+' * int(cumulative_profit / 50)
            else:
                profit_bar = '-' * int(abs(cumulative_profit) / 50)
            print(f"Week {row['week']:2d}: ${cumulative_profit:+8.2f} {profit_bar}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    # Save detailed results
    if len(bets_df) > 0:
        bets_df.to_csv('backtest_bets.csv', index=False)
        results_df.to_csv('backtest_weekly.csv', index=False)
        print(f"\nDetailed results saved to 'backtest_bets.csv' and 'backtest_weekly.csv'")
