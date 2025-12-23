"""
Comprehensive Historical Prediction Analysis for CFB Betting.

This script performs walk-forward validation to generate predictions for all
2025 games, storing comprehensive results for error analysis.

NO SHORTCUTS - Full analysis of every game with all features preserved.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = 'cfb_data_safe.csv'
MODEL_FILE = 'cfb_spread_error_v15.pkl'
CONFIG_FILE = 'cfb_v15_config.pkl'
OUTPUT_FILE = 'predictions_2025_comprehensive.csv'

# Load V15 features
V15_CONFIG = joblib.load(CONFIG_FILE)
FEATURE_COLS = V15_CONFIG['features']


def load_data():
    """Load and prepare the data."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"Total games loaded: {len(df)}")

    # Filter to games with spread data
    df = df[df['vegas_spread'].notna()].copy()
    print(f"Games with spread data: {len(df)}")

    # Ensure spread_error exists
    if 'spread_error' not in df.columns:
        df['margin'] = df['home_points'] - df['away_points']
        df['spread_error'] = df['margin'] - (-df['vegas_spread'])

    # Sort chronologically
    df = df.sort_values(['season', 'week']).reset_index(drop=True)

    return df


def get_xgb_params():
    """Get the XGBoost parameters from V15 training or use defaults."""
    return {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
    }


def walk_forward_predictions(df, feature_cols):
    """
    Generate walk-forward predictions for all 2025 games.

    For each week in 2025:
    1. Train on all games BEFORE that week (2022-2024 + earlier 2025 weeks)
    2. Predict that week's games
    3. Store comprehensive results
    """
    print("\n" + "="*60)
    print("WALK-FORWARD PREDICTION GENERATION")
    print("="*60)

    results = []
    all_predictions = []

    # Get 2025 weeks
    df_2025 = df[df['season'] == 2025]
    weeks = sorted(df_2025['week'].unique())
    print(f"Processing {len(weeks)} weeks in 2025...")

    for week in weeks:
        # Training data: everything before this week
        train_mask = (df['season'] < 2025) | ((df['season'] == 2025) & (df['week'] < week))
        train_data = df[train_mask].copy()

        # Test data: this week
        test_mask = (df['season'] == 2025) & (df['week'] == week)
        test_data = df[test_mask].copy()

        if len(test_data) == 0:
            continue

        # Prepare features
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data['spread_error']
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data['spread_error']

        # Train model
        model = XGBRegressor(**get_xgb_params())
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Calculate metrics for this week
        week_mae = mean_absolute_error(y_test, predictions)
        week_rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Direction accuracy
        direction_correct = ((predictions > 0) == (y_test > 0)).sum()
        direction_total = len(y_test)
        direction_accuracy = direction_correct / direction_total if direction_total > 0 else 0

        # Vegas baseline
        vegas_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))

        print(f"Week {week:2d}: {len(test_data):3d} games | "
              f"MAE: {week_mae:.2f} | Vegas: {vegas_mae:.2f} | "
              f"Direction: {direction_accuracy*100:.1f}%")

        # Store week summary
        results.append({
            'week': week,
            'games': len(test_data),
            'mae': week_mae,
            'rmse': week_rmse,
            'vegas_mae': vegas_mae,
            'beat_vegas': week_mae < vegas_mae,
            'direction_accuracy': direction_accuracy,
        })

        # Store individual game predictions
        for i, (idx, row) in enumerate(test_data.iterrows()):
            game_result = {
                'game_id': row.get('game_id', idx),
                'season': row['season'],
                'week': week,
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'vegas_spread': row['vegas_spread'],
                'home_points': row['home_points'],
                'away_points': row['away_points'],
                'actual_margin': row['home_points'] - row['away_points'],
                'actual_spread_error': row['spread_error'],
                'predicted_spread_error': predictions[i],
                'prediction_error': abs(predictions[i] - row['spread_error']),

                # Derived metrics
                'model_signal': 'BUY' if predictions[i] > 0 else 'FADE',
                'actual_result': 'BUY' if row['spread_error'] > 0 else 'FADE',
                'correct_direction': (predictions[i] > 0) == (row['spread_error'] > 0),

                # Confidence metrics
                'prediction_magnitude': abs(predictions[i]),
                'confidence_tier': get_confidence_tier(predictions[i]),

                # Key features for analysis
                'home_pregame_elo': row.get('home_pregame_elo', 0),
                'away_pregame_elo': row.get('away_pregame_elo', 0),
                'elo_diff': row.get('elo_diff', 0),
                'spread_magnitude': abs(row['vegas_spread']),
            }

            # Add all features for deep analysis
            for feat in feature_cols:
                game_result[f'feature_{feat}'] = row.get(feat, 0)

            all_predictions.append(game_result)

    return pd.DataFrame(results), pd.DataFrame(all_predictions)


def get_confidence_tier(spread_error):
    """Assign confidence tier based on predicted spread error magnitude."""
    error_mag = abs(spread_error)
    if error_mag >= 5.0:
        return 'HIGH'
    elif error_mag >= 3.5:
        return 'MEDIUM-HIGH'
    elif error_mag >= 2.0:
        return 'MEDIUM'
    elif error_mag >= 1.0:
        return 'LOW'
    else:
        return 'VERY LOW'


def analyze_results(week_results, game_results):
    """Analyze the prediction results comprehensively."""
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("="*60)

    # Overall metrics
    total_games = len(game_results)
    overall_mae = game_results['prediction_error'].mean()
    overall_rmse = np.sqrt((game_results['prediction_error'] ** 2).mean())
    vegas_mae = game_results['actual_spread_error'].abs().mean()
    direction_accuracy = game_results['correct_direction'].mean()

    print(f"\nOVERALL PERFORMANCE ({total_games} games)")
    print("-" * 40)
    print(f"Model MAE:          {overall_mae:.2f} points")
    print(f"Vegas MAE:          {vegas_mae:.2f} points")
    print(f"Improvement:        {vegas_mae - overall_mae:+.2f} points")
    print(f"Direction Accuracy: {direction_accuracy*100:.1f}%")

    # Weeks beating Vegas
    weeks_beat_vegas = week_results['beat_vegas'].sum()
    total_weeks = len(week_results)
    print(f"\nWeeks beating Vegas: {weeks_beat_vegas}/{total_weeks} ({weeks_beat_vegas/total_weeks*100:.1f}%)")

    # By confidence tier
    print("\nPERFORMANCE BY CONFIDENCE TIER")
    print("-" * 40)
    for tier in ['HIGH', 'MEDIUM-HIGH', 'MEDIUM', 'LOW', 'VERY LOW']:
        tier_games = game_results[game_results['confidence_tier'] == tier]
        if len(tier_games) > 0:
            tier_mae = tier_games['prediction_error'].mean()
            tier_direction = tier_games['correct_direction'].mean()
            print(f"{tier:12s}: {len(tier_games):4d} games | "
                  f"MAE: {tier_mae:.2f} | Direction: {tier_direction*100:.1f}%")

    # By spread magnitude
    print("\nPERFORMANCE BY SPREAD MAGNITUDE")
    print("-" * 40)
    spread_bins = [
        ('Pick-em (0-3)', game_results['spread_magnitude'] <= 3),
        ('Small (3-7)', (game_results['spread_magnitude'] > 3) & (game_results['spread_magnitude'] <= 7)),
        ('Medium (7-14)', (game_results['spread_magnitude'] > 7) & (game_results['spread_magnitude'] <= 14)),
        ('Large (14-21)', (game_results['spread_magnitude'] > 14) & (game_results['spread_magnitude'] <= 21)),
        ('Blowout (21+)', game_results['spread_magnitude'] > 21),
    ]

    for name, mask in spread_bins:
        bin_games = game_results[mask]
        if len(bin_games) > 0:
            bin_mae = bin_games['prediction_error'].mean()
            bin_direction = bin_games['correct_direction'].mean()
            print(f"{name:18s}: {len(bin_games):4d} games | "
                  f"MAE: {bin_mae:.2f} | Direction: {bin_direction*100:.1f}%")

    # Betting performance simulation
    print("\nBETTING PERFORMANCE SIMULATION")
    print("-" * 40)
    for threshold in [0, 1, 2, 3, 5]:
        bets = game_results[game_results['prediction_magnitude'] >= threshold]
        if len(bets) > 0:
            wins = bets['correct_direction'].sum()
            losses = len(bets) - wins
            win_rate = wins / len(bets) * 100
            profit = wins * 100 - losses * 110  # -110 odds
            roi = profit / (len(bets) * 110) * 100
            print(f"Threshold {threshold}pt: {len(bets):4d} bets | "
                  f"Win: {win_rate:.1f}% | Profit: {profit/100:+.1f}u | ROI: {roi:+.1f}%")

    # Worst predictions (for root cause analysis)
    print("\nTOP 10 WORST PREDICTIONS (for root cause analysis)")
    print("-" * 40)
    worst = game_results.nlargest(10, 'prediction_error')
    for _, row in worst.iterrows():
        print(f"Week {row['week']:2d}: {row['away_team']} @ {row['home_team']}")
        print(f"        Predicted: {row['predicted_spread_error']:+.1f} | "
              f"Actual: {row['actual_spread_error']:+.1f} | "
              f"Error: {row['prediction_error']:.1f}")

    return {
        'total_games': total_games,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'vegas_mae': vegas_mae,
        'improvement': vegas_mae - overall_mae,
        'direction_accuracy': direction_accuracy,
        'weeks_beat_vegas': weeks_beat_vegas,
        'total_weeks': total_weeks,
    }


def save_results(week_results, game_results, analysis):
    """Save all results to files."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save comprehensive game results
    game_results.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved game predictions to: {OUTPUT_FILE}")

    # Save week summary
    week_results.to_csv('predictions_2025_weekly.csv', index=False)
    print(f"Saved weekly summary to: predictions_2025_weekly.csv")

    # Save analysis summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_games': int(analysis['total_games']),
        'overall_mae': float(analysis['overall_mae']),
        'vegas_mae': float(analysis['vegas_mae']),
        'improvement': float(analysis['improvement']),
        'direction_accuracy': float(analysis['direction_accuracy']),
        'weeks_beat_vegas': int(analysis['weeks_beat_vegas']),
        'total_weeks': int(analysis['total_weeks']),
    }

    import json
    with open('predictions_2025_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved analysis summary to: predictions_2025_summary.json")


def main():
    print("="*60)
    print("COMPREHENSIVE HISTORICAL PREDICTION ANALYSIS")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using {len(FEATURE_COLS)} features from V15 model")

    # Load data
    df = load_data()

    # Generate walk-forward predictions
    week_results, game_results = walk_forward_predictions(df, FEATURE_COLS)

    # Analyze results
    analysis = analyze_results(week_results, game_results)

    # Save results
    save_results(week_results, game_results, analysis)

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\nNEXT STEPS:")
    print("1. Run error_pattern_analysis.py for detailed error segmentation")
    print("2. Run error_root_cause.py for SHAP-based root cause analysis")
    print("3. Run train_v16_self_learning.py to train improved model")
    print("="*60)


if __name__ == "__main__":
    main()
