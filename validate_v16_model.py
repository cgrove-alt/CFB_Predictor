"""
Rigorous Validation: V15 vs V16 Model Comparison.

Head-to-head comparison using identical walk-forward methodology
to ensure fair evaluation.

NO SHORTCUTS - Every game evaluated independently.
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
V15_CONFIG_FILE = 'cfb_v15_config.pkl'
V16_CONFIG_FILE = 'cfb_v16_config.pkl'
OUTPUT_FILE = 'v15_vs_v16_comparison.txt'


def load_data():
    """Load data and prepare for comparison."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df = df[df['vegas_spread'].notna()].copy()

    # Ensure spread_error exists
    if 'spread_error' not in df.columns:
        df['margin'] = df['home_points'] - df['away_points']
        df['spread_error'] = df['margin'] - (-df['vegas_spread'])

    df = df.sort_values(['season', 'week']).reset_index(drop=True)
    print(f"Loaded {len(df)} games with spreads")

    return df


def load_configs():
    """Load model configurations."""
    v15_config = joblib.load(V15_CONFIG_FILE)
    print(f"V15: {len(v15_config['features'])} features")

    try:
        v16_config = joblib.load(V16_CONFIG_FILE)
        print(f"V16: {len(v16_config['features'])} features")
    except:
        print("V16 config not found - will use V15 features")
        v16_config = v15_config.copy()

    return v15_config, v16_config


def prepare_v16_features(df, v16_features):
    """Prepare V16-specific features."""
    # Add uncertainty features if missing
    if 'is_pickem' not in df.columns:
        df['is_pickem'] = (df['vegas_spread'].abs() <= 3).astype(int)

    if 'is_mismatch' not in df.columns:
        df['is_mismatch'] = (df['elo_diff'].abs() > 300).astype(int)

    if 'is_early_season' not in df.columns:
        df['is_early_season'] = (df['week'] <= 3).astype(int)

    if 'is_rivalry_week' not in df.columns:
        df['is_rivalry_week'] = df['week'].isin([12, 13]).astype(int)

    if 'home_team_historical_error' not in df.columns:
        df['home_team_historical_error'] = 10.0

    if 'away_team_historical_error' not in df.columns:
        df['away_team_historical_error'] = 10.0

    if 'spread_bucket_error' not in df.columns:
        df['spread_bucket_error'] = 10.0

    if 'feature_completeness' not in df.columns:
        df['feature_completeness'] = 1.0

    if 'is_post_bye' not in df.columns:
        df['is_post_bye'] = 0

    if 'is_short_rest' not in df.columns:
        df['is_short_rest'] = 0

    # Fill any remaining missing columns with 0
    for feat in v16_features:
        if feat not in df.columns:
            df[feat] = 0

    return df


def walk_forward_comparison(df, v15_features, v16_features):
    """Run walk-forward comparison of both models."""
    print("\n" + "="*60)
    print("WALK-FORWARD COMPARISON")
    print("="*60)

    results = {
        'v15': {'predictions': [], 'actuals': [], 'weeks': [], 'games': []},
        'v16': {'predictions': [], 'actuals': [], 'weeks': [], 'games': []},
    }

    week_results = []

    for test_week in range(1, 16):
        train_mask = (df['season'] < 2025) | ((df['season'] == 2025) & (df['week'] < test_week))
        test_mask = (df['season'] == 2025) & (df['week'] == test_week)

        train_data = df[train_mask]
        test_data = df[test_mask]

        if len(test_data) == 0:
            continue

        # V15 predictions
        X_train_v15 = train_data[v15_features].fillna(0)
        y_train = train_data['spread_error']
        X_test_v15 = test_data[v15_features].fillna(0)
        y_test = test_data['spread_error']

        model_v15 = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1
        )
        model_v15.fit(X_train_v15, y_train)
        v15_pred = model_v15.predict(X_test_v15)

        # V16 predictions
        X_train_v16 = train_data[v16_features].fillna(0)
        X_test_v16 = test_data[v16_features].fillna(0)

        model_v16 = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1
        )
        model_v16.fit(X_train_v16, y_train)
        v16_pred = model_v16.predict(X_test_v16)

        # Calculate metrics for this week
        v15_mae = mean_absolute_error(y_test, v15_pred)
        v16_mae = mean_absolute_error(y_test, v16_pred)
        vegas_mae = mean_absolute_error(y_test, np.zeros(len(y_test)))

        v15_direction = ((v15_pred > 0) == (y_test > 0)).mean()
        v16_direction = ((v16_pred > 0) == (y_test > 0)).mean()

        winner = 'V16' if v16_mae < v15_mae else 'V15'

        print(f"Week {test_week:2d}: {len(test_data):3d} games | "
              f"V15 MAE: {v15_mae:.2f} | V16 MAE: {v16_mae:.2f} | "
              f"Winner: {winner}")

        week_results.append({
            'week': test_week,
            'games': len(test_data),
            'v15_mae': v15_mae,
            'v16_mae': v16_mae,
            'vegas_mae': vegas_mae,
            'v15_direction': v15_direction,
            'v16_direction': v16_direction,
            'v16_better': v16_mae < v15_mae,
            'v15_beats_vegas': v15_mae < vegas_mae,
            'v16_beats_vegas': v16_mae < vegas_mae,
        })

        # Store predictions
        for name, preds in [('v15', v15_pred), ('v16', v16_pred)]:
            results[name]['predictions'].extend(preds)
            results[name]['actuals'].extend(y_test.values)
            results[name]['weeks'].extend([test_week] * len(test_data))
            for _, row in test_data.iterrows():
                results[name]['games'].append(f"{row['away_team']} @ {row['home_team']}")

    return results, pd.DataFrame(week_results)


def calculate_overall_metrics(results, week_results):
    """Calculate overall comparison metrics."""
    metrics = {}

    for name in ['v15', 'v16']:
        preds = np.array(results[name]['predictions'])
        actuals = np.array(results[name]['actuals'])

        # Basic metrics
        metrics[name] = {
            'total_games': len(preds),
            'mae': mean_absolute_error(actuals, preds),
            'rmse': np.sqrt(mean_squared_error(actuals, preds)),
            'direction_accuracy': ((preds > 0) == (actuals > 0)).mean(),
        }

        # Betting simulation
        wins = ((preds > 0) == (actuals > 0)).sum()
        losses = len(preds) - wins
        profit = wins * 100 - losses * 110
        roi = profit / (len(preds) * 110) * 100

        metrics[name]['wins'] = wins
        metrics[name]['losses'] = losses
        metrics[name]['profit_units'] = profit / 100
        metrics[name]['roi_pct'] = roi

    # Week-level summary
    metrics['v16_wins_weeks'] = week_results['v16_better'].sum()
    metrics['total_weeks'] = len(week_results)
    metrics['v15_beats_vegas_weeks'] = week_results['v15_beats_vegas'].sum()
    metrics['v16_beats_vegas_weeks'] = week_results['v16_beats_vegas'].sum()

    return metrics


def analyze_by_situation(results, week_results, df):
    """Analyze which situations V16 improves vs V15."""
    analysis = []

    # By spread magnitude
    df_2025 = df[df['season'] == 2025].copy()

    spread_bins = [
        ('Pick-em (0-3)', df_2025['vegas_spread'].abs() <= 3),
        ('Small (3-7)', (df_2025['vegas_spread'].abs() > 3) & (df_2025['vegas_spread'].abs() <= 7)),
        ('Medium (7-14)', (df_2025['vegas_spread'].abs() > 7) & (df_2025['vegas_spread'].abs() <= 14)),
        ('Large (14+)', df_2025['vegas_spread'].abs() > 14),
    ]

    # Note: This is a simplified analysis - in a full implementation,
    # we'd track predictions per game and match them back to situations

    print("\nSITUATION ANALYSIS (from weekly averages):")
    print("-" * 40)

    # Early season vs late season
    early = week_results[week_results['week'] <= 3]
    late = week_results[week_results['week'] >= 10]

    if len(early) > 0:
        print(f"Early Season (weeks 1-3):")
        print(f"  V15 avg MAE: {early['v15_mae'].mean():.2f}")
        print(f"  V16 avg MAE: {early['v16_mae'].mean():.2f}")
        print(f"  V16 improvement: {early['v15_mae'].mean() - early['v16_mae'].mean():+.2f}")

    if len(late) > 0:
        print(f"\nLate Season (weeks 10+):")
        print(f"  V15 avg MAE: {late['v15_mae'].mean():.2f}")
        print(f"  V16 avg MAE: {late['v16_mae'].mean():.2f}")
        print(f"  V16 improvement: {late['v15_mae'].mean() - late['v16_mae'].mean():+.2f}")

    return analysis


def generate_report(metrics, week_results, output_file):
    """Generate comprehensive comparison report."""
    print("\n" + "="*60)
    print("GENERATING COMPARISON REPORT")
    print("="*60)

    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("V15 vs V16 MODEL COMPARISON\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Head-to-head comparison
        f.write("HEAD-TO-HEAD COMPARISON\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Metric':<25} {'V15':>12} {'V16':>12} {'Diff':>12}\n")
        f.write("-"*40 + "\n")

        v15 = metrics['v15']
        v16 = metrics['v16']

        rows = [
            ('Total Games', v15['total_games'], v16['total_games'], 0),
            ('MAE', v15['mae'], v16['mae'], v15['mae'] - v16['mae']),
            ('RMSE', v15['rmse'], v16['rmse'], v15['rmse'] - v16['rmse']),
            ('Direction Accuracy', v15['direction_accuracy'], v16['direction_accuracy'],
             v16['direction_accuracy'] - v15['direction_accuracy']),
            ('Wins', v15['wins'], v16['wins'], v16['wins'] - v15['wins']),
            ('Losses', v15['losses'], v16['losses'], v15['losses'] - v16['losses']),
            ('Profit (units)', v15['profit_units'], v16['profit_units'],
             v16['profit_units'] - v15['profit_units']),
            ('ROI (%)', v15['roi_pct'], v16['roi_pct'],
             v16['roi_pct'] - v15['roi_pct']),
        ]

        for name, v15_val, v16_val, diff in rows:
            if isinstance(v15_val, int):
                f.write(f"{name:<25} {v15_val:>12d} {v16_val:>12d} {diff:>+12d}\n")
            else:
                f.write(f"{name:<25} {v15_val:>12.2f} {v16_val:>12.2f} {diff:>+12.2f}\n")

        f.write("\n")

        # Week-by-week
        f.write("WEEK-BY-WEEK RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Week':<6} {'Games':>6} {'V15':>10} {'V16':>10} {'Winner':>10}\n")
        f.write("-"*40 + "\n")

        for _, row in week_results.iterrows():
            winner = 'V16' if row['v16_better'] else 'V15'
            f.write(f"{row['week']:<6d} {row['games']:>6d} "
                    f"{row['v15_mae']:>10.2f} {row['v16_mae']:>10.2f} {winner:>10}\n")

        f.write("-"*40 + "\n")
        f.write(f"V16 wins: {metrics['v16_wins_weeks']}/{metrics['total_weeks']} weeks\n")
        f.write(f"V15 beats Vegas: {metrics['v15_beats_vegas_weeks']}/{metrics['total_weeks']} weeks\n")
        f.write(f"V16 beats Vegas: {metrics['v16_beats_vegas_weeks']}/{metrics['total_weeks']} weeks\n")

        f.write("\n")

        # Verdict
        f.write("="*60 + "\n")
        f.write("VERDICT\n")
        f.write("="*60 + "\n")

        if v16['mae'] < v15['mae']:
            improvement = (v15['mae'] - v16['mae']) / v15['mae'] * 100
            f.write(f"\nV16 WINS: MAE improved by {improvement:.1f}%\n")
            f.write(f"  V16 MAE: {v16['mae']:.2f} vs V15 MAE: {v15['mae']:.2f}\n")
        else:
            f.write(f"\nV15 WINS: No improvement in MAE\n")
            f.write(f"  V15 MAE: {v15['mae']:.2f} vs V16 MAE: {v16['mae']:.2f}\n")

        if v16['direction_accuracy'] > v15['direction_accuracy']:
            f.write(f"\nV16 direction accuracy improved: "
                    f"{v16['direction_accuracy']*100:.1f}% vs {v15['direction_accuracy']*100:.1f}%\n")
        else:
            f.write(f"\nV15 direction accuracy better: "
                    f"{v15['direction_accuracy']*100:.1f}% vs {v16['direction_accuracy']*100:.1f}%\n")

        if v16['profit_units'] > v15['profit_units']:
            f.write(f"\nV16 more profitable: "
                    f"+{v16['profit_units']:.1f}u vs +{v15['profit_units']:.1f}u\n")
        else:
            f.write(f"\nV15 more profitable: "
                    f"+{v15['profit_units']:.1f}u vs +{v16['profit_units']:.1f}u\n")

        f.write("\n" + "="*60 + "\n")

    print(f"Report saved to: {output_file}")


def main():
    print("="*60)
    print("V15 vs V16 MODEL VALIDATION")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()

    # Load configs
    v15_config, v16_config = load_configs()
    v15_features = v15_config['features']
    v16_features = v16_config['features']

    # Prepare V16 features
    df = prepare_v16_features(df, v16_features)

    # Run comparison
    results, week_results = walk_forward_comparison(df, v15_features, v16_features)

    # Calculate metrics
    metrics = calculate_overall_metrics(results, week_results)

    # Analyze by situation
    analyze_by_situation(results, week_results, df)

    # Generate report
    generate_report(metrics, week_results, OUTPUT_FILE)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    v15 = metrics['v15']
    v16 = metrics['v16']

    print(f"\nV15 Performance:")
    print(f"  MAE: {v15['mae']:.2f}")
    print(f"  Direction: {v15['direction_accuracy']*100:.1f}%")
    print(f"  Profit: {v15['profit_units']:+.1f} units")
    print(f"  ROI: {v15['roi_pct']:+.1f}%")

    print(f"\nV16 Performance:")
    print(f"  MAE: {v16['mae']:.2f}")
    print(f"  Direction: {v16['direction_accuracy']*100:.1f}%")
    print(f"  Profit: {v16['profit_units']:+.1f} units")
    print(f"  ROI: {v16['roi_pct']:+.1f}%")

    print(f"\nV16 vs V15:")
    print(f"  MAE improvement: {v15['mae'] - v16['mae']:+.2f} points")
    print(f"  Direction improvement: {(v16['direction_accuracy'] - v15['direction_accuracy'])*100:+.1f}%")
    print(f"  Profit improvement: {v16['profit_units'] - v15['profit_units']:+.1f} units")
    print(f"  Weeks V16 wins: {metrics['v16_wins_weeks']}/{metrics['total_weeks']}")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
