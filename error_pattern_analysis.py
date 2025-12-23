"""
Error Pattern Analysis for CFB Betting Model.

This script performs deep analysis of prediction errors to identify patterns
that can be used to improve the model.

NO SHORTCUTS - Comprehensive segmentation across all dimensions.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
PREDICTIONS_FILE = 'predictions_2025_comprehensive.csv'
OUTPUT_FILE = 'error_patterns_report.txt'


def load_predictions():
    """Load the comprehensive predictions file."""
    print("Loading predictions...")
    df = pd.read_csv(PREDICTIONS_FILE)
    print(f"Loaded {len(df)} game predictions")
    return df


def analyze_by_spread_magnitude(df):
    """Analyze errors by spread magnitude."""
    results = []

    bins = [
        ('Pick-em (0-3)', df['spread_magnitude'] <= 3),
        ('Small (3-7)', (df['spread_magnitude'] > 3) & (df['spread_magnitude'] <= 7)),
        ('Medium (7-14)', (df['spread_magnitude'] > 7) & (df['spread_magnitude'] <= 14)),
        ('Large (14-21)', (df['spread_magnitude'] > 14) & (df['spread_magnitude'] <= 21)),
        ('Blowout (21+)', df['spread_magnitude'] > 21),
    ]

    for name, mask in bins:
        subset = df[mask]
        if len(subset) > 0:
            results.append({
                'category': name,
                'games': len(subset),
                'pct_of_total': len(subset) / len(df) * 100,
                'mae': subset['prediction_error'].mean(),
                'std': subset['prediction_error'].std(),
                'direction_accuracy': subset['correct_direction'].mean() * 100,
                'mean_prediction': subset['predicted_spread_error'].mean(),
                'mean_actual': subset['actual_spread_error'].mean(),
                'prediction_bias': (subset['predicted_spread_error'] - subset['actual_spread_error']).mean(),
            })

    return pd.DataFrame(results)


def analyze_by_elo_matchup(df):
    """Analyze errors by team quality (Elo-based)."""
    results = []

    # Define Elo thresholds
    elite_threshold = 1600
    weak_threshold = 1400

    bins = [
        ('Elite vs Elite (both > 1600)',
         (df['home_pregame_elo'] > elite_threshold) & (df['away_pregame_elo'] > elite_threshold)),
        ('Elite vs Average',
         ((df['home_pregame_elo'] > elite_threshold) & (df['away_pregame_elo'] <= elite_threshold)) |
         ((df['away_pregame_elo'] > elite_threshold) & (df['home_pregame_elo'] <= elite_threshold))),
        ('Average vs Average',
         (df['home_pregame_elo'] <= elite_threshold) & (df['away_pregame_elo'] <= elite_threshold) &
         (df['home_pregame_elo'] > weak_threshold) & (df['away_pregame_elo'] > weak_threshold)),
        ('Mismatch (Elo diff > 300)',
         abs(df['elo_diff']) > 300),
        ('Weak Team Involved (< 1400)',
         (df['home_pregame_elo'] < weak_threshold) | (df['away_pregame_elo'] < weak_threshold)),
    ]

    for name, mask in bins:
        subset = df[mask]
        if len(subset) > 0:
            results.append({
                'category': name,
                'games': len(subset),
                'pct_of_total': len(subset) / len(df) * 100,
                'mae': subset['prediction_error'].mean(),
                'std': subset['prediction_error'].std(),
                'direction_accuracy': subset['correct_direction'].mean() * 100,
                'avg_elo_diff': subset['elo_diff'].abs().mean(),
            })

    return pd.DataFrame(results)


def analyze_by_week(df):
    """Analyze errors by week (cold start effect)."""
    results = []

    for week in sorted(df['week'].unique()):
        subset = df[df['week'] == week]
        results.append({
            'week': week,
            'games': len(subset),
            'mae': subset['prediction_error'].mean(),
            'std': subset['prediction_error'].std(),
            'direction_accuracy': subset['correct_direction'].mean() * 100,
            'prediction_bias': (subset['predicted_spread_error'] - subset['actual_spread_error']).mean(),
        })

    return pd.DataFrame(results)


def analyze_by_confidence(df):
    """Analyze errors by model confidence tier."""
    results = []

    tiers = ['HIGH', 'MEDIUM-HIGH', 'MEDIUM', 'LOW', 'VERY LOW']

    for tier in tiers:
        subset = df[df['confidence_tier'] == tier]
        if len(subset) > 0:
            # Calculate betting performance
            wins = subset['correct_direction'].sum()
            losses = len(subset) - wins
            profit = wins * 100 - losses * 110
            roi = profit / (len(subset) * 110) * 100 if len(subset) > 0 else 0

            results.append({
                'confidence_tier': tier,
                'games': len(subset),
                'pct_of_total': len(subset) / len(df) * 100,
                'mae': subset['prediction_error'].mean(),
                'std': subset['prediction_error'].std(),
                'direction_accuracy': subset['correct_direction'].mean() * 100,
                'wins': wins,
                'losses': losses,
                'profit_units': profit / 100,
                'roi_pct': roi,
            })

    return pd.DataFrame(results)


def analyze_by_prediction_direction(df):
    """Analyze errors by predicted direction (BUY vs FADE)."""
    results = []

    for signal in ['BUY', 'FADE']:
        subset = df[df['model_signal'] == signal]
        if len(subset) > 0:
            results.append({
                'signal': signal,
                'games': len(subset),
                'pct_of_total': len(subset) / len(df) * 100,
                'mae': subset['prediction_error'].mean(),
                'direction_accuracy': subset['correct_direction'].mean() * 100,
                'mean_prediction': subset['predicted_spread_error'].mean(),
                'mean_actual': subset['actual_spread_error'].mean(),
            })

    return pd.DataFrame(results)


def analyze_home_vs_away(df):
    """Analyze if predictions favor home or away teams incorrectly."""
    results = []

    # Home team covered (positive spread error)
    home_covered = df[df['actual_spread_error'] > 0]
    away_covered = df[df['actual_spread_error'] < 0]

    results.append({
        'category': 'Home Team Covered',
        'games': len(home_covered),
        'predicted_correctly': home_covered['correct_direction'].sum(),
        'accuracy': home_covered['correct_direction'].mean() * 100 if len(home_covered) > 0 else 0,
        'avg_prediction_error': home_covered['prediction_error'].mean() if len(home_covered) > 0 else 0,
    })

    results.append({
        'category': 'Away Team Covered',
        'games': len(away_covered),
        'predicted_correctly': away_covered['correct_direction'].sum(),
        'accuracy': away_covered['correct_direction'].mean() * 100 if len(away_covered) > 0 else 0,
        'avg_prediction_error': away_covered['prediction_error'].mean() if len(away_covered) > 0 else 0,
    })

    # Model bias analysis
    predicted_buy = len(df[df['model_signal'] == 'BUY'])
    actual_buy = len(df[df['actual_result'] == 'BUY'])

    results.append({
        'category': 'Model Predicted BUY',
        'games': predicted_buy,
        'predicted_correctly': np.nan,
        'accuracy': np.nan,
        'avg_prediction_error': np.nan,
    })

    results.append({
        'category': 'Actual BUY Results',
        'games': actual_buy,
        'predicted_correctly': np.nan,
        'accuracy': np.nan,
        'avg_prediction_error': np.nan,
    })

    return pd.DataFrame(results)


def analyze_error_distribution(df):
    """Analyze the distribution of prediction errors."""
    errors = df['prediction_error']

    return {
        'mean': errors.mean(),
        'std': errors.std(),
        'min': errors.min(),
        'max': errors.max(),
        'median': errors.median(),
        'p25': errors.quantile(0.25),
        'p75': errors.quantile(0.75),
        'p90': errors.quantile(0.90),
        'p95': errors.quantile(0.95),
        'p99': errors.quantile(0.99),
        'skewness': errors.skew(),
        'kurtosis': errors.kurtosis(),
    }


def identify_systematic_errors(df):
    """Identify teams or situations with systematic prediction errors."""
    results = []

    # By home team - identify teams we consistently misjudge
    home_team_errors = df.groupby('home_team').agg({
        'prediction_error': ['mean', 'std', 'count'],
        'correct_direction': 'mean',
        'predicted_spread_error': 'mean',
        'actual_spread_error': 'mean',
    }).round(2)

    home_team_errors.columns = ['mae', 'std', 'games', 'direction_accuracy', 'mean_pred', 'mean_actual']
    home_team_errors['bias'] = home_team_errors['mean_pred'] - home_team_errors['mean_actual']

    # Filter to teams with at least 5 home games
    home_team_errors = home_team_errors[home_team_errors['games'] >= 5]

    # Sort by MAE (worst predictions)
    worst_home_teams = home_team_errors.nlargest(10, 'mae')

    # Teams with consistent positive bias (we underestimate them)
    underestimated = home_team_errors[home_team_errors['bias'] < -3].nlargest(10, 'bias')

    # Teams with consistent negative bias (we overestimate them)
    overestimated = home_team_errors[home_team_errors['bias'] > 3].nlargest(10, 'bias')

    return {
        'worst_home_teams': worst_home_teams,
        'underestimated_at_home': underestimated,
        'overestimated_at_home': overestimated,
    }


def generate_report(df, output_file):
    """Generate comprehensive error pattern report."""
    print("\n" + "="*60)
    print("GENERATING ERROR PATTERN REPORT")
    print("="*60)

    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CFB BETTING MODEL - ERROR PATTERN ANALYSIS\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Games Analyzed: {len(df)}\n")
        f.write("\n")

        # Overall metrics
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*40 + "\n")
        f.write(f"Mean Absolute Error: {df['prediction_error'].mean():.2f}\n")
        f.write(f"Direction Accuracy:  {df['correct_direction'].mean()*100:.1f}%\n")
        f.write(f"Prediction Bias:     {(df['predicted_spread_error'] - df['actual_spread_error']).mean():+.2f}\n")
        f.write("\n")

        # By spread magnitude
        f.write("ANALYSIS BY SPREAD MAGNITUDE\n")
        f.write("-"*40 + "\n")
        spread_df = analyze_by_spread_magnitude(df)
        f.write(spread_df.to_string(index=False) + "\n\n")
        print("Analyzed by spread magnitude")

        # By Elo matchup
        f.write("ANALYSIS BY ELO MATCHUP\n")
        f.write("-"*40 + "\n")
        elo_df = analyze_by_elo_matchup(df)
        f.write(elo_df.to_string(index=False) + "\n\n")
        print("Analyzed by Elo matchup")

        # By week
        f.write("ANALYSIS BY WEEK (Cold Start Effect)\n")
        f.write("-"*40 + "\n")
        week_df = analyze_by_week(df)
        f.write(week_df.to_string(index=False) + "\n\n")
        print("Analyzed by week")

        # By confidence
        f.write("ANALYSIS BY CONFIDENCE TIER\n")
        f.write("-"*40 + "\n")
        conf_df = analyze_by_confidence(df)
        f.write(conf_df.to_string(index=False) + "\n\n")
        print("Analyzed by confidence tier")

        # By prediction direction
        f.write("ANALYSIS BY PREDICTION DIRECTION\n")
        f.write("-"*40 + "\n")
        dir_df = analyze_by_prediction_direction(df)
        f.write(dir_df.to_string(index=False) + "\n\n")
        print("Analyzed by prediction direction")

        # Home vs Away
        f.write("ANALYSIS BY HOME/AWAY COVERAGE\n")
        f.write("-"*40 + "\n")
        home_df = analyze_home_vs_away(df)
        f.write(home_df.to_string(index=False) + "\n\n")
        print("Analyzed home vs away")

        # Error distribution
        f.write("ERROR DISTRIBUTION\n")
        f.write("-"*40 + "\n")
        error_dist = analyze_error_distribution(df)
        for key, value in error_dist.items():
            f.write(f"{key:15s}: {value:.2f}\n")
        f.write("\n")
        print("Analyzed error distribution")

        # Systematic errors
        f.write("SYSTEMATIC ERRORS BY TEAM\n")
        f.write("-"*40 + "\n")
        sys_errors = identify_systematic_errors(df)

        f.write("\nWORST PREDICTIONS (by home team):\n")
        f.write(sys_errors['worst_home_teams'].to_string() + "\n\n")

        f.write("\nUNDERESTIMATED AT HOME (we predict too low):\n")
        if len(sys_errors['underestimated_at_home']) > 0:
            f.write(sys_errors['underestimated_at_home'].to_string() + "\n\n")
        else:
            f.write("None with bias < -3\n\n")

        f.write("\nOVERESTIMATED AT HOME (we predict too high):\n")
        if len(sys_errors['overestimated_at_home']) > 0:
            f.write(sys_errors['overestimated_at_home'].to_string() + "\n\n")
        else:
            f.write("None with bias > 3\n\n")

        print("Analyzed systematic errors")

        # Key findings summary
        f.write("\n" + "="*60 + "\n")
        f.write("KEY FINDINGS SUMMARY\n")
        f.write("="*60 + "\n")

        # Find worst performing categories
        spread_worst = spread_df.loc[spread_df['mae'].idxmax()]
        f.write(f"\n1. HARDEST SPREAD CATEGORY: {spread_worst['category']}\n")
        f.write(f"   MAE: {spread_worst['mae']:.2f} | Direction: {spread_worst['direction_accuracy']:.1f}%\n")

        conf_best = conf_df.loc[conf_df['direction_accuracy'].idxmax()]
        f.write(f"\n2. BEST CONFIDENCE TIER: {conf_best['confidence_tier']}\n")
        f.write(f"   MAE: {conf_best['mae']:.2f} | Direction: {conf_best['direction_accuracy']:.1f}%\n")

        # Early season effect
        week_early = week_df[week_df['week'] <= 3]['mae'].mean()
        week_late = week_df[week_df['week'] >= 10]['mae'].mean()
        f.write(f"\n3. COLD START EFFECT:\n")
        f.write(f"   Weeks 1-3 MAE: {week_early:.2f}\n")
        f.write(f"   Weeks 10+ MAE: {week_late:.2f}\n")
        f.write(f"   Improvement: {week_early - week_late:+.2f}\n")

        # Prediction bias
        overall_bias = (df['predicted_spread_error'] - df['actual_spread_error']).mean()
        f.write(f"\n4. PREDICTION BIAS: {overall_bias:+.2f}\n")
        if overall_bias > 0.5:
            f.write("   Model tends to OVERPREDICT spread errors (too confident)\n")
        elif overall_bias < -0.5:
            f.write("   Model tends to UNDERPREDICT spread errors (too conservative)\n")
        else:
            f.write("   Model is relatively well-calibrated\n")

        f.write("\n" + "="*60 + "\n")

    print(f"\nReport saved to: {output_file}")


def main():
    print("="*60)
    print("ERROR PATTERN ANALYSIS")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load predictions
    df = load_predictions()

    # Generate comprehensive report
    generate_report(df, OUTPUT_FILE)

    # Also save analysis dataframes for further use
    print("\nSaving analysis dataframes...")

    analyze_by_spread_magnitude(df).to_csv('error_by_spread.csv', index=False)
    analyze_by_elo_matchup(df).to_csv('error_by_elo.csv', index=False)
    analyze_by_week(df).to_csv('error_by_week.csv', index=False)
    analyze_by_confidence(df).to_csv('error_by_confidence.csv', index=False)

    print("Analysis dataframes saved.")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\nNEXT STEPS:")
    print("1. Review error_patterns_report.txt for insights")
    print("2. Run error_root_cause.py for SHAP-based analysis")
    print("3. Run train_v16_self_learning.py with learned patterns")
    print("="*60)


if __name__ == "__main__":
    main()
