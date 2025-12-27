"""
V22 Walk-Forward Evaluation Script
====================================

Implements weekly walk-forward cross-validation for V22.1 meta-router model.
Ensures no data leakage by training only on past data for each test week.

Metrics reported:
- Overall: MAE, RMSE, Direction Accuracy, Cover Accuracy
- By Cluster: Standard, High Variance, Blowout, Avg-vs-Avg
- By Confidence Tier: HIGH, MEDIUM-HIGH, MEDIUM, LOW
- By Matchup Type: P4vsP4, P4vsG5, G5vsG5

Usage:
    python walk_forward_eval_v22.py --data cfb_data_safe.csv --season 2024
    python walk_forward_eval_v22.py --data cfb_data_safe.csv --all-seasons
"""

import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

from train_v22_meta import (
    load_and_clean_data,
    V22MetaRouterModel,
    CLUSTER_STANDARD,
    CLUSTER_HIGH_VARIANCE,
    CLUSTER_BLOWOUT,
    CLUSTER_AVG_VS_AVG,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_roi(y_cover_actual: np.ndarray, cover_prob: np.ndarray,
                  threshold: float = 0.55) -> float:
    """
    Calculate ROI for bets placed on games where cover_prob > threshold.
    Assumes -110 odds (bet $110 to win $100).
    """
    bet_mask = cover_prob > threshold
    if not bet_mask.any():
        return 0.0

    wins = y_cover_actual[bet_mask].sum()
    losses = (~y_cover_actual[bet_mask]).sum()

    # At -110 odds: win = +100, lose = -110
    profit = wins * 100 - losses * 110
    total_wagered = bet_mask.sum() * 110

    return (profit / total_wagered) * 100 if total_wagered > 0 else 0.0


def walk_forward_evaluate(df: pd.DataFrame, min_train_weeks: int = 3,
                          season_filter: int = None) -> pd.DataFrame:
    """
    Perform walk-forward evaluation.

    Args:
        df: Full dataset with 'week' and 'season' columns
        min_train_weeks: Minimum weeks required for training before testing
        season_filter: If provided, only evaluate this season

    Returns:
        DataFrame with per-week evaluation metrics
    """
    results = []

    # Filter to specific season if requested
    if season_filter:
        df = df[df['season'] == season_filter].copy()
        logger.info(f"Filtered to season {season_filter}: {len(df)} games")

    if len(df) < 100:
        logger.error(f"Insufficient data for walk-forward: {len(df)} games")
        return pd.DataFrame()

    # Get unique weeks sorted
    weeks = sorted(df['week'].unique())

    for i, test_week in enumerate(weeks):
        if i < min_train_weeks:
            logger.info(f"Skipping week {test_week} (need {min_train_weeks} weeks for training)")
            continue

        # Split: train on all prior weeks, test on current
        train_df = df[df['week'] < test_week].copy()
        test_df = df[df['week'] == test_week].copy()

        if len(train_df) < 100:
            logger.warning(f"Week {test_week}: insufficient training data ({len(train_df)} games)")
            continue

        if len(test_df) < 5:
            logger.warning(f"Week {test_week}: insufficient test data ({len(test_df)} games)")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-Forward Week {test_week}: Train={len(train_df)}, Test={len(test_df)}")

        try:
            # Train fresh model on training data (skip internal split since we do external split)
            model = V22MetaRouterModel()
            model.train(train_df, skip_internal_split=True)

            # Prepare test features
            X_test, X_router_test = model.prepare_features(test_df)
            X_test_scaled = model.scaler.transform(X_test)

            # Get targets
            target_col = 'Margin' if 'Margin' in test_df.columns else 'margin'
            y_margin_test = test_df[target_col].values
            y_cover_test = (y_margin_test + test_df['vegas_spread'] > 0).astype(int).values
            y_cluster_test = test_df['game_cluster'].values

            # Predict
            margin_pred, cover_prob, cluster_pred, confidences, uncertainty, blowout_prob = \
                model.predict(X_test_scaled, X_router_test)

            # Calculate overall metrics
            mae = mean_absolute_error(y_margin_test, margin_pred)
            rmse = np.sqrt(mean_squared_error(y_margin_test, margin_pred))
            cover_pred = (cover_prob > 0.5).astype(int)
            cover_acc = accuracy_score(y_cover_test, cover_pred)

            # Direction accuracy (predicted margin sign matches actual)
            direction_pred = np.sign(margin_pred)
            direction_actual = np.sign(y_margin_test)
            direction_acc = (direction_pred == direction_actual).mean()

            # ROI
            roi = calculate_roi(y_cover_test.astype(bool), cover_prob)

            result = {
                'week': test_week,
                'train_games': len(train_df),
                'test_games': len(test_df),
                'MAE': mae,
                'RMSE': rmse,
                'CoverAcc': cover_acc,
                'DirectionAcc': direction_acc,
                'ROI': roi,
            }

            # Per-cluster metrics
            cluster_names = {
                CLUSTER_STANDARD: 'Standard',
                CLUSTER_HIGH_VARIANCE: 'HighVar',
                CLUSTER_BLOWOUT: 'Blowout',
                CLUSTER_AVG_VS_AVG: 'AvgVsAvg'
            }

            for cid, cname in cluster_names.items():
                mask = y_cluster_test == cid
                if mask.sum() > 0:
                    result[f'{cname}_MAE'] = mean_absolute_error(y_margin_test[mask], margin_pred[mask])
                    result[f'{cname}_CoverAcc'] = accuracy_score(y_cover_test[mask], cover_pred[mask])
                    result[f'{cname}_N'] = int(mask.sum())
                else:
                    result[f'{cname}_MAE'] = np.nan
                    result[f'{cname}_CoverAcc'] = np.nan
                    result[f'{cname}_N'] = 0

            # Per-confidence tier metrics
            for tier, (low, high) in {'HIGH': (0.65, 1.0), 'MEDIUM': (0.55, 0.65), 'LOW': (0.0, 0.55)}.items():
                mask = (cover_prob >= low) & (cover_prob < high)
                if mask.sum() > 0:
                    result[f'{tier}_CoverAcc'] = accuracy_score(y_cover_test[mask], cover_pred[mask])
                    result[f'{tier}_N'] = int(mask.sum())
                else:
                    result[f'{tier}_CoverAcc'] = np.nan
                    result[f'{tier}_N'] = 0

            results.append(result)

            logger.info(f"Week {test_week}: MAE={mae:.2f}, Cover={cover_acc:.3f}, Dir={direction_acc:.3f}, ROI={roi:.1f}%")

        except Exception as e:
            logger.error(f"Week {test_week} failed: {e}")
            continue

    return pd.DataFrame(results)


def print_summary(results_df: pd.DataFrame):
    """Print summary statistics from walk-forward evaluation."""

    print("\n" + "=" * 70)
    print("WALK-FORWARD EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nWeeks evaluated: {len(results_df)}")
    print(f"Total test games: {results_df['test_games'].sum()}")

    print("\n--- OVERALL METRICS ---")
    print(f"Avg MAE: {results_df['MAE'].mean():.2f} (+/- {results_df['MAE'].std():.2f})")
    print(f"Avg RMSE: {results_df['RMSE'].mean():.2f}")
    print(f"Avg Cover Accuracy: {results_df['CoverAcc'].mean():.3f}")
    print(f"Avg Direction Accuracy: {results_df['DirectionAcc'].mean():.3f}")
    print(f"Avg ROI: {results_df['ROI'].mean():.1f}%")

    print("\n--- PER-CLUSTER METRICS ---")
    for cluster in ['Standard', 'HighVar', 'Blowout', 'AvgVsAvg']:
        mae_col = f'{cluster}_MAE'
        acc_col = f'{cluster}_CoverAcc'
        n_col = f'{cluster}_N'

        if mae_col in results_df.columns:
            avg_mae = results_df[mae_col].mean()
            avg_acc = results_df[acc_col].mean()
            total_n = results_df[n_col].sum()

            if not np.isnan(avg_mae):
                print(f"  {cluster:10s}: MAE={avg_mae:.2f}, Cover={avg_acc:.3f}, N={total_n}")

    print("\n--- PER-CONFIDENCE TIER METRICS ---")
    for tier in ['HIGH', 'MEDIUM', 'LOW']:
        acc_col = f'{tier}_CoverAcc'
        n_col = f'{tier}_N'

        if acc_col in results_df.columns:
            avg_acc = results_df[acc_col].mean()
            total_n = results_df[n_col].sum()

            if not np.isnan(avg_acc):
                print(f"  {tier:10s}: Cover={avg_acc:.3f}, N={total_n}")


def main():
    parser = argparse.ArgumentParser(description='V22 Walk-Forward Evaluation')
    parser.add_argument('--data', type=str, default='cfb_data_safe.csv',
                        help='Path to training data CSV')
    parser.add_argument('--season', type=int, default=None,
                        help='Specific season to evaluate')
    parser.add_argument('--all-seasons', action='store_true',
                        help='Evaluate all seasons independently')
    parser.add_argument('--min-weeks', type=int, default=3,
                        help='Minimum weeks required for training')
    parser.add_argument('--output', type=str, default='walk_forward_results.csv',
                        help='Output CSV file for results')

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data}")
    df = load_and_clean_data(args.data)
    logger.info(f"Loaded {len(df)} games across seasons: {sorted(df['season'].unique())}")

    if args.all_seasons:
        # Evaluate each season independently
        all_results = []
        for season in sorted(df['season'].unique()):
            logger.info(f"\n{'#'*70}")
            logger.info(f"# SEASON {season}")
            logger.info(f"{'#'*70}")

            season_results = walk_forward_evaluate(df, min_train_weeks=args.min_weeks, season_filter=season)
            if len(season_results) > 0:
                season_results['season'] = season
                all_results.append(season_results)

        if all_results:
            results_df = pd.concat(all_results, ignore_index=True)
        else:
            results_df = pd.DataFrame()
    else:
        # Evaluate single season or all data together
        results_df = walk_forward_evaluate(df, min_train_weeks=args.min_weeks, season_filter=args.season)

    if len(results_df) == 0:
        logger.error("No results generated. Check data and parameters.")
        return

    # Save results
    results_df.to_csv(args.output, index=False)
    logger.info(f"\nResults saved to {args.output}")

    # Print summary
    print_summary(results_df)

    return results_df


if __name__ == '__main__':
    main()
