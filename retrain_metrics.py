"""
Retrain Metrics Tracking Module

Tracks model performance metrics and determines when retraining is needed.
Reads from predictions_2025_comprehensive.csv and backtest_weekly.csv.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
METRICS_FILE = BASE_DIR / '.retrain_metrics.json'
PREDICTIONS_FILE = BASE_DIR / 'predictions_2025_comprehensive.csv'
BACKTEST_FILE = BASE_DIR / 'backtest_weekly.csv'
DATA_FILE = BASE_DIR / 'cfb_data_safe.csv'

# Trigger thresholds
ACCURACY_THRESHOLD = 0.48      # Below this = trigger full retrain
ROI_THRESHOLD = -0.05          # -5% ROI = trigger fast retrain
CONSECUTIVE_BAD_WEEKS = 2      # 2 bad weeks = trigger full retrain
NEW_GAMES_THRESHOLD = 100      # 100+ new games = trigger fast retrain
WEEKS_SINCE_RETRAIN_MAX = 4    # Force retrain after 4 weeks


class RetrainMetrics:
    """Track and evaluate model performance for retraining decisions."""

    def __init__(self):
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> Dict:
        """Load metrics from JSON file or create default."""
        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted metrics file, creating new one")

        return self._create_default_metrics()

    def _create_default_metrics(self) -> Dict:
        """Create default metrics structure."""
        return {
            "last_retrain": datetime.now().isoformat(),
            "last_retrain_reason": "initial",
            "model_version": "v18_stacking_initial",
            "weeks_since_retrain": 0,
            "games_since_retrain": 0,
            "games_at_last_retrain": self._count_total_games(),
            "rolling_4week_accuracy": 0.55,
            "rolling_4week_mae": 8.0,
            "rolling_4week_roi": 0.0,
            "consecutive_bad_weeks": 0,
            "trigger_thresholds": {
                "accuracy": ACCURACY_THRESHOLD,
                "roi": ROI_THRESHOLD,
                "consecutive_bad": CONSECUTIVE_BAD_WEEKS,
                "new_games": NEW_GAMES_THRESHOLD
            },
            "history": []
        }

    def save_metrics(self):
        """Save metrics to JSON file."""
        with open(METRICS_FILE, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {METRICS_FILE}")

    def _count_total_games(self) -> int:
        """Count total completed games in dataset."""
        if not DATA_FILE.exists():
            return 0
        try:
            df = pd.read_csv(DATA_FILE)
            # Count games with actual results (Margin exists)
            return len(df[df['Margin'].notna()])
        except Exception as e:
            logger.error(f"Error counting games: {e}")
            return 0

    def calculate_rolling_accuracy(self, weeks: int = 4) -> float:
        """Calculate direction accuracy over last N weeks."""
        if not PREDICTIONS_FILE.exists():
            logger.warning(f"Predictions file not found: {PREDICTIONS_FILE}")
            return 0.55  # Default

        try:
            df = pd.read_csv(PREDICTIONS_FILE)
            if 'correct_direction' not in df.columns:
                logger.warning("correct_direction column not found")
                return 0.55

            # Get unique weeks and take last N
            if 'week' in df.columns:
                recent_weeks = sorted(df['week'].unique())[-weeks:]
                df = df[df['week'].isin(recent_weeks)]

            # Calculate accuracy
            if len(df) == 0:
                return 0.55

            accuracy = df['correct_direction'].mean()
            return float(accuracy) if not pd.isna(accuracy) else 0.55

        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return 0.55

    def calculate_rolling_mae(self, weeks: int = 4) -> float:
        """Calculate MAE over last N weeks."""
        if not PREDICTIONS_FILE.exists():
            return 8.0

        try:
            df = pd.read_csv(PREDICTIONS_FILE)

            # Need actual and predicted spread error
            if 'actual_spread_error' not in df.columns or 'predicted_spread_error' not in df.columns:
                # Try alternative column names
                if 'prediction_error' in df.columns:
                    return float(df['prediction_error'].abs().mean())
                return 8.0

            # Get last N weeks
            if 'week' in df.columns:
                recent_weeks = sorted(df['week'].unique())[-weeks:]
                df = df[df['week'].isin(recent_weeks)]

            mae = np.abs(df['actual_spread_error'] - df['predicted_spread_error']).mean()
            return float(mae) if not pd.isna(mae) else 8.0

        except Exception as e:
            logger.error(f"Error calculating MAE: {e}")
            return 8.0

    def calculate_rolling_roi(self, weeks: int = 4) -> float:
        """Calculate ROI over last N weeks from backtest file."""
        if not BACKTEST_FILE.exists():
            return 0.0

        try:
            df = pd.read_csv(BACKTEST_FILE)

            if 'roi' not in df.columns:
                return 0.0

            # Get last N weeks
            recent = df.tail(weeks)

            if len(recent) == 0:
                return 0.0

            # Average ROI over period
            avg_roi = recent['roi'].mean()
            return float(avg_roi) if not pd.isna(avg_roi) else 0.0

        except Exception as e:
            logger.error(f"Error calculating ROI: {e}")
            return 0.0

    def count_consecutive_bad_weeks(self) -> int:
        """Count consecutive weeks with accuracy below 50%."""
        if not BACKTEST_FILE.exists():
            return 0

        try:
            df = pd.read_csv(BACKTEST_FILE)

            if 'win_rate' not in df.columns:
                return 0

            # Check from most recent backwards
            count = 0
            for _, row in df.iloc[::-1].iterrows():
                if row['win_rate'] < 0.50:
                    count += 1
                else:
                    break

            return count

        except Exception as e:
            logger.error(f"Error counting bad weeks: {e}")
            return 0

    def get_current_week(self) -> Tuple[int, int]:
        """Get current season and week from data."""
        if not DATA_FILE.exists():
            return (2025, 1)

        try:
            df = pd.read_csv(DATA_FILE)
            season = int(df['season'].max())
            week = int(df['week'].max())
            return (season, week)
        except Exception as e:
            logger.error(f"Error getting current week: {e}")
            return (2025, 1)

    def is_week_completed(self, season: int, week: int) -> bool:
        """Check if a week's games have all completed (95%+ completion)."""
        if not DATA_FILE.exists():
            return False

        try:
            df = pd.read_csv(DATA_FILE)
            week_games = df[(df['season'] == season) & (df['week'] == week)]

            if len(week_games) == 0:
                return False

            # Count games with results (Margin exists)
            completed = week_games[week_games['Margin'].notna()]
            completion_rate = len(completed) / len(week_games)

            return completion_rate >= 0.95

        except Exception as e:
            logger.error(f"Error checking week completion: {e}")
            return False

    def update_metrics(self):
        """Update all metrics from current data."""
        logger.info("Updating retrain metrics...")

        # Calculate current metrics
        self.metrics['rolling_4week_accuracy'] = self.calculate_rolling_accuracy(4)
        self.metrics['rolling_4week_mae'] = self.calculate_rolling_mae(4)
        self.metrics['rolling_4week_roi'] = self.calculate_rolling_roi(4)
        self.metrics['consecutive_bad_weeks'] = self.count_consecutive_bad_weeks()

        # Update games since last retrain
        current_games = self._count_total_games()
        games_at_retrain = self.metrics.get('games_at_last_retrain', 0)
        self.metrics['games_since_retrain'] = current_games - games_at_retrain

        # Calculate weeks since retrain
        try:
            last_retrain = datetime.fromisoformat(self.metrics['last_retrain'])
            days_since = (datetime.now() - last_retrain).days
            self.metrics['weeks_since_retrain'] = days_since // 7
        except:
            self.metrics['weeks_since_retrain'] = 0

        self.save_metrics()

        logger.info(f"Metrics updated: accuracy={self.metrics['rolling_4week_accuracy']:.3f}, "
                   f"MAE={self.metrics['rolling_4week_mae']:.2f}, "
                   f"ROI={self.metrics['rolling_4week_roi']:.3f}")

    def check_triggers(self) -> Tuple[bool, Optional[str], str]:
        """
        Check if any retraining triggers are met.

        Returns:
            Tuple of (should_retrain, trigger_reason, retrain_type)
            retrain_type is 'fast' or 'full'
        """
        self.update_metrics()

        accuracy = self.metrics['rolling_4week_accuracy']
        roi = self.metrics['rolling_4week_roi']
        bad_weeks = self.metrics['consecutive_bad_weeks']
        new_games = self.metrics['games_since_retrain']
        weeks_since = self.metrics['weeks_since_retrain']

        # Check triggers in priority order

        # 1. Severe accuracy drop - FULL retrain
        if accuracy < ACCURACY_THRESHOLD:
            return (True, f"accuracy_drop ({accuracy:.1%} < {ACCURACY_THRESHOLD:.0%})", 'full')

        # 2. Consecutive bad weeks - FULL retrain
        if bad_weeks >= CONSECUTIVE_BAD_WEEKS:
            return (True, f"consecutive_losses ({bad_weeks} weeks)", 'full')

        # 3. Negative ROI - FAST retrain
        if roi < ROI_THRESHOLD:
            return (True, f"negative_roi ({roi:.1%} < {ROI_THRESHOLD:.0%})", 'fast')

        # 4. Many new games - FAST retrain
        if new_games >= NEW_GAMES_THRESHOLD:
            return (True, f"new_data_volume ({new_games} games)", 'fast')

        # 5. Too long since retrain - FAST retrain
        if weeks_since >= WEEKS_SINCE_RETRAIN_MAX:
            return (True, f"max_weeks_exceeded ({weeks_since} weeks)", 'fast')

        return (False, None, 'none')

    def is_monday(self) -> bool:
        """Check if today is Monday."""
        return datetime.now().weekday() == 0

    def should_weekly_retrain(self) -> Tuple[bool, Optional[str]]:
        """Check if weekly scheduled retrain should occur."""
        if not self.is_monday():
            return (False, None)

        season, week = self.get_current_week()

        # Check if previous week completed
        prev_week = week - 1 if week > 1 else 16  # Handle bowl season wrap

        if self.is_week_completed(season, prev_week):
            return (True, f"weekly_schedule (week {prev_week} completed)")

        return (False, None)

    def record_retrain(self, reason: str, retrain_type: str, model_version: str):
        """Record that a retraining occurred."""
        now = datetime.now()

        # Add to history
        self.metrics['history'].append({
            'date': now.strftime('%Y-%m-%d'),
            'reason': reason,
            'type': retrain_type,
            'accuracy': self.metrics['rolling_4week_accuracy'],
            'roi': self.metrics['rolling_4week_roi'],
            'mae': self.metrics['rolling_4week_mae']
        })

        # Keep only last 20 history entries
        self.metrics['history'] = self.metrics['history'][-20:]

        # Update last retrain info
        self.metrics['last_retrain'] = now.isoformat()
        self.metrics['last_retrain_reason'] = reason
        self.metrics['model_version'] = model_version
        self.metrics['games_at_last_retrain'] = self._count_total_games()
        self.metrics['games_since_retrain'] = 0
        self.metrics['weeks_since_retrain'] = 0

        self.save_metrics()
        logger.info(f"Recorded retrain: {reason} ({retrain_type})")

    def get_status_summary(self) -> str:
        """Get a human-readable status summary."""
        return f"""
Model Status:
  Version: {self.metrics['model_version']}
  Last Retrain: {self.metrics['last_retrain'][:10]}
  Reason: {self.metrics['last_retrain_reason']}

Performance (4-week rolling):
  Accuracy: {self.metrics['rolling_4week_accuracy']:.1%}
  MAE: {self.metrics['rolling_4week_mae']:.2f}
  ROI: {self.metrics['rolling_4week_roi']:.1%}

Triggers:
  Bad Weeks: {self.metrics['consecutive_bad_weeks']} (threshold: {CONSECUTIVE_BAD_WEEKS})
  Games Since Retrain: {self.metrics['games_since_retrain']} (threshold: {NEW_GAMES_THRESHOLD})
  Weeks Since Retrain: {self.metrics['weeks_since_retrain']} (max: {WEEKS_SINCE_RETRAIN_MAX})
"""


def main():
    """Test metrics tracking."""
    metrics = RetrainMetrics()
    metrics.update_metrics()

    print(metrics.get_status_summary())

    should_retrain, reason, retrain_type = metrics.check_triggers()
    print(f"\nRetrain needed: {should_retrain}")
    if should_retrain:
        print(f"Reason: {reason}")
        print(f"Type: {retrain_type}")

    weekly, weekly_reason = metrics.should_weekly_retrain()
    print(f"\nWeekly retrain: {weekly}")
    if weekly:
        print(f"Reason: {weekly_reason}")


if __name__ == '__main__':
    main()
