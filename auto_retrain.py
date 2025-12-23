"""
Auto-Retrain Orchestrator

Checks retraining triggers and executes appropriate retraining:
1. Weekly retraining - After each week's games complete (Monday)
2. Performance-triggered - When accuracy drops below threshold
3. Incremental learning - Fast retrain with new data

Usage:
    python auto_retrain.py [--force fast|full] [--dry-run]
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
FAST_TRAIN_SCRIPT = BASE_DIR / 'train_v18_fast.py'
FULL_TRAIN_SCRIPT = BASE_DIR / 'train_v18_stacking.py'

# Import metrics module
try:
    from retrain_metrics import RetrainMetrics
except ImportError:
    logger.error("Could not import retrain_metrics module")
    sys.exit(1)


def run_fast_retrain(reason: str, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Run fast retraining (15-20 minutes).
    Uses pre-optimized hyperparameters, no Optuna tuning.
    """
    logger.info(f"=" * 60)
    logger.info(f"FAST RETRAIN TRIGGERED")
    logger.info(f"Reason: {reason}")
    logger.info(f"=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Would run fast retraining")
        return (True, "dry_run")

    if not FAST_TRAIN_SCRIPT.exists():
        logger.error(f"Fast train script not found: {FAST_TRAIN_SCRIPT}")
        return (False, "script_not_found")

    try:
        # Run the fast training script
        result = subprocess.run(
            [sys.executable, str(FAST_TRAIN_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )

        if result.returncode == 0:
            logger.info("Fast retraining completed successfully")
            logger.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            return (True, "success")
        else:
            logger.error(f"Fast retraining failed: {result.stderr}")
            return (False, result.stderr[:500])

    except subprocess.TimeoutExpired:
        logger.error("Fast retraining timed out after 30 minutes")
        return (False, "timeout")
    except Exception as e:
        logger.error(f"Fast retraining error: {e}")
        return (False, str(e))


def run_full_retrain(reason: str, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Run full retraining with hyperparameter optimization (3-4 hours).
    Uses Optuna for hyperparameter tuning.
    """
    logger.info(f"=" * 60)
    logger.info(f"FULL RETRAIN TRIGGERED")
    logger.info(f"Reason: {reason}")
    logger.info(f"=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Would run full retraining")
        return (True, "dry_run")

    if not FULL_TRAIN_SCRIPT.exists():
        logger.error(f"Full train script not found: {FULL_TRAIN_SCRIPT}")
        return (False, "script_not_found")

    try:
        # Run the full training script
        result = subprocess.run(
            [sys.executable, str(FULL_TRAIN_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=18000  # 5 hour timeout
        )

        if result.returncode == 0:
            logger.info("Full retraining completed successfully")
            logger.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            return (True, "success")
        else:
            logger.error(f"Full retraining failed: {result.stderr}")
            return (False, result.stderr[:500])

    except subprocess.TimeoutExpired:
        logger.error("Full retraining timed out after 5 hours")
        return (False, "timeout")
    except Exception as e:
        logger.error(f"Full retraining error: {e}")
        return (False, str(e))


def check_and_retrain(dry_run: bool = False) -> Optional[dict]:
    """
    Check all retraining triggers and execute appropriate retraining.

    Returns:
        dict with retrain results, or None if no retrain needed
    """
    logger.info("Checking retraining triggers...")
    metrics = RetrainMetrics()

    # TRIGGER 1: Weekly schedule (Monday after week completes)
    weekly_needed, weekly_reason = metrics.should_weekly_retrain()
    if weekly_needed:
        logger.info(f"Weekly retrain trigger: {weekly_reason}")
        success, status = run_fast_retrain(weekly_reason, dry_run)
        if success and not dry_run:
            metrics.record_retrain(weekly_reason, 'fast', f"v18_stacking_{datetime.now().strftime('%Y%m%d')}")
        return {
            'triggered': True,
            'reason': weekly_reason,
            'type': 'fast',
            'success': success,
            'status': status
        }

    # TRIGGER 2: Performance-based triggers
    should_retrain, trigger_reason, retrain_type = metrics.check_triggers()

    if should_retrain:
        if retrain_type == 'full':
            success, status = run_full_retrain(trigger_reason, dry_run)
        else:
            success, status = run_fast_retrain(trigger_reason, dry_run)

        if success and not dry_run:
            metrics.record_retrain(trigger_reason, retrain_type, f"v18_stacking_{datetime.now().strftime('%Y%m%d')}")

        return {
            'triggered': True,
            'reason': trigger_reason,
            'type': retrain_type,
            'success': success,
            'status': status
        }

    logger.info("No retraining triggers met")
    logger.info(metrics.get_status_summary())
    return None


def force_retrain(retrain_type: str, dry_run: bool = False) -> dict:
    """
    Force a retraining of specified type.

    Args:
        retrain_type: 'fast' or 'full'
        dry_run: If True, don't actually retrain
    """
    reason = f"manual_force_{retrain_type}"

    if retrain_type == 'full':
        success, status = run_full_retrain(reason, dry_run)
    else:
        success, status = run_fast_retrain(reason, dry_run)

    if success and not dry_run:
        metrics = RetrainMetrics()
        metrics.record_retrain(reason, retrain_type, f"v18_stacking_{datetime.now().strftime('%Y%m%d')}")

    return {
        'triggered': True,
        'reason': reason,
        'type': retrain_type,
        'success': success,
        'status': status
    }


def get_status() -> str:
    """Get current retraining status summary."""
    metrics = RetrainMetrics()
    metrics.update_metrics()

    status = metrics.get_status_summary()

    # Add trigger check info
    should_retrain, reason, retrain_type = metrics.check_triggers()
    if should_retrain:
        status += f"\nNext Action: {retrain_type.upper()} RETRAIN needed ({reason})"
    else:
        status += "\nNext Action: No retraining needed"

    weekly, weekly_reason = metrics.should_weekly_retrain()
    if weekly:
        status += f"\nWeekly Retrain: Pending ({weekly_reason})"

    return status


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Auto-retrain orchestrator for CFB prediction models'
    )
    parser.add_argument(
        '--force',
        choices=['fast', 'full'],
        help='Force a specific type of retraining'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Check triggers without actually retraining'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current status and exit'
    )

    args = parser.parse_args()

    # Status check
    if args.status:
        print(get_status())
        return 0

    # Force retrain
    if args.force:
        result = force_retrain(args.force, args.dry_run)
        print(f"\nRetrain Result:")
        print(f"  Type: {result['type']}")
        print(f"  Reason: {result['reason']}")
        print(f"  Success: {result['success']}")
        print(f"  Status: {result['status']}")
        return 0 if result['success'] else 1

    # Normal trigger check
    result = check_and_retrain(args.dry_run)

    if result:
        print(f"\nRetrain Result:")
        print(f"  Type: {result['type']}")
        print(f"  Reason: {result['reason']}")
        print(f"  Success: {result['success']}")
        print(f"  Status: {result['status']}")
        return 0 if result['success'] else 1
    else:
        print("\nNo retraining needed at this time.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
