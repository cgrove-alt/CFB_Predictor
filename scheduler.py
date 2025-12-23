"""
Automatic Data Refresh Scheduler for Sharp Sports Predictor.

This script handles automatic refreshing of CFB data from the CFBD API.

Usage:
    python scheduler.py              # Run once and exit
    python scheduler.py --daemon     # Run continuously as daemon

The scheduler will:
1. Run refresh_all_data.py to fetch fresh data from CFBD API
2. Run prepare_safe_features.py to generate safe features CSV
3. Write status to .refresh_status.json for the app to display
"""

import os
import sys
import time
import logging
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# =============================================================================
# CONFIGURATION
# =============================================================================
REFRESH_INTERVAL_DEFAULT = 3600 * 6     # 6 hours normally
REFRESH_INTERVAL_GAMEDAY = 3600 * 2     # 2 hours on game days
BASE_DIR = Path(__file__).parent
STATUS_FILE = BASE_DIR / '.refresh_status.json'
LOG_DIR = BASE_DIR / 'logs'

# Setup logging
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# SCHEDULER CLASS
# =============================================================================
class RefreshScheduler:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.refresh_all_script = self.base_dir / 'refresh_all_data.py'
        self.fetch_lines_script = self.base_dir / 'fetch_betting_lines.py'
        self.prepare_safe_script = self.base_dir / 'prepare_safe_features.py'
        # V19: New data fetchers for improved predictions
        self.fetch_injuries_script = self.base_dir / 'fetch_injuries.py'
        self.fetch_line_movement_script = self.base_dir / 'fetch_line_movement.py'

    def is_gameday(self) -> bool:
        """Check if today is a CFB game day."""
        today = datetime.now()

        # Saturday is primary CFB day
        if today.weekday() == 5:
            return True

        # Bowl season: late December through early January
        if today.month == 12 and today.day >= 20:
            return True
        if today.month == 1 and today.day <= 10:
            return True

        # Thursday/Friday games during season (Sept-Dec)
        if today.month in [9, 10, 11, 12] and today.weekday() in [3, 4]:
            return True

        return False

    def get_refresh_interval(self) -> int:
        """Get appropriate refresh interval based on schedule."""
        if self.is_gameday():
            return REFRESH_INTERVAL_GAMEDAY
        return REFRESH_INTERVAL_DEFAULT

    def run_script(self, script_path: Path, timeout: int = 300) -> tuple:
        """Run a Python script and return success status and output."""
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Script timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def run_refresh(self) -> tuple:
        """Execute the full data refresh pipeline."""
        logger.info("=" * 60)
        logger.info("Starting data refresh...")
        start_time = time.time()

        # Step 1: Refresh raw data from API
        logger.info("Step 1/5: Running refresh_all_data.py")
        success1, stdout1, stderr1 = self.run_script(self.refresh_all_script, timeout=300)

        if not success1:
            error_msg = f"refresh_all_data.py failed: {stderr1}"
            logger.error(error_msg)
            return False, error_msg

        logger.info("Step 1/5: Complete")

        # Step 2: Fetch betting lines
        logger.info("Step 2/5: Running fetch_betting_lines.py")
        success2, stdout2, stderr2 = self.run_script(self.fetch_lines_script, timeout=180)

        if not success2:
            error_msg = f"fetch_betting_lines.py failed: {stderr2}"
            logger.error(error_msg)
            return False, error_msg

        logger.info("Step 2/5: Complete")

        # Step 3: V19 - Fetch enhanced line movement data
        logger.info("Step 3/5: Running fetch_line_movement.py")
        if self.fetch_line_movement_script.exists():
            success3, stdout3, stderr3 = self.run_script(self.fetch_line_movement_script, timeout=180)
            if not success3:
                logger.warning(f"fetch_line_movement.py failed (non-critical): {stderr3}")
            else:
                logger.info("Step 3/5: Complete")
        else:
            logger.info("Step 3/5: Skipped (script not found)")

        # Step 4: V19 - Fetch injury data (on game days only)
        if self.is_gameday():
            logger.info("Step 4/5: Running fetch_injuries.py (gameday)")
            if self.fetch_injuries_script.exists():
                success4, stdout4, stderr4 = self.run_script(self.fetch_injuries_script, timeout=300)
                if not success4:
                    logger.warning(f"fetch_injuries.py failed (non-critical): {stderr4}")
                else:
                    logger.info("Step 4/5: Complete")
            else:
                logger.info("Step 4/5: Skipped (script not found)")
        else:
            logger.info("Step 4/5: Skipped (not gameday)")

        # Step 5: Prepare safe features
        logger.info("Step 5/5: Running prepare_safe_features.py")
        success5, stdout5, stderr5 = self.run_script(self.prepare_safe_script, timeout=120)

        if not success5:
            error_msg = f"prepare_safe_features.py failed: {stderr5}"
            logger.error(error_msg)
            return False, error_msg

        logger.info("Step 5/5: Complete")

        duration = time.time() - start_time
        success_msg = f"Completed in {duration:.1f}s"
        logger.info(f"Data refresh completed successfully in {duration:.1f}s")

        return True, success_msg

    def update_status(self, success: bool, message: str):
        """Write status to JSON file for app to read."""
        next_interval = self.get_refresh_interval()
        status = {
            'last_refresh': datetime.now().isoformat(),
            'success': success,
            'message': message,
            'next_refresh': (datetime.now() + timedelta(seconds=next_interval)).isoformat(),
            'is_gameday': self.is_gameday(),
            'interval_hours': next_interval / 3600
        }

        try:
            with open(STATUS_FILE, 'w') as f:
                json.dump(status, f, indent=2)
            logger.info(f"Status written to {STATUS_FILE}")
        except Exception as e:
            logger.error(f"Failed to write status file: {e}")

    def run_once(self):
        """Run a single refresh cycle."""
        success, message = self.run_refresh()
        self.update_status(success, message)
        return success

    def run_daemon(self):
        """Run as continuous daemon."""
        logger.info("Starting scheduler daemon...")
        logger.info(f"Game day: {self.is_gameday()}")
        logger.info(f"Refresh interval: {self.get_refresh_interval() / 3600:.1f} hours")

        while True:
            try:
                success, message = self.run_refresh()
                self.update_status(success, message)

                interval = self.get_refresh_interval()
                next_run = datetime.now() + timedelta(seconds=interval)
                logger.info(f"Next refresh at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Sleeping for {interval / 3600:.1f} hours...")

                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Daemon stopped by user")
                break
            except Exception as e:
                logger.error(f"Daemon error: {e}")
                # Sleep for 5 minutes on error before retrying
                time.sleep(300)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Automatic data refresh scheduler for Sharp Sports Predictor'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run continuously as a daemon'
    )
    args = parser.parse_args()

    scheduler = RefreshScheduler()

    if args.daemon:
        scheduler.run_daemon()
    else:
        success = scheduler.run_once()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
