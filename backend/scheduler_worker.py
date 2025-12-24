"""
Scheduler Worker for Railway Cron

This script runs as a Railway cron job to refresh data periodically.
Configure in Railway with: cron schedule "0 */6 * * *" (every 6 hours)
On game days (Saturday), can be increased to every 2 hours.

Usage:
    python scheduler_worker.py
"""

import os
import sys
import json
import time
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")
CFBD_BASE_URL = "https://api.collegefootballdata.com"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_headers():
    """Get CFBD API headers."""
    return {'Authorization': f'Bearer {CFBD_API_KEY}'}


def is_gameday():
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


def fetch_games(year):
    """Fetch all games for a year."""
    url = f"{CFBD_BASE_URL}/games?year={year}"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=30)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"Games API returned {resp.status_code}")
        return []
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        return []


def fetch_lines(year):
    """Fetch betting lines for a year."""
    url = f"{CFBD_BASE_URL}/lines?year={year}"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=30)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"Lines API returned {resp.status_code}")
        return []
    except Exception as e:
        logger.error(f"Error fetching lines: {e}")
        return []


def fetch_team_talent(year):
    """Fetch team talent ratings."""
    url = f"{CFBD_BASE_URL}/talent?year={year}"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=30)
        if resp.status_code == 200:
            return resp.json()
        return []
    except:
        return []


def fetch_team_stats(year):
    """Fetch team season stats."""
    url = f"{CFBD_BASE_URL}/stats/season?year={year}"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=30)
        if resp.status_code == 200:
            return resp.json()
        return []
    except:
        return []


def refresh_data():
    """Main data refresh logic."""
    logger.info("=" * 60)
    logger.info("Starting data refresh...")
    start_time = time.time()

    current_year = datetime.now().year
    years = [current_year - 2, current_year - 1, current_year]

    all_games = []
    all_lines = {}

    # Fetch games and lines for each year
    for year in years:
        logger.info(f"Fetching data for {year}...")

        games = fetch_games(year)
        logger.info(f"  Games: {len(games)}")
        all_games.extend(games)

        lines = fetch_lines(year)
        logger.info(f"  Lines: {len(lines)}")

        # Process lines into lookup dict
        for line in lines:
            game_id = line.get('id')
            if game_id:
                line_books = line.get('lines', [])
                if line_books:
                    # Use first valid spread
                    for book in line_books:
                        if book.get('spread') is not None:
                            all_lines[game_id] = {
                                'spread': book['spread'],
                                'over_under': book.get('overUnder'),
                                'spread_open': book.get('spreadOpen', book['spread']),
                            }
                            break

    logger.info(f"Total games: {len(all_games)}")
    logger.info(f"Total lines: {len(all_lines)}")

    # Build DataFrame
    df_data = []
    for game in all_games:
        game_id = game.get('id')
        line_info = all_lines.get(game_id, {})

        df_data.append({
            'id': game_id,
            'season': game.get('season'),
            'week': game.get('week'),
            'home_team': game.get('home_team'),
            'away_team': game.get('away_team'),
            'home_points': game.get('home_points'),
            'away_points': game.get('away_points'),
            'venue': game.get('venue'),
            'start_date': game.get('start_date'),
            'completed': game.get('completed', False),
            'vegas_spread': line_info.get('spread'),
            'over_under': line_info.get('over_under'),
            'spread_open': line_info.get('spread_open'),
        })

    df = pd.DataFrame(df_data)

    # Calculate margin if game is completed
    df['Margin'] = None
    completed_mask = df['completed'] & df['home_points'].notna() & df['away_points'].notna()
    df.loc[completed_mask, 'Margin'] = df.loc[completed_mask, 'home_points'] - df.loc[completed_mask, 'away_points']

    # Save to data directory
    output_path = DATA_DIR / 'cfb_data_safe.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} games to {output_path}")

    duration = time.time() - start_time
    return True, f"Completed in {duration:.1f}s"


def update_status(success, message):
    """Write status file for API to read."""
    status = {
        'last_refresh': datetime.now().isoformat(),
        'success': success,
        'message': message,
        'next_refresh': (datetime.now() + timedelta(hours=6)).isoformat(),
        'is_gameday': is_gameday(),
        'interval_hours': 2 if is_gameday() else 6,
    }

    status_file = DATA_DIR / '.refresh_status.json'
    try:
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        logger.info(f"Status written to {status_file}")
    except Exception as e:
        logger.error(f"Failed to write status: {e}")


def main():
    logger.info("=" * 60)
    logger.info("RAILWAY SCHEDULER WORKER")
    logger.info("=" * 60)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    logger.info(f"Is gameday: {is_gameday()}")

    if not CFBD_API_KEY:
        logger.error("CFBD_API_KEY not set!")
        update_status(False, "CFBD_API_KEY not configured")
        sys.exit(1)

    try:
        success, message = refresh_data()
        update_status(success, message)
        logger.info(f"Refresh {'succeeded' if success else 'failed'}: {message}")
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        update_status(False, str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
