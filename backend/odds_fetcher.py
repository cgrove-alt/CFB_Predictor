"""Fetch live odds from The Odds API."""
import os
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Preferred sportsbooks in order of priority
PREFERRED_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'bovada']


def fetch_ncaaf_spreads() -> Dict[str, dict]:
    """
    Fetch NCAAF spreads from The Odds API.

    Returns:
        Dict keyed by home_team with spread info
    """
    if not ODDS_API_KEY:
        logger.warning("THE_ODDS_API_KEY not set - skipping live odds")
        return {}

    url = f"{ODDS_API_BASE}/sports/americanfootball_ncaaf/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "spreads",
        "oddsFormat": "american"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        games = response.json()

        # Log API usage
        remaining = response.headers.get('x-requests-remaining', '?')
        used = response.headers.get('x-requests-used', '?')
        logger.info(f"Fetched {len(games)} NCAAF games from Odds API. Remaining: {remaining}, Used: {used}")

        return _parse_spreads(games)

    except requests.exceptions.Timeout:
        logger.error("Odds API request timed out")
        return {}
    except requests.exceptions.HTTPError as e:
        logger.error(f"Odds API HTTP error: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        return {}


def _parse_spreads(games: List[dict]) -> Dict[str, dict]:
    """
    Parse API response into spread lookup by home_team.

    Args:
        games: Raw API response list

    Returns:
        Dict keyed by normalized home_team name
    """
    spreads = {}

    for game in games:
        home = game.get('home_team', '')
        away = game.get('away_team', '')
        commence_time = game.get('commence_time', '')

        if not home or not away:
            continue

        # Find the best available spread from preferred bookmakers
        best_spread = None
        best_book = None

        # Sort bookmakers by preference
        bookmakers = game.get('bookmakers', [])
        sorted_books = sorted(
            bookmakers,
            key=lambda b: (
                PREFERRED_BOOKS.index(b.get('key', '').lower())
                if b.get('key', '').lower() in PREFERRED_BOOKS
                else 999
            )
        )

        for bookmaker in sorted_books:
            book_key = bookmaker.get('key', '')
            book_title = bookmaker.get('title', '')

            for market in bookmaker.get('markets', []):
                if market.get('key') == 'spreads':
                    for outcome in market.get('outcomes', []):
                        if outcome.get('name') == home:
                            spread = outcome.get('point', 0)
                            price = outcome.get('price', -110)

                            # Use first (preferred) bookmaker's spread
                            if best_spread is None:
                                best_spread = spread
                                best_book = book_title
                            break
                    break

            if best_spread is not None:
                break

        if best_spread is not None:
            # Normalize team name for matching
            normalized_home = _normalize_team_name(home)

            spreads[normalized_home] = {
                'home_team': home,
                'away_team': away,
                'spread': best_spread,
                'provider': best_book,
                'commence_time': commence_time,
                'fetched_at': datetime.utcnow().isoformat()
            }

            # Also store by original name for backup matching
            spreads[home] = spreads[normalized_home]

    return spreads


def _normalize_team_name(name: str) -> str:
    """
    Normalize team name for matching between APIs.

    The Odds API and CFBD may use different team names.
    """
    # Common normalizations
    replacements = {
        'State': 'St',
        'University': '',
        'Fighting ': '',
        'Golden ': '',
        'Ole Miss': 'Mississippi',
        'Pitt': 'Pittsburgh',
        'USC': 'Southern California',
        'LSU': 'Louisiana State',
        'UCF': 'Central Florida',
        'SMU': 'Southern Methodist',
        'TCU': 'Texas Christian',
        'BYU': 'Brigham Young',
        'UNLV': 'Nevada-Las Vegas',
        'UConn': 'Connecticut',
        'UMass': 'Massachusetts',
    }

    normalized = name.strip()
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    return normalized.strip()


def get_spread_for_game(home_team: str, away_team: str, spreads: Dict[str, dict]) -> Optional[float]:
    """
    Get the spread for a specific game from the spreads dict.

    Args:
        home_team: Home team name (from CFBD)
        away_team: Away team name (from CFBD)
        spreads: Dict of spreads from fetch_ncaaf_spreads()

    Returns:
        Home team spread or None if not found
    """
    # Try direct match first
    if home_team in spreads:
        return spreads[home_team]['spread']

    # Try normalized match
    normalized = _normalize_team_name(home_team)
    if normalized in spreads:
        return spreads[normalized]['spread']

    # Try fuzzy matching on all spreads
    home_lower = home_team.lower()
    for key, data in spreads.items():
        if home_lower in key.lower() or key.lower() in home_lower:
            return data['spread']

    return None


def fetch_odds_api_status() -> dict:
    """Check The Odds API status and remaining quota."""
    if not ODDS_API_KEY:
        return {
            'configured': False,
            'error': 'THE_ODDS_API_KEY not set'
        }

    # Just fetch sports list to check API status
    url = f"{ODDS_API_BASE}/sports"
    params = {"apiKey": ODDS_API_KEY}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        return {
            'configured': True,
            'remaining_requests': response.headers.get('x-requests-remaining'),
            'used_requests': response.headers.get('x-requests-used'),
            'status': 'active'
        }
    except Exception as e:
        return {
            'configured': True,
            'status': 'error',
            'error': str(e)
        }
