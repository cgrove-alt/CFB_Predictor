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

    Collects spreads from ALL available bookmakers for line comparison
    and sharp money detection.

    Args:
        games: Raw API response list

    Returns:
        Dict keyed by normalized home_team name with spread info and multi-book data
    """
    spreads = {}

    for game in games:
        home = game.get('home_team', '')
        away = game.get('away_team', '')
        commence_time = game.get('commence_time', '')

        if not home or not away:
            continue

        # Collect spreads from ALL bookmakers for analysis
        all_spreads = []
        best_spread = None
        best_book = None

        # Sort bookmakers by preference for primary spread
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

                            # Collect ALL book spreads for analysis
                            all_spreads.append({
                                'book': book_title,
                                'spread': spread,
                                'price': price,
                            })

                            # Use first (preferred) bookmaker's spread as primary
                            if best_spread is None:
                                best_spread = spread
                                best_book = book_title
                            break
                    break

        if best_spread is not None:
            # Calculate multi-book statistics for sharp money detection
            if all_spreads:
                spread_values = [s['spread'] for s in all_spreads]
                avg_spread = sum(spread_values) / len(spread_values)
                spread_range = max(spread_values) - min(spread_values)

                # Consensus is average across books
                consensus_spread = round(avg_spread, 1)

                # Large disagreement between books may signal sharp action
                has_book_disagreement = spread_range >= 1.0
            else:
                consensus_spread = best_spread
                spread_range = 0
                has_book_disagreement = False

            # Normalize team name for matching
            normalized_home = _normalize_team_name(home)

            spreads[normalized_home] = {
                'home_team': home,
                'away_team': away,
                'spread': best_spread,
                'provider': best_book,
                'commence_time': commence_time,
                'fetched_at': datetime.utcnow().isoformat(),
                # NEW: Multi-book analysis for sharp money detection
                'all_book_spreads': all_spreads,
                'consensus_spread': consensus_spread,
                'spread_range': spread_range,
                'has_book_disagreement': has_book_disagreement,
                'num_books': len(all_spreads),
            }

            # Also store by original name for backup matching
            spreads[home] = spreads[normalized_home]

    return spreads


def _normalize_team_name(name: str) -> str:
    """
    Normalize team name for matching between APIs.

    The Odds API and CFBD may use different team names.
    """
    if name is None:
        return ''

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


def calculate_sharp_indicators(
    current_spread: float,
    opening_spread: float,
    consensus_spread: float,
    has_book_disagreement: bool = False
) -> dict:
    """
    Calculate sharp money indicators from line movement data.

    Args:
        current_spread: Current betting spread
        opening_spread: Opening spread when line was first posted
        consensus_spread: Average spread across all books
        has_book_disagreement: Whether books have >1 point spread difference

    Returns:
        Dict with sharp money indicators:
        - line_movement: Change from opening
        - is_significant_move: Movement >= 1 point
        - sharp_action_signal: Likely sharp money involved
        - move_direction: 'toward_home' or 'toward_away' or 'none'
    """
    line_movement = current_spread - opening_spread

    # Significant movement is 1+ points
    is_significant_move = abs(line_movement) >= 1.0

    # Steam move is 1.5+ points in short time (we don't have timing, so use 1.5+)
    is_steam_move = abs(line_movement) >= 1.5

    # Sharp action signals:
    # 1. Large movement (1.5+)
    # 2. Book disagreement (sharps may have hit some books)
    # 3. Movement against typical public patterns
    sharp_action_signal = is_steam_move or has_book_disagreement

    # Movement direction
    if line_movement < -0.5:
        move_direction = 'toward_home'  # Line moved to favor home more (sharps on home)
    elif line_movement > 0.5:
        move_direction = 'toward_away'  # Line moved to favor away more (sharps on away)
    else:
        move_direction = 'none'

    return {
        'line_movement': line_movement,
        'is_significant_move': is_significant_move,
        'is_steam_move': is_steam_move,
        'sharp_action_signal': sharp_action_signal,
        'move_direction': move_direction,
        'consensus_spread': consensus_spread,
    }


def get_enhanced_spread_for_game(
    home_team: str,
    away_team: str,
    live_odds: Dict[str, dict],
    cfbd_lines: Dict[str, dict]
) -> Optional[dict]:
    """
    Get enhanced spread data combining live odds with CFBD historical data.

    This function merges data sources to provide:
    - Current spread from live odds
    - Opening spread from CFBD
    - Sharp money indicators

    Args:
        home_team: Home team name
        away_team: Away team name
        live_odds: Dict from fetch_ncaaf_spreads()
        cfbd_lines: Dict from CFBD lines API

    Returns:
        Enhanced spread dict with sharp indicators, or None if not found
    """
    # Get live spread
    live_spread = get_spread_for_game(home_team, away_team, live_odds)
    if live_spread is None:
        return None

    # Get live odds data
    live_data = None
    for key in [home_team, _normalize_team_name(home_team)]:
        if key in live_odds:
            live_data = live_odds[key]
            break

    if not live_data:
        return None

    # Get CFBD opening spread
    cfbd_data = cfbd_lines.get(home_team, {})
    opening_spread = cfbd_data.get('spread_opening', live_spread)

    # Calculate sharp indicators
    sharp_indicators = calculate_sharp_indicators(
        current_spread=live_spread,
        opening_spread=opening_spread,
        consensus_spread=live_data.get('consensus_spread', live_spread),
        has_book_disagreement=live_data.get('has_book_disagreement', False)
    )

    return {
        'spread_current': live_spread,
        'spread_opening': opening_spread,
        'provider': live_data.get('provider', 'Unknown'),
        'num_books': live_data.get('num_books', 1),
        **sharp_indicators
    }


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
