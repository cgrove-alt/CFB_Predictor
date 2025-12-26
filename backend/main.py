"""
Sharp Sports Predictor - FastAPI Backend

Railway deployment backend for CFB predictions.

Deployed: 2025-12-26 16:12 PST (ESPN fallback + CFBD lines fix)
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import requests
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor import (
    generate_predictions, load_v19_model, load_history_data,
    build_lines_dict, fetch_schedule, fetch_lines
)
from odds_fetcher import fetch_ncaaf_spreads, get_spread_for_game, fetch_odds_api_status

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
CFBD_BASE_URL = "https://api.collegefootballdata.com"

# Allowed origins for CORS (update with your Vercel domain)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://*.vercel.app",
]

# Book preference order for selecting lines
PREFERRED_BOOKS = ['consensus', 'Caesars', 'DraftKings', 'FanDuel', 'BetMGM', 'Bovada']


# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class AuthRequest(BaseModel):
    password: str


class AuthResponse(BaseModel):
    authenticated: bool
    message: str


class Prediction(BaseModel):
    home_team: str
    away_team: str
    game: str
    signal: str
    team_to_bet: str
    opponent: str
    spread_to_bet: float
    vegas_spread: float
    predicted_margin: float
    predicted_edge: float
    cover_probability: float
    bet_recommendation: str
    confidence_tier: str
    bet_size: float
    kelly_fraction: float
    line_movement: float
    game_quality_score: int
    start_date: Optional[str] = None
    completed: bool = False
    game_id: Optional[int] = None


class PredictionsResponse(BaseModel):
    season: int
    week: int
    season_type: str
    last_refresh: Optional[str] = None
    predictions: List[Prediction]
    total_games: int
    bet_count: int
    lean_count: int
    pass_count: int


class StatusResponse(BaseModel):
    status: str
    last_refresh: Optional[str] = None
    next_refresh: Optional[str] = None
    is_gameday: bool
    interval_hours: float
    model_loaded: bool
    data_loaded: bool
    games_in_history: int


class Game(BaseModel):
    id: int
    home_team: str
    away_team: str
    start_date: Optional[str] = None
    completed: bool
    venue: Optional[str] = None
    home_points: Optional[int] = None
    away_points: Optional[int] = None
    vegas_spread: Optional[float] = None
    over_under: Optional[float] = None


class GamesResponse(BaseModel):
    season: int
    week: int
    season_type: str
    games: List[Game]


class OddsGame(BaseModel):
    home_team: str
    away_team: str
    spread: float
    provider: str
    commence_time: Optional[str] = None
    fetched_at: Optional[str] = None


class OddsResponse(BaseModel):
    count: int
    source: str
    configured: bool
    games: List[OddsGame]


class GameResult(BaseModel):
    game: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    pick: str
    signal: str
    spread_to_bet: float
    result: str  # "WIN" or "LOSS"
    ats_margin: float
    confidence_tier: str
    bet_size: float
    bet_recommendation: str


class ResultsResponse(BaseModel):
    season: int
    week: int
    season_type: str
    results: List[GameResult]
    total_games: int
    wins: int
    losses: int
    win_rate: float
    status: str  # "profitable", "break_even", "review"


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="Sharp Sports Predictor API",
    description="CFB spread prediction API using V19 dual-target model",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to specific Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler to ensure all errors return JSON
from starlette.requests import Request
from starlette.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_cfbd_headers():
    """Get authorization headers for CFBD API."""
    if not CFBD_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="CFBD_API_KEY not configured"
        )
    return {'Authorization': f'Bearer {CFBD_API_KEY}'}


def fetch_games_from_espn(season_type: str = 'regular') -> list:
    """
    Fetch games from ESPN API as fallback when CFBD returns empty.

    ESPN API is free and doesn't require authentication.
    Returns games in CFBD-compatible format.
    """
    try:
        # ESPN seasontype: 2=regular, 3=postseason
        espn_season_type = "3" if season_type == 'postseason' else "2"
        # groups=80 is FBS (top-level college football)
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=100&seasontype={espn_season_type}"

        logger.info(f"Fetching games from ESPN: {url}")
        resp = requests.get(url, timeout=10)

        if resp.status_code != 200:
            logger.warning(f"ESPN API returned {resp.status_code}")
            return []

        data = resp.json()
        events = data.get('events', [])

        # Convert ESPN format to CFBD-compatible format
        games = []
        for event in events:
            # Skip completed games for predictions
            status = event.get('status', {}).get('type', {}).get('name', '')
            if 'FINAL' in status:
                continue

            competitions = event.get('competitions', [])
            if not competitions:
                continue

            comp = competitions[0]
            competitors = comp.get('competitors', [])

            if len(competitors) != 2:
                continue

            # ESPN: order=0 is home, order=1 is away
            home_team = None
            away_team = None
            for c in competitors:
                team_data = c.get('team', {})
                # Use 'location' for cleaner name (e.g., "Ohio State" vs "Ohio State Buckeyes")
                team_name = team_data.get('location', team_data.get('displayName', team_data.get('name', '')))

                if c.get('homeAway') == 'home':
                    home_team = team_name
                else:
                    away_team = team_name

            if home_team and away_team:
                games.append({
                    'id': int(event.get('id', 0)),
                    'home_team': home_team,
                    'away_team': away_team,
                    'homeTeam': home_team,  # CFBD uses camelCase
                    'awayTeam': away_team,
                    'start_date': event.get('date'),
                    'completed': 'FINAL' in status,
                    'venue': comp.get('venue', {}).get('fullName', ''),
                    'source': 'ESPN',
                })

        logger.info(f"Fetched {len(games)} upcoming games from ESPN")
        return games

    except Exception as e:
        logger.error(f"Error fetching from ESPN: {e}")
        return []


def fetch_games_from_cfbd(season: int, week: int, season_type: str = 'regular') -> list:
    """Fetch games from CFBD API, with ESPN fallback for postseason."""
    try:
        if season_type == 'postseason':
            url = f"{CFBD_BASE_URL}/games?year={season}&seasonType={season_type}"
        else:
            url = f"{CFBD_BASE_URL}/games?year={season}&week={week}&seasonType={season_type}"

        headers = get_cfbd_headers()
        logger.info(f"Fetching games from {url}")
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            games = resp.json()
            logger.info(f"Fetched {len(games)} games from CFBD")

            # If CFBD returns empty for postseason, try ESPN as fallback
            if not games and season_type == 'postseason':
                logger.info("CFBD returned no postseason games, trying ESPN fallback...")
                games = fetch_games_from_espn(season_type)

            return games
        elif resp.status_code == 401:
            logger.error(f"CFBD API returned 401 Unauthorized - check CFBD_API_KEY")
            raise HTTPException(status_code=401, detail="CFBD API key is invalid or expired")
        else:
            logger.warning(f"CFBD games API returned {resp.status_code}: {resp.text}")
            # Try ESPN fallback for postseason
            if season_type == 'postseason':
                logger.info("CFBD failed, trying ESPN fallback...")
                return fetch_games_from_espn(season_type)
            return []
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        # Try ESPN fallback for postseason
        if season_type == 'postseason':
            logger.info("CFBD error, trying ESPN fallback...")
            espn_games = fetch_games_from_espn(season_type)
            if espn_games:
                return espn_games
        raise HTTPException(status_code=500, detail=f"Error fetching games: {str(e)}")


def fetch_lines_from_cfbd(season: int, week: int, season_type: str = 'regular') -> dict:
    """Fetch betting lines from CFBD API and build lookup dict."""
    try:
        if season_type == 'postseason':
            url = f"{CFBD_BASE_URL}/lines?year={season}&seasonType={season_type}"
        else:
            url = f"{CFBD_BASE_URL}/lines?year={season}&week={week}&seasonType={season_type}"

        resp = requests.get(url, headers=get_cfbd_headers(), timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Lines API returned {resp.status_code}")
            return {}

        data = resp.json()
        lines = {}

        for line in data:
            line_books = line.get('lines', [])
            if not line_books:
                continue

            # Find best book based on preference
            selected_book = None
            for preferred in PREFERRED_BOOKS:
                for book in line_books:
                    provider = book.get('provider', '').lower()
                    if preferred.lower() in provider and book.get('spread') is not None:
                        selected_book = book
                        break
                if selected_book:
                    break

            # Fallback to first valid spread
            if not selected_book:
                for book in line_books:
                    if book.get('spread') is not None:
                        selected_book = book
                        break

            if selected_book:
                spread = float(selected_book['spread'])
                opening = float(selected_book.get('spreadOpen', spread)) if selected_book.get('spreadOpen') else spread
                over_under = selected_book.get('overUnder')

                lines[line['homeTeam']] = {
                    'spread_current': spread,
                    'spread_opening': opening,
                    'line_movement': spread - opening,
                    'over_under': float(over_under) if over_under else None,
                    'provider': selected_book.get('provider', 'Unknown'),
                }

        return lines
    except Exception as e:
        logger.error(f"Error fetching lines: {e}")
        return {}


def get_refresh_status() -> dict:
    """Read refresh status from JSON file."""
    status_file = DATA_DIR / '.refresh_status.json'
    try:
        if status_file.exists():
            with open(status_file) as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read status file: {e}")

    # Try local path
    local_status = Path('.refresh_status.json')
    try:
        if local_status.exists():
            with open(local_status) as f:
                return json.load(f)
    except:
        pass

    return {}


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.get("/api/health")
async def health_check():
    """Health check endpoint for Railway."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/debug")
async def debug_info():
    """Debug endpoint to check configuration (non-sensitive)."""
    return {
        "cfbd_api_key_configured": bool(CFBD_API_KEY),
        "cfbd_api_key_length": len(CFBD_API_KEY) if CFBD_API_KEY else 0,
        "the_odds_api_key_configured": bool(THE_ODDS_API_KEY),
        "app_password_configured": bool(APP_PASSWORD),
        "data_dir": str(DATA_DIR),
    }


@app.post("/api/auth", response_model=AuthResponse)
async def authenticate(request: AuthRequest):
    """Simple password authentication."""
    if not APP_PASSWORD:
        raise HTTPException(status_code=500, detail="APP_PASSWORD not configured")

    if request.password == APP_PASSWORD:
        return AuthResponse(authenticated=True, message="Authentication successful")
    else:
        return AuthResponse(authenticated=False, message="Invalid password")


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get scheduler and model status."""
    refresh_status = get_refresh_status()

    # Check if model and data are loaded
    model_loaded = False
    data_loaded = False
    games_count = 0

    try:
        load_v19_model()
        model_loaded = True
    except:
        pass

    try:
        df = load_history_data()
        data_loaded = True
        games_count = len(df)
    except:
        pass

    return StatusResponse(
        status="healthy" if model_loaded and data_loaded else "degraded",
        last_refresh=refresh_status.get('last_refresh'),
        next_refresh=refresh_status.get('next_refresh'),
        is_gameday=refresh_status.get('is_gameday', False),
        interval_hours=refresh_status.get('interval_hours', 6),
        model_loaded=model_loaded,
        data_loaded=data_loaded,
        games_in_history=games_count,
    )


@app.get("/api/games", response_model=GamesResponse)
async def get_games(
    season: int = Query(..., description="Season year"),
    week: int = Query(..., description="Week number (1-15 for regular, 1 for postseason)"),
    season_type: str = Query("regular", description="regular or postseason"),
):
    """Get games for a specific week."""
    try:
        games_data = fetch_games_from_cfbd(season, week, season_type)
        lines = fetch_lines_from_cfbd(season, week, season_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_games: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching games: {str(e)}")

    games = []
    for g in games_data:
        # Support both snake_case (ESPN/normalized) and camelCase (CFBD)
        home = g.get('home_team') or g.get('homeTeam')
        away = g.get('away_team') or g.get('awayTeam')

        # Skip games with missing team names
        if not home or not away:
            logger.warning(f"Skipping game with missing team: home={home}, away={away}")
            continue

        line_info = lines.get(home, {})

        games.append(Game(
            id=g.get('id'),
            home_team=home,
            away_team=away,
            start_date=g.get('start_date'),
            completed=g.get('completed', False),
            venue=g.get('venue'),
            home_points=g.get('home_points') or g.get('homePoints'),
            away_points=g.get('away_points') or g.get('awayPoints'),
            vegas_spread=line_info.get('spread_current'),
            over_under=line_info.get('over_under'),
        ))

    return GamesResponse(
        season=season,
        week=week,
        season_type=season_type,
        games=games,
    )


@app.get("/api/odds", response_model=OddsResponse)
async def get_live_odds():
    """
    Get live NCAAF spreads from The Odds API.

    Returns real-time betting lines from major sportsbooks.
    Falls back gracefully if API key is not configured.
    """
    status = fetch_odds_api_status()

    if not status.get('configured'):
        return OddsResponse(
            count=0,
            source="the-odds-api",
            configured=False,
            games=[],
        )

    spreads = fetch_ncaaf_spreads()

    games = []
    for home_team, data in spreads.items():
        # Skip duplicate entries (we store by both normalized and original name)
        if data.get('home_team') != home_team:
            continue

        games.append(OddsGame(
            home_team=data.get('home_team', ''),
            away_team=data.get('away_team', ''),
            spread=data.get('spread', 0),
            provider=data.get('provider', 'Unknown'),
            commence_time=data.get('commence_time'),
            fetched_at=data.get('fetched_at'),
        ))

    return OddsResponse(
        count=len(games),
        source="the-odds-api",
        configured=True,
        games=games,
    )


@app.get("/api/debug/predictions")
async def debug_predictions(
    season: int = Query(2025),
    season_type: str = Query("postseason"),
):
    """Debug endpoint to trace prediction flow."""
    from odds_fetcher import fetch_ncaaf_spreads

    debug_info = {"steps": []}

    # Step 1: Fetch live odds
    live_odds = fetch_ncaaf_spreads()
    debug_info["steps"].append(f"1. Live odds: {len(live_odds)} games")

    # Step 2: Fetch CFBD games
    try:
        cfbd_games = fetch_games_from_cfbd(season, 1, season_type)
        debug_info["steps"].append(f"2. CFBD games: {len(cfbd_games)} games")
        if cfbd_games:
            sample = cfbd_games[0]
            debug_info["sample_game_keys"] = list(sample.keys())[:10]
            debug_info["sample_home"] = sample.get('homeTeam') or sample.get('home_team')
    except Exception as e:
        debug_info["steps"].append(f"2. CFBD games ERROR: {str(e)}")

    # Step 3: Filter valid games
    valid_cfbd_games = []
    for game in cfbd_games:
        home = game.get('homeTeam') or game.get('home_team')
        away = game.get('awayTeam') or game.get('away_team')
        if home and away:
            game['home_team'] = home
            game['away_team'] = away
            valid_cfbd_games.append(game)
    debug_info["steps"].append(f"3. Valid games: {len(valid_cfbd_games)}")

    # Step 4: Fetch CFBD lines
    try:
        cfbd_lines = fetch_lines_from_cfbd(season, 1, season_type)
        debug_info["steps"].append(f"4. CFBD lines: {len(cfbd_lines)} games")
        if cfbd_lines:
            debug_info["sample_line_keys"] = list(cfbd_lines.keys())[:5]
    except Exception as e:
        debug_info["steps"].append(f"4. CFBD lines ERROR: {str(e)}")

    # Step 5: Match games to lines
    lines_dict = {}
    for game in valid_cfbd_games:
        home = game.get('home_team')
        if home in cfbd_lines:
            lines_dict[home] = cfbd_lines[home]
    debug_info["steps"].append(f"5. Games with CFBD lines: {len(lines_dict)}")

    # Step 6: Try to generate predictions
    try:
        from predictor import generate_predictions, load_v19_model, load_history_data

        model = load_v19_model()
        debug_info["steps"].append("6. Model loaded: OK")

        history_df = load_history_data()
        debug_info["steps"].append(f"7. History data: {len(history_df)} rows")

        # Try generating for first 3 games only
        test_games = valid_cfbd_games[:3]
        debug_info["test_games"] = [{
            "home_team": g.get('home_team'),
            "homeTeam": g.get('homeTeam'),
        } for g in test_games]

        test_lines = {g['home_team']: lines_dict[g['home_team']] for g in test_games if g['home_team'] in lines_dict}
        debug_info["test_lines_keys"] = list(test_lines.keys())

        if test_lines:
            predictions = generate_predictions(
                games=test_games,
                lines_dict=test_lines,
                season=season,
                week=1,
                bankroll=1000,
                season_type=season_type,
            )
            debug_info["steps"].append(f"8. Test predictions: {len(predictions)} results")
            if predictions:
                debug_info["sample_prediction"] = {
                    "home": predictions[0].get("home_team"),
                    "recommendation": predictions[0].get("bet_recommendation"),
                }
        else:
            debug_info["steps"].append("8. No test games with lines")
    except Exception as e:
        import traceback
        debug_info["steps"].append(f"6-8. Prediction ERROR: {str(e)}")
        debug_info["error_traceback"] = traceback.format_exc()

    return debug_info


@app.get("/api/predictions", response_model=PredictionsResponse)
async def get_predictions(
    season: int = Query(..., description="Season year"),
    week: int = Query(..., description="Week number"),
    season_type: str = Query("regular", description="regular or postseason"),
    bankroll: int = Query(1000, description="Bankroll for Kelly sizing"),
):
    """Get predictions for a specific week."""
    # Try to get live odds from The Odds API first
    live_odds = fetch_ncaaf_spreads()
    logger.info(f"Fetched {len(live_odds)} live odds from The Odds API")

    # Fetch games from CFBD
    cfbd_games = fetch_games_from_cfbd(season, week, season_type)

    # Filter valid CFBD games (with team names)
    # Note: CFBD API returns camelCase (homeTeam), but we normalize to snake_case
    valid_cfbd_games = []
    for game in cfbd_games:
        # Support both camelCase (CFBD) and snake_case (Odds API)
        home = game.get('homeTeam') or game.get('home_team')
        away = game.get('awayTeam') or game.get('away_team')
        if home and away:
            # Normalize to snake_case for consistency
            game['home_team'] = home
            game['away_team'] = away
            valid_cfbd_games.append(game)
        else:
            logger.warning(f"Skipping CFBD game with missing team: home={home}, away={away}")

    # If CFBD returns no valid games, create games from Odds API
    if not valid_cfbd_games and live_odds:
        logger.info("No valid CFBD games found, using Odds API games as source")
        games = []
        for home_team, data in live_odds.items():
            # Skip duplicate entries (we store by both normalized and original name)
            if data.get('home_team') != home_team:
                continue
            games.append({
                'home_team': data.get('home_team'),
                'away_team': data.get('away_team'),
                'start_date': data.get('commence_time'),
                'completed': False,
                'id': None,
            })
    else:
        games = valid_cfbd_games

    if not games:
        raise HTTPException(status_code=404, detail="No games found from CFBD or Odds API")

    # Fetch CFBD lines as fallback
    cfbd_lines = fetch_lines_from_cfbd(season, week, season_type)
    logger.info(f"Fetched {len(cfbd_lines)} lines from CFBD")

    # Merge lines: CFBD primary (has historical opening spread), live odds supplementary
    # This matches Streamlit app behavior for consistent predictions
    lines_dict = {}
    valid_games = []
    for game in games:
        home = game.get('home_team')
        away = game.get('away_team')

        # Skip games with missing team names (shouldn't happen now but keep as safety)
        if not home or not away:
            logger.warning(f"Skipping game with missing team in predictions: home={home}, away={away}")
            continue

        valid_games.append(game)

        # CFBD first - has historical opening spread for accurate line_movement calculation
        if home in cfbd_lines:
            # Start with CFBD data (has historical opening)
            line_data = cfbd_lines[home].copy()

            # Optionally update spread_current with live odds if more recent
            live_spread = get_spread_for_game(home, away, live_odds)
            if live_spread is not None:
                # Update current spread but keep CFBD's opening for line_movement calc
                opening = line_data.get('spread_opening', line_data['spread_current'])
                line_data['spread_current'] = live_spread
                line_data['line_movement'] = live_spread - opening
                line_data['provider'] = 'The Odds API + CFBD'
                logger.debug(f"Updated {home} with live spread {live_spread} (opening: {opening})")

            lines_dict[home] = line_data
            logger.debug(f"Using CFBD for {home}: {line_data['spread_current']}")
        else:
            # Only use Odds API if CFBD doesn't have this game
            live_spread = get_spread_for_game(home, away, live_odds)
            if live_spread is not None:
                lines_dict[home] = {
                    'spread_current': live_spread,
                    'spread_opening': live_spread,
                    'line_movement': 0,
                    'over_under': None,
                    'provider': 'The Odds API (no CFBD)',
                }
                logger.debug(f"Using live odds only for {home}: {live_spread} (no CFBD data)")

    # If no lines available, return empty predictions (not an error for bowl games)
    if not lines_dict:
        logger.warning(f"No betting lines available for {len(valid_games)} games")
        # Return empty predictions with game info for display
        return PredictionsResponse(
            season=season,
            week=week,
            season_type=season_type,
            last_refresh=None,
            predictions=[],
            total_games=len(valid_games),
            bet_count=0,
            lean_count=0,
            pass_count=0,
        )

    # Generate predictions (with weather data if CFBD Pro tier available)
    try:
        predictions_data = generate_predictions(
            games=valid_games,
            lines_dict=lines_dict,
            season=season,
            week=week,
            bankroll=bankroll,
            season_type=season_type,  # Pass season_type for weather API
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Convert to Pydantic models
    predictions = [Prediction(**p) for p in predictions_data]

    # Count recommendations
    bet_count = sum(1 for p in predictions if p.bet_recommendation == 'BET')
    lean_count = sum(1 for p in predictions if p.bet_recommendation == 'LEAN')
    pass_count = sum(1 for p in predictions if p.bet_recommendation == 'PASS')

    # Get refresh status
    refresh_status = get_refresh_status()

    return PredictionsResponse(
        season=season,
        week=week,
        season_type=season_type,
        last_refresh=refresh_status.get('last_refresh'),
        predictions=predictions,
        total_games=len(predictions),
        bet_count=bet_count,
        lean_count=lean_count,
        pass_count=pass_count,
    )


@app.get("/api/results", response_model=ResultsResponse)
async def get_results(
    season: int = Query(..., description="Season year"),
    week: int = Query(..., description="Week number"),
    season_type: str = Query("regular", description="regular or postseason"),
    bankroll: int = Query(1000, description="Bankroll for bet sizing"),
):
    """Get results for completed games with W/L tracking."""
    # Fetch games from CFBD
    cfbd_games = fetch_games_from_cfbd(season, week, season_type)

    # Filter to completed games only
    completed_games = [g for g in cfbd_games if g.get('completed', False)]

    if not completed_games:
        return ResultsResponse(
            season=season,
            week=week,
            season_type=season_type,
            results=[],
            total_games=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            status="no_games",
        )

    # Get predictions for these games to compare
    # We need to re-generate predictions to see what we would have bet
    try:
        # Fetch CFBD lines
        cfbd_lines = fetch_lines_from_cfbd(season, week, season_type)

        # Build lines dict for completed games
        lines_dict = {}
        for game in completed_games:
            home = game.get('homeTeam') or game.get('home_team')
            if home and home in cfbd_lines:
                lines_dict[home] = cfbd_lines[home]

        # Normalize game team names
        valid_games = []
        for game in completed_games:
            home = game.get('homeTeam') or game.get('home_team')
            away = game.get('awayTeam') or game.get('away_team')
            if home and away:
                game['home_team'] = home
                game['away_team'] = away
                valid_games.append(game)

        if not valid_games or not lines_dict:
            return ResultsResponse(
                season=season,
                week=week,
                season_type=season_type,
                results=[],
                total_games=0,
                wins=0,
                losses=0,
                win_rate=0.0,
                status="no_lines",
            )

        # Generate predictions
        predictions_data = generate_predictions(
            games=valid_games,
            lines_dict=lines_dict,
            season=season,
            week=week,
            bankroll=bankroll,
        )
    except Exception as e:
        logger.error(f"Error generating predictions for results: {e}")
        predictions_data = []

    # Build results by matching predictions to completed games
    results = []
    for game in valid_games:
        home = game.get('home_team')
        away = game.get('away_team')
        home_score = game.get('homePoints') or game.get('home_points') or 0
        away_score = game.get('awayPoints') or game.get('away_points') or 0

        # Find the prediction for this game
        pred = None
        for p in predictions_data:
            if p.get('home_team') == home or p.get('away_team') == away:
                pred = p
                break

        if not pred:
            continue

        # Calculate ATS result
        actual_margin = home_score - away_score  # Positive = home won by X
        vegas_spread = pred.get('vegas_spread', 0)
        signal = pred.get('signal', '')

        # Determine if our pick was correct
        if signal == 'BUY':
            # We bet on home team to cover
            ats_result = actual_margin + vegas_spread  # If positive, home covered
            pick_won = ats_result > 0
        else:
            # We bet on away team to cover (FADE home)
            ats_result = -actual_margin - vegas_spread  # If positive, away covered
            pick_won = ats_result > 0

        results.append(GameResult(
            game=f"{away} @ {home}",
            home_team=home,
            away_team=away,
            home_score=home_score,
            away_score=away_score,
            pick=f"{pred.get('team_to_bet', '')} {pred.get('spread_to_bet', 0):+.1f}",
            signal=signal,
            spread_to_bet=pred.get('spread_to_bet', 0),
            result="WIN" if pick_won else "LOSS",
            ats_margin=round(ats_result, 1),
            confidence_tier=pred.get('confidence_tier', 'N/A'),
            bet_size=round(pred.get('bet_size', 0) * bankroll),
            bet_recommendation=pred.get('bet_recommendation', 'PASS'),
        ))

    # Calculate summary stats
    wins = sum(1 for r in results if r.result == "WIN")
    losses = len(results) - wins
    win_rate = (wins / len(results) * 100) if results else 0.0

    # Determine status
    if win_rate >= 55:
        status = "profitable"
    elif win_rate >= 50:
        status = "break_even"
    else:
        status = "review"

    return ResultsResponse(
        season=season,
        week=week,
        season_type=season_type,
        results=results,
        total_games=len(results),
        wins=wins,
        losses=losses,
        win_rate=round(win_rate, 1),
        status=status,
    )


# =============================================================================
# STARTUP
# =============================================================================
@app.on_event("startup")
async def startup_event():
    """Pre-load models on startup."""
    logger.info("Starting Sharp Sports Predictor API...")
    logger.info(f"DATA_DIR: {DATA_DIR}")

    try:
        load_v19_model()
        logger.info("V19 model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load V19 model: {e}")

    try:
        df = load_history_data()
        logger.info(f"History data loaded: {len(df)} games")
    except Exception as e:
        logger.warning(f"Could not pre-load history data: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
