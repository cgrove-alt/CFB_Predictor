"""
Sharp Sports Predictor - FastAPI Backend

Railway deployment backend for CFB predictions.
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

from predictor import generate_predictions, load_v19_model, load_history_data

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")
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


def fetch_games_from_cfbd(season: int, week: int, season_type: str = 'regular') -> list:
    """Fetch games from CFBD API."""
    try:
        if season_type == 'postseason':
            url = f"{CFBD_BASE_URL}/games?year={season}&seasonType={season_type}"
        else:
            url = f"{CFBD_BASE_URL}/games?year={season}&week={week}&seasonType={season_type}"

        resp = requests.get(url, headers=get_cfbd_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"CFBD games API returned {resp.status_code}")
        return []
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        return []


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
    games_data = fetch_games_from_cfbd(season, week, season_type)
    lines = fetch_lines_from_cfbd(season, week, season_type)

    games = []
    for g in games_data:
        home = g.get('home_team')
        line_info = lines.get(home, {})

        games.append(Game(
            id=g.get('id'),
            home_team=home,
            away_team=g.get('away_team'),
            start_date=g.get('start_date'),
            completed=g.get('completed', False),
            venue=g.get('venue'),
            home_points=g.get('home_points'),
            away_points=g.get('away_points'),
            vegas_spread=line_info.get('spread_current'),
            over_under=line_info.get('over_under'),
        ))

    return GamesResponse(
        season=season,
        week=week,
        season_type=season_type,
        games=games,
    )


@app.get("/api/predictions", response_model=PredictionsResponse)
async def get_predictions(
    season: int = Query(..., description="Season year"),
    week: int = Query(..., description="Week number"),
    season_type: str = Query("regular", description="regular or postseason"),
    bankroll: int = Query(1000, description="Bankroll for Kelly sizing"),
):
    """Get predictions for a specific week."""
    # Fetch games and lines
    games = fetch_games_from_cfbd(season, week, season_type)
    lines_dict = fetch_lines_from_cfbd(season, week, season_type)

    if not games:
        raise HTTPException(status_code=404, detail="No games found for this week")

    if not lines_dict:
        raise HTTPException(status_code=404, detail="No betting lines available")

    # Generate predictions
    try:
        predictions_data = generate_predictions(
            games=games,
            lines_dict=lines_dict,
            season=season,
            week=week,
            bankroll=bankroll,
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
