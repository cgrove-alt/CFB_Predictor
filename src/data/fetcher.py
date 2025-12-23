"""
Data Fetching Module for Sharp Sports Predictor.

Provides robust API data fetching with:
- Proper error handling
- Caching
- Rate limiting
- Data validation
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cfbd
import pandas as pd

from ..utils.config import get_config
from ..utils.logging_config import get_logger, StructuredLogger
from ..utils.cache import APICache, cached
from ..utils.validation import GameDataValidator, BettingLineValidator

logger = get_logger(__name__)
structured_logger = StructuredLogger(__name__)


@dataclass
class GameData:
    """Structured game data from API."""

    id: int
    season: int
    week: int
    home_team: str
    away_team: str
    home_points: Optional[int] = None
    away_points: Optional[int] = None
    home_pregame_elo: Optional[float] = None
    away_pregame_elo: Optional[float] = None
    start_date: Optional[str] = None
    venue: Optional[str] = None


@dataclass
class BettingLine:
    """Structured betting line data."""

    game_id: int
    home_team: str
    away_team: str
    spread: float
    spread_open: Optional[float] = None
    over_under: Optional[float] = None
    over_under_open: Optional[float] = None
    provider: Optional[str] = None

    @property
    def line_movement(self) -> float:
        """Calculate line movement from open to current."""
        if self.spread_open is None:
            return 0.0
        return self.spread - self.spread_open


class APIError(Exception):
    """Raised when API call fails."""

    def __init__(self, message: str, endpoint: str, params: dict):
        self.endpoint = endpoint
        self.params = params
        super().__init__(f"{endpoint}: {message}")


class CFBDataFetcher:
    """
    Fetches College Football data from CFBD API.

    Features:
    - Automatic retry on failure
    - Caching of responses
    - Rate limiting
    - Proper error handling and logging
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the data fetcher.

        Args:
            api_key: Optional API key override. If not provided, uses config.
        """
        config = get_config()

        if api_key:
            self._api_key = api_key
        else:
            self._api_key = config.api.cfbd_api_key

        # Set up API client
        self._configuration = cfbd.Configuration()
        self._configuration.access_token = self._api_key
        self._api_client = cfbd.ApiClient(self._configuration)

        # API instances
        self._games_api = cfbd.GamesApi(self._api_client)
        self._betting_api = cfbd.BettingApi(self._api_client)
        self._metrics_api = cfbd.MetricsApi(self._api_client)
        self._stats_api = cfbd.StatsApi(self._api_client)
        self._ratings_api = cfbd.RatingsApi(self._api_client)

        # Cache
        self._cache = APICache(
            schedule_ttl=config.cache.schedule_cache_ttl,
            lines_ttl=config.cache.lines_cache_ttl,
            stats_ttl=config.cache.stats_cache_ttl,
        )

        # Validators
        self._game_validator = GameDataValidator()
        self._line_validator = BettingLineValidator()

        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # 500ms between requests

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _handle_api_error(
        self,
        error: Exception,
        endpoint: str,
        params: dict,
        retry_count: int = 0,
        max_retries: int = 3,
    ) -> None:
        """Handle API errors with retry logic."""
        structured_logger.log_api_call(
            endpoint=endpoint,
            params=params,
            success=False,
            error=str(error),
        )

        if retry_count < max_retries:
            wait_time = 2 ** retry_count  # Exponential backoff
            logger.warning(
                f"API error on {endpoint}, retrying in {wait_time}s: {error}"
            )
            time.sleep(wait_time)
        else:
            raise APIError(str(error), endpoint, params)

    def fetch_games(
        self,
        season: int,
        week: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[GameData]:
        """
        Fetch games for a season/week.

        Args:
            season: Season year
            week: Optional week number
            use_cache: Whether to use cache

        Returns:
            List of GameData objects
        """
        # Check cache
        cache_key = f"games:{season}:{week or 'all'}"
        if use_cache:
            cached_data = self._cache.get_schedule(season, week or 0)
            if cached_data is not None:
                logger.debug(f"Cache hit for games {season} week {week}")
                return cached_data

        # Fetch from API
        self._rate_limit()
        params = {"year": season}
        if week is not None:
            params["week"] = week

        for retry in range(3):
            try:
                if week is not None:
                    games = self._games_api.get_games(year=season, week=week)
                else:
                    games = self._games_api.get_games(year=season)

                # Convert to GameData objects
                result = []
                for game in games:
                    result.append(GameData(
                        id=game.id,
                        season=game.season,
                        week=game.week,
                        home_team=game.home_team,
                        away_team=game.away_team,
                        home_points=game.home_points,
                        away_points=game.away_points,
                        home_pregame_elo=game.home_pregame_elo,
                        away_pregame_elo=game.away_pregame_elo,
                        start_date=str(game.start_date) if game.start_date else None,
                        venue=game.venue if hasattr(game, 'venue') else None,
                    ))

                structured_logger.log_api_call(
                    endpoint="get_games",
                    params=params,
                    success=True,
                    response_size=len(result),
                )

                # Cache result
                if use_cache:
                    self._cache.set_schedule(season, week or 0, result)

                return result

            except Exception as e:
                self._handle_api_error(e, "get_games", params, retry)

        return []

    def fetch_betting_lines(
        self,
        season: int,
        week: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[BettingLine]:
        """
        Fetch betting lines for a season/week.

        Args:
            season: Season year
            week: Optional week number
            use_cache: Whether to use cache

        Returns:
            List of BettingLine objects
        """
        # Check cache
        if use_cache:
            cached_data = self._cache.get_lines(season, week or 0)
            if cached_data is not None:
                logger.debug(f"Cache hit for lines {season} week {week}")
                return cached_data

        # Fetch from API
        self._rate_limit()
        params = {"year": season}
        if week is not None:
            params["week"] = week

        for retry in range(3):
            try:
                if week is not None:
                    lines = self._betting_api.get_lines(year=season, week=week)
                else:
                    lines = self._betting_api.get_lines(year=season)

                # Convert to BettingLine objects
                result = []
                for game_lines in lines:
                    if game_lines.lines and len(game_lines.lines) > 0:
                        # Get first available line (usually consensus)
                        for line in game_lines.lines:
                            if line.spread is not None:
                                spread_open = line.spread_open if hasattr(line, 'spread_open') else None
                                over_under = line.over_under if hasattr(line, 'over_under') else None
                                over_under_open = line.over_under_open if hasattr(line, 'over_under_open') else None

                                result.append(BettingLine(
                                    game_id=game_lines.id,
                                    home_team=game_lines.home_team,
                                    away_team=game_lines.away_team,
                                    spread=float(line.spread),
                                    spread_open=float(spread_open) if spread_open else None,
                                    over_under=float(over_under) if over_under else None,
                                    over_under_open=float(over_under_open) if over_under_open else None,
                                    provider=line.provider if hasattr(line, 'provider') else None,
                                ))
                                break

                structured_logger.log_api_call(
                    endpoint="get_lines",
                    params=params,
                    success=True,
                    response_size=len(result),
                )

                # Cache result
                if use_cache:
                    self._cache.set_lines(season, week or 0, result)

                return result

            except Exception as e:
                self._handle_api_error(e, "get_lines", params, retry)

        return []

    def fetch_ppa(
        self,
        season: int,
        exclude_garbage_time: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch Predicted Points Added (PPA) data.

        Args:
            season: Season year
            exclude_garbage_time: Whether to exclude garbage time plays

        Returns:
            DataFrame with PPA data
        """
        self._rate_limit()
        params = {"year": season, "exclude_garbage_time": exclude_garbage_time}

        for retry in range(3):
            try:
                ppa_data = self._metrics_api.get_predicted_points_added_by_game(
                    year=season,
                    exclude_garbage_time=exclude_garbage_time,
                )

                records = []
                for game in ppa_data:
                    records.append({
                        'game_id': game.game_id,
                        'team': game.team,
                        'opponent': game.opponent,
                        'off_ppa': game.offense.overall if game.offense else None,
                        'def_ppa': game.defense.overall if game.defense else None,
                        'pass_ppa': game.offense.passing if game.offense else None,
                        'rush_ppa': game.offense.rushing if game.offense else None,
                    })

                structured_logger.log_api_call(
                    endpoint="get_ppa",
                    params=params,
                    success=True,
                    response_size=len(records),
                )

                return pd.DataFrame(records)

            except Exception as e:
                self._handle_api_error(e, "get_ppa", params, retry)

        return pd.DataFrame()

    def fetch_all_seasons(
        self,
        seasons: List[int],
        include_lines: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch game data for multiple seasons.

        Args:
            seasons: List of season years
            include_lines: Whether to include betting lines

        Returns:
            DataFrame with all game data
        """
        all_games = []
        all_lines = {}

        for season in seasons:
            logger.info(f"Fetching data for {season}...")

            # Fetch games
            games = self.fetch_games(season)
            for game in games:
                all_games.append({
                    'id': game.id,
                    'season': game.season,
                    'week': game.week,
                    'home_team': game.home_team,
                    'away_team': game.away_team,
                    'home_points': game.home_points,
                    'away_points': game.away_points,
                    'home_pregame_elo': game.home_pregame_elo,
                    'away_pregame_elo': game.away_pregame_elo,
                })

            # Fetch lines if requested
            if include_lines:
                lines = self.fetch_betting_lines(season)
                for line in lines:
                    all_lines[line.game_id] = {
                        'spread_line': line.spread,
                        'spread_open': line.spread_open,
                        'line_movement': line.line_movement,
                        'over_under': line.over_under,
                    }

            time.sleep(1)  # Rate limiting between seasons

        # Create DataFrame
        df = pd.DataFrame(all_games)

        # Filter to completed games
        df = df[df['home_points'].notna()].copy()

        # Calculate margin
        df['Margin'] = df['home_points'] - df['away_points']

        # Merge lines
        if include_lines and all_lines:
            lines_df = pd.DataFrame.from_dict(all_lines, orient='index')
            lines_df.index.name = 'id'
            lines_df = lines_df.reset_index()
            df = df.merge(lines_df, on='id', how='left')

        logger.info(f"Fetched {len(df)} games across {len(seasons)} seasons")
        return df

    def build_lines_dict(
        self,
        lines: List[BettingLine],
    ) -> Dict[str, Dict[str, float]]:
        """
        Build a dictionary of betting lines keyed by home team.

        Args:
            lines: List of BettingLine objects

        Returns:
            Dictionary mapping home_team -> line info
        """
        result = {}
        for line in lines:
            result[line.home_team] = {
                'spread_current': line.spread,
                'spread_opening': line.spread_open or line.spread,
                'line_movement': line.line_movement,
                'over_under': line.over_under,
            }
        return result
