"""
Configuration Management for Sharp Sports Predictor.

Centralizes all configuration including:
- API keys (from environment variables)
- Model hyperparameters
- Betting thresholds
- Feature definitions
- Magic numbers extracted from code
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class APIConfig:
    """API configuration - keys loaded from environment variables."""

    cfbd_api_key: str = field(default_factory=lambda: os.getenv("CFBD_API_KEY", ""))
    odds_api_key: str = field(default_factory=lambda: os.getenv("THE_ODDS_API_KEY", ""))
    sportradar_api_key: str = field(default_factory=lambda: os.getenv("SPORTRADAR_API_KEY", ""))
    require_api_key: bool = field(default_factory=lambda: os.getenv("REQUIRE_API_KEY", "false").lower() != "false")

    def __post_init__(self):
        if not self.cfbd_api_key and self.require_api_key:
            raise ValueError(
                "CFBD_API_KEY environment variable is not set. "
                "Get your free API key at: https://collegefootballdata.com/key "
                "Then set it: export CFBD_API_KEY='your_key_here'"
            )

    def validate_for_api_calls(self):
        """Validate that API key is set before making API calls."""
        if not self.cfbd_api_key:
            raise ValueError(
                "CFBD_API_KEY environment variable is not set. "
                "Get your free API key at: https://collegefootballdata.com/key "
                "Then set it: export CFBD_API_KEY='your_key_here'"
            )


@dataclass
class ModelConfig:
    """Model hyperparameters and training configuration."""

    # HistGradientBoosting parameters (from best_params.txt)
    hgb_max_iter: int = 100
    hgb_max_depth: int = 3
    hgb_learning_rate: float = 0.05
    hgb_l2_regularization: float = 0.1

    # RandomForest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2

    # Ridge parameters
    ridge_alpha: float = 1.0
    ridge_cv_alphas: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])

    # Ensemble weights
    ensemble_hgb_weight: float = 0.5
    ensemble_rf_weight: float = 0.3
    ensemble_lr_weight: float = 0.2

    # Training configuration
    train_seasons: List[int] = field(default_factory=lambda: [2022, 2023, 2024])
    test_season: int = 2025
    fallback_test_season: int = 2024
    stacking_cv_folds: int = 5
    random_state: int = 42


@dataclass
class BettingConfig:
    """Betting thresholds and money management parameters."""

    # Kelly Criterion
    kelly_fraction: float = 0.25  # Quarter Kelly (conservative)
    default_bankroll: float = 1000.0
    min_bet: float = 10.0
    max_bet: float = 200.0
    max_bankroll_pct: float = 0.10  # Max 10% of bankroll per bet

    # Standard odds
    default_american_odds: int = -110

    # Edge thresholds
    min_edge_to_bet: float = 4.0  # Minimum edge to place a bet
    buy_threshold: float = 0.55  # Win probability above which to BUY
    fade_threshold: float = 0.45  # Win probability below which to FADE

    # Variance/Confusion filter
    high_variance_threshold: float = 7.0  # Models heavily disagree

    # Monte Carlo simulation
    monte_carlo_simulations: int = 10000
    monte_carlo_std_dev: float = 14.0  # Standard deviation for CFB margins

    # Line movement
    significant_line_move: float = 1.5


@dataclass
class FeatureConfig:
    """Feature definitions and thresholds."""

    # V6 Features (16 total - original)
    v6_features: List[str] = field(default_factory=lambda: [
        'home_pregame_elo', 'away_pregame_elo',
        'home_last5_score_avg', 'away_last5_score_avg',
        'home_last5_defense_avg', 'away_last5_defense_avg',
        'home_comp_off_ppa', 'away_comp_off_ppa',
        'home_comp_def_ppa', 'away_comp_def_ppa',
        'net_epa',
        'home_team_hfa', 'away_team_hfa',
        'home_rest', 'away_rest',
        'rest_advantage',
    ])

    # V7 Features (21 total - with opponent adjustments)
    v7_features: List[str] = field(default_factory=lambda: [
        'home_pregame_elo', 'away_pregame_elo',
        'home_last5_score_avg', 'away_last5_score_avg',
        'home_last5_defense_avg', 'away_last5_defense_avg',
        'home_comp_off_ppa', 'away_comp_off_ppa',
        'home_comp_def_ppa', 'away_comp_def_ppa',
        'home_adj_off_epa', 'away_adj_off_epa',
        'home_adj_def_epa', 'away_adj_def_epa',
        'adj_net_epa',
        'net_epa',
        'home_team_hfa', 'away_team_hfa',
        'home_rest', 'away_rest',
        'rest_advantage',
    ])

    # Stacking Features (21 total - with interactions)
    stacking_features: List[str] = field(default_factory=lambda: [
        'home_pregame_elo', 'away_pregame_elo',
        'home_last5_score_avg', 'away_last5_score_avg',
        'home_last5_defense_avg', 'away_last5_defense_avg',
        'home_comp_off_ppa', 'away_comp_off_ppa',
        'home_comp_def_ppa', 'away_comp_def_ppa',
        'net_epa',
        'home_team_hfa', 'away_team_hfa',
        'home_rest', 'away_rest',
        'rest_advantage',
        'rest_diff',
        'elo_diff',
        'pass_efficiency_diff',
        'epa_elo_interaction',
        'success_diff',
    ])

    # Default values for missing features
    default_elo: float = 1500.0
    default_score_avg: float = 28.0
    default_defense_avg: float = 24.0
    default_ppa: float = 0.0
    default_hfa: float = 2.0
    default_rest_days: int = 7

    # Rolling stats window
    rolling_window: int = 5

    # Situational thresholds
    lookahead_elo_threshold: float = 1700.0
    blowout_score_diff: int = 28

    # West Coast teams for travel disadvantage
    west_coast_teams: List[str] = field(default_factory=lambda: [
        'USC', 'UCLA', 'Oregon', 'Oregon State', 'Washington', 'Washington State',
        'California', 'Stanford', 'Arizona', 'Arizona State', 'Colorado', 'Utah',
        'Boise State', 'San Diego State', 'Fresno State', 'Nevada', 'UNLV',
        'San Jose State', 'Hawaii', "Hawai'i",
        'Colorado State', 'Wyoming', 'Air Force', 'New Mexico', 'Utah State',
        'BYU',
    ])


@dataclass
class PathConfig:
    """File and directory paths."""

    # Base directory
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    # Data files
    raw_data_file: str = "cfb_data.csv"
    processed_data_file: str = "cfb_data_smart.csv"

    # Model files
    stacking_model_file: str = "cfb_stacking.pkl"
    ensemble_model_file: str = "cfb_ensemble.pkl"
    fallback_model_file: str = "cfb_smart_model.pkl"
    compressed_model_suffix: str = "_compressed.pkl"

    # NFL files
    nfl_data_file: str = "nfl_data.csv"
    nfl_model_file: str = "nfl_model.pkl"

    # Totals files
    totals_data_file: str = "totals_data.csv"
    totals_model_file: str = "cfb_totals_model.pkl"

    # Backtest outputs
    backtest_bets_file: str = "backtest_bets.csv"
    backtest_weekly_file: str = "backtest_weekly.csv"

    @property
    def data_dir(self) -> Path:
        return self.base_dir

    def get_model_path(self, model_name: str) -> Path:
        return self.base_dir / model_name


@dataclass
class CacheConfig:
    """Caching configuration."""

    # Cache TTL in seconds
    schedule_cache_ttl: int = 300  # 5 minutes
    lines_cache_ttl: int = 300
    stats_cache_ttl: int = 3600  # 1 hour

    # Cache directory
    cache_dir: str = ".cache"

    # Enable/disable caching
    enable_api_cache: bool = True
    enable_model_cache: bool = True


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Environment
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


# Singleton configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Force reload of configuration (useful for testing)."""
    global _config
    _config = Config()
    return _config
