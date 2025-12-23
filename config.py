"""
Sharp Sports Predictor - API Configuration

IMPORTANT: API keys should be set via environment variables for security:
    export CFBD_API_KEY="your_key_here"

Get your free API key at: https://collegefootballdata.com/key
"""

import os
import warnings

# =============================================================================
# College Football Data API (cfbd)
# =============================================================================
# Load from environment variable (secure)
CFBD_API_KEY = os.getenv("CFBD_API_KEY", "")

# Check if key is set
if not CFBD_API_KEY:
    warnings.warn(
        "CFBD_API_KEY environment variable is not set. "
        "Set it with: export CFBD_API_KEY='your_key_here' "
        "Get your free API key at: https://collegefootballdata.com/key",
        UserWarning,
    )

# =============================================================================
# Future API Integrations (set via environment variables)
# =============================================================================
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY", "")

# ESPN API (no key required for public endpoints)
ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"
