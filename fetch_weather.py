"""
Fetch Weather Data for CFB Games.

Weather can significantly impact game outcomes:
- Wind >15 mph: Reduces passing efficiency, impacts kicking
- Temperature <40F: Affects ball handling, skill positions
- Rain/Snow: Increases fumbles, favors run game
- High altitude (5000+ ft): Impacts 2nd half performance

Sources:
1. weather.gov API (free, official NWS data)
2. CFBD venues endpoint (for stadium locations)

Usage:
    python fetch_weather.py
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from config import CFBD_API_KEY

# =============================================================================
# STADIUM DATA
# =============================================================================
# CFB stadium locations with coordinates and dome status
# Key: team name (lowercase), Value: dict with lat, lon, is_dome, altitude_ft
STADIUMS = {
    # SEC
    'alabama': {'lat': 33.2080, 'lon': -87.5503, 'is_dome': False, 'alt': 200},
    'arkansas': {'lat': 36.0678, 'lon': -94.1789, 'is_dome': False, 'alt': 1400},
    'auburn': {'lat': 32.6028, 'lon': -85.4900, 'is_dome': False, 'alt': 800},
    'florida': {'lat': 29.6500, 'lon': -82.3486, 'is_dome': False, 'alt': 50},
    'georgia': {'lat': 33.9494, 'lon': -83.3733, 'is_dome': False, 'alt': 750},
    'kentucky': {'lat': 38.0225, 'lon': -84.5050, 'is_dome': False, 'alt': 1000},
    'lsu': {'lat': 30.4119, 'lon': -91.1836, 'is_dome': False, 'alt': 50},
    'mississippi state': {'lat': 33.4575, 'lon': -88.7936, 'is_dome': False, 'alt': 200},
    'missouri': {'lat': 38.9358, 'lon': -92.3339, 'is_dome': False, 'alt': 800},
    'ole miss': {'lat': 34.3619, 'lon': -89.5342, 'is_dome': False, 'alt': 500},
    'south carolina': {'lat': 34.0094, 'lon': -81.0219, 'is_dome': False, 'alt': 300},
    'tennessee': {'lat': 35.9550, 'lon': -83.9250, 'is_dome': False, 'alt': 1000},
    'texas a&m': {'lat': 30.6103, 'lon': -96.3406, 'is_dome': False, 'alt': 300},
    'vanderbilt': {'lat': 36.1439, 'lon': -86.8094, 'is_dome': False, 'alt': 600},
    'texas': {'lat': 30.2836, 'lon': -97.7325, 'is_dome': False, 'alt': 500},
    'oklahoma': {'lat': 35.2058, 'lon': -97.4425, 'is_dome': False, 'alt': 1200},

    # Big Ten
    'illinois': {'lat': 40.0992, 'lon': -88.2361, 'is_dome': False, 'alt': 750},
    'indiana': {'lat': 39.1806, 'lon': -86.5261, 'is_dome': False, 'alt': 800},
    'iowa': {'lat': 41.6589, 'lon': -91.5511, 'is_dome': False, 'alt': 700},
    'maryland': {'lat': 38.9908, 'lon': -76.9481, 'is_dome': False, 'alt': 100},
    'michigan': {'lat': 42.2656, 'lon': -83.7486, 'is_dome': False, 'alt': 900},
    'michigan state': {'lat': 42.7283, 'lon': -84.4803, 'is_dome': False, 'alt': 900},
    'minnesota': {'lat': 44.9764, 'lon': -93.2244, 'is_dome': False, 'alt': 850},
    'nebraska': {'lat': 40.8203, 'lon': -96.7058, 'is_dome': False, 'alt': 1200},
    'northwestern': {'lat': 42.0650, 'lon': -87.6983, 'is_dome': False, 'alt': 600},
    'ohio state': {'lat': 40.0017, 'lon': -83.0197, 'is_dome': False, 'alt': 800},
    'penn state': {'lat': 40.8122, 'lon': -77.8567, 'is_dome': False, 'alt': 1200},
    'purdue': {'lat': 40.4417, 'lon': -86.9189, 'is_dome': False, 'alt': 700},
    'rutgers': {'lat': 40.5137, 'lon': -74.4647, 'is_dome': False, 'alt': 100},
    'wisconsin': {'lat': 43.0700, 'lon': -89.4128, 'is_dome': False, 'alt': 900},
    'ucla': {'lat': 34.1614, 'lon': -118.1675, 'is_dome': False, 'alt': 300},
    'usc': {'lat': 34.0141, 'lon': -118.2879, 'is_dome': False, 'alt': 300},
    'oregon': {'lat': 44.0582, 'lon': -123.0686, 'is_dome': False, 'alt': 450},
    'washington': {'lat': 47.6506, 'lon': -122.3017, 'is_dome': False, 'alt': 100},

    # Big 12
    'baylor': {'lat': 31.5589, 'lon': -97.1153, 'is_dome': False, 'alt': 500},
    'byu': {'lat': 40.2578, 'lon': -111.6544, 'is_dome': False, 'alt': 4550},  # High altitude!
    'cincinnati': {'lat': 39.1319, 'lon': -84.5167, 'is_dome': False, 'alt': 500},
    'colorado': {'lat': 40.0092, 'lon': -105.2669, 'is_dome': False, 'alt': 5430},  # High altitude!
    'houston': {'lat': 29.7222, 'lon': -95.3489, 'is_dome': False, 'alt': 50},
    'iowa state': {'lat': 42.0140, 'lon': -93.6358, 'is_dome': False, 'alt': 1000},
    'kansas': {'lat': 38.9361, 'lon': -95.2525, 'is_dome': False, 'alt': 900},
    'kansas state': {'lat': 39.2014, 'lon': -96.5936, 'is_dome': False, 'alt': 1100},
    'oklahoma state': {'lat': 36.1261, 'lon': -97.0664, 'is_dome': False, 'alt': 1000},
    'tcu': {'lat': 32.7092, 'lon': -97.3683, 'is_dome': False, 'alt': 650},
    'texas tech': {'lat': 33.5906, 'lon': -101.8728, 'is_dome': False, 'alt': 3250},
    'ucf': {'lat': 28.6078, 'lon': -81.1917, 'is_dome': False, 'alt': 100},
    'west virginia': {'lat': 39.6500, 'lon': -79.9550, 'is_dome': False, 'alt': 1000},
    'arizona': {'lat': 32.2289, 'lon': -110.9486, 'is_dome': False, 'alt': 2400},
    'arizona state': {'lat': 33.4256, 'lon': -111.9325, 'is_dome': False, 'alt': 1100},
    'utah': {'lat': 40.7600, 'lon': -111.8481, 'is_dome': False, 'alt': 4700},  # High altitude!

    # ACC
    'boston college': {'lat': 42.3353, 'lon': -71.1667, 'is_dome': False, 'alt': 300},
    'clemson': {'lat': 34.6786, 'lon': -82.8442, 'is_dome': False, 'alt': 850},
    'duke': {'lat': 35.9994, 'lon': -78.9428, 'is_dome': False, 'alt': 400},
    'florida state': {'lat': 30.4383, 'lon': -84.3044, 'is_dome': False, 'alt': 200},
    'georgia tech': {'lat': 33.7725, 'lon': -84.3928, 'is_dome': False, 'alt': 1000},
    'louisville': {'lat': 38.2069, 'lon': -85.7583, 'is_dome': False, 'alt': 450},
    'miami': {'lat': 25.9581, 'lon': -80.2389, 'is_dome': False, 'alt': 10},
    'nc state': {'lat': 35.7864, 'lon': -78.7114, 'is_dome': False, 'alt': 400},
    'north carolina': {'lat': 35.9050, 'lon': -79.0478, 'is_dome': False, 'alt': 500},
    'pitt': {'lat': 40.4444, 'lon': -79.9533, 'is_dome': False, 'alt': 1200},
    'syracuse': {'lat': 43.0361, 'lon': -76.1364, 'is_dome': True, 'alt': 400},  # DOME
    'virginia': {'lat': 38.0314, 'lon': -78.5133, 'is_dome': False, 'alt': 500},
    'virginia tech': {'lat': 37.2200, 'lon': -80.4181, 'is_dome': False, 'alt': 2100},
    'wake forest': {'lat': 36.1297, 'lon': -80.2511, 'is_dome': False, 'alt': 1000},
    'stanford': {'lat': 37.4346, 'lon': -122.1609, 'is_dome': False, 'alt': 100},
    'california': {'lat': 37.8708, 'lon': -122.2506, 'is_dome': False, 'alt': 200},
    'notre dame': {'lat': 41.6983, 'lon': -86.2339, 'is_dome': False, 'alt': 750},
    'smu': {'lat': 32.8361, 'lon': -96.7833, 'is_dome': False, 'alt': 500},

    # Domes
    'new mexico': {'lat': 35.0492, 'lon': -106.6194, 'is_dome': True, 'alt': 5100},  # The Pit
    'louisiana tech': {'lat': 32.5286, 'lon': -92.6378, 'is_dome': True, 'alt': 250},
    'georgia state': {'lat': 33.7550, 'lon': -84.4003, 'is_dome': True, 'alt': 1000},  # Shares with Falcons
    'tulane': {'lat': 29.9508, 'lon': -90.0806, 'is_dome': True, 'alt': 10},  # Superdome

    # Other notable
    'air force': {'lat': 38.9977, 'lon': -104.8436, 'is_dome': False, 'alt': 6621},  # Highest altitude!
    'army': {'lat': 41.3906, 'lon': -73.9653, 'is_dome': False, 'alt': 200},
    'navy': {'lat': 38.9847, 'lon': -76.4883, 'is_dome': False, 'alt': 50},
    'boise state': {'lat': 43.6028, 'lon': -116.1972, 'is_dome': False, 'alt': 2700},
    'fresno state': {'lat': 36.8133, 'lon': -119.7506, 'is_dome': False, 'alt': 300},
    'san diego state': {'lat': 32.7831, 'lon': -117.1536, 'is_dome': False, 'alt': 400},
    'unlv': {'lat': 36.0867, 'lon': -115.1572, 'is_dome': True, 'alt': 2000},  # Allegiant Stadium
    'hawaii': {'lat': 21.3747, 'lon': -157.9303, 'is_dome': False, 'alt': 50},
    'wyoming': {'lat': 41.1336, 'lon': -105.5703, 'is_dome': False, 'alt': 7220},  # High altitude!
    'colorado state': {'lat': 40.5761, 'lon': -105.0842, 'is_dome': False, 'alt': 5000},
}

# Weather impact thresholds
WEATHER_THRESHOLDS = {
    'high_wind': 15,  # mph - affects passing
    'extreme_wind': 25,  # mph - major impact
    'cold_temp': 40,  # F - affects ball handling
    'extreme_cold': 25,  # F - major impact
    'rain_threshold': 0.1,  # inches/hour - moderate rain
    'heavy_rain': 0.3,  # inches/hour - heavy rain
    'high_altitude': 4500,  # ft - noticeable impact
    'extreme_altitude': 5500,  # ft - major 2nd half impact
}


def get_stadium_info(team_name):
    """Get stadium information for a team."""
    name_lower = team_name.lower().strip()

    # Direct lookup
    if name_lower in STADIUMS:
        return STADIUMS[name_lower]

    # Try partial matches
    for key, info in STADIUMS.items():
        if name_lower in key or key in name_lower:
            return info

    # Try common variations
    variations = {
        'mississippi': 'ole miss',
        'ohio': 'ohio state',
        'penn': 'penn state',
        'michigan st': 'michigan state',
        'florida st': 'florida state',
        'san jose st': 'san jose state',
        'north carolina state': 'nc state',
        'pittsburgh': 'pitt',
        'southern california': 'usc',
    }

    for var, actual in variations.items():
        if var in name_lower:
            if actual in STADIUMS:
                return STADIUMS[actual]

    return None


# =============================================================================
# WEATHER.GOV API
# =============================================================================
def fetch_weather_forecast(lat, lon):
    """
    Fetch weather forecast from weather.gov API.

    Returns hourly forecast data for the location.
    """
    headers = {
        'User-Agent': 'CFB-Predictor-App (contact@example.com)',
        'Accept': 'application/geo+json'
    }

    # First, get the forecast grid endpoint
    points_url = f"https://api.weather.gov/points/{lat},{lon}"

    try:
        resp = requests.get(points_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        forecast_url = data.get('properties', {}).get('forecastHourly')

        if not forecast_url:
            return None

        # Fetch hourly forecast
        forecast_resp = requests.get(forecast_url, headers=headers, timeout=10)
        if forecast_resp.status_code != 200:
            return None

        forecast_data = forecast_resp.json()
        periods = forecast_data.get('properties', {}).get('periods', [])

        return periods

    except Exception as e:
        print(f"Weather API error: {e}")
        return None


def get_game_weather(lat, lon, game_datetime):
    """
    Get weather conditions for a specific game time.

    Args:
        lat: Stadium latitude
        lon: Stadium longitude
        game_datetime: datetime of game kickoff

    Returns:
        dict with weather conditions
    """
    periods = fetch_weather_forecast(lat, lon)

    if not periods:
        return None

    # Find the period closest to game time
    target_time = game_datetime
    best_period = None
    min_diff = timedelta(hours=24)

    for period in periods:
        try:
            period_time = datetime.fromisoformat(period['startTime'].replace('Z', '+00:00'))
            period_time = period_time.replace(tzinfo=None)  # Make naive for comparison

            diff = abs(period_time - target_time)
            if diff < min_diff:
                min_diff = diff
                best_period = period
        except Exception:
            continue

    if not best_period:
        return None

    # Extract weather data
    weather = {
        'temperature': best_period.get('temperature', 70),
        'temperature_unit': best_period.get('temperatureUnit', 'F'),
        'wind_speed': 0,
        'wind_direction': '',
        'precipitation_probability': best_period.get('probabilityOfPrecipitation', {}).get('value', 0) or 0,
        'conditions': best_period.get('shortForecast', 'Unknown'),
        'is_rainy': False,
        'is_snowy': False,
    }

    # Parse wind speed
    wind_str = best_period.get('windSpeed', '0 mph')
    try:
        wind_parts = wind_str.split()
        if len(wind_parts) >= 1:
            # Handle ranges like "10 to 15 mph"
            wind_nums = [int(x) for x in wind_parts if x.isdigit()]
            if wind_nums:
                weather['wind_speed'] = max(wind_nums)  # Use higher end
    except Exception:
        weather['wind_speed'] = 0

    weather['wind_direction'] = best_period.get('windDirection', '')

    # Check for rain/snow
    conditions_lower = weather['conditions'].lower()
    weather['is_rainy'] = any(word in conditions_lower for word in ['rain', 'shower', 'thunder', 'storm'])
    weather['is_snowy'] = any(word in conditions_lower for word in ['snow', 'sleet', 'flurries', 'blizzard'])

    return weather


def calculate_weather_impact(weather, altitude=0):
    """
    Calculate weather impact on game in points.

    Positive = favors home team (they're more used to it)
    Negative = neutral impact

    Returns tuple: (home_adjustment, weather_flags dict)
    """
    if not weather:
        return 0, {}

    impact = 0
    flags = {
        'high_wind': False,
        'extreme_wind': False,
        'cold': False,
        'extreme_cold': False,
        'rain': False,
        'heavy_rain': False,
        'snow': False,
        'high_altitude': False,
        'extreme_altitude': False,
    }

    # Wind impact
    wind = weather.get('wind_speed', 0)
    if wind >= WEATHER_THRESHOLDS['extreme_wind']:
        flags['extreme_wind'] = True
        impact -= 0  # Neutral - both teams affected
    elif wind >= WEATHER_THRESHOLDS['high_wind']:
        flags['high_wind'] = True
        impact -= 0  # Neutral

    # Temperature impact
    temp = weather.get('temperature', 70)
    if temp <= WEATHER_THRESHOLDS['extreme_cold']:
        flags['extreme_cold'] = True
        impact += 1  # Home team slight advantage (used to conditions)
    elif temp <= WEATHER_THRESHOLDS['cold_temp']:
        flags['cold'] = True
        impact += 0.5

    # Precipitation
    if weather.get('is_snowy'):
        flags['snow'] = True
        impact += 1.5  # Home team advantage in snow

    if weather.get('is_rainy'):
        precip_prob = weather.get('precipitation_probability', 0)
        if precip_prob >= 70:
            flags['heavy_rain'] = True
            impact += 0.5  # Slight home advantage
        elif precip_prob >= 40:
            flags['rain'] = True

    # Altitude impact
    if altitude >= WEATHER_THRESHOLDS['extreme_altitude']:
        flags['extreme_altitude'] = True
        impact += 2  # Significant home advantage at extreme altitude
    elif altitude >= WEATHER_THRESHOLDS['high_altitude']:
        flags['high_altitude'] = True
        impact += 1  # Home team advantage at altitude

    return impact, flags


def generate_weather_features(home_team, away_team, game_datetime, is_dome_override=None):
    """
    Generate weather features for a matchup.

    Args:
        home_team: Home team name
        away_team: Away team name
        game_datetime: datetime of kickoff
        is_dome_override: If True/False, override dome status

    Returns:
        dict of weather features
    """
    stadium = get_stadium_info(home_team)

    if not stadium:
        # No stadium data - return neutral
        return {
            'is_dome': 0,
            'temperature': 70,
            'wind_speed': 0,
            'precipitation_prob': 0,
            'is_cold': 0,
            'is_windy': 0,
            'is_rainy': 0,
            'is_snowy': 0,
            'high_altitude': 0,
            'weather_home_advantage': 0,
        }

    is_dome = is_dome_override if is_dome_override is not None else stadium.get('is_dome', False)

    if is_dome:
        # Dome game - weather doesn't matter
        return {
            'is_dome': 1,
            'temperature': 72,  # Controlled
            'wind_speed': 0,
            'precipitation_prob': 0,
            'is_cold': 0,
            'is_windy': 0,
            'is_rainy': 0,
            'is_snowy': 0,
            'high_altitude': 0,
            'weather_home_advantage': 0,
        }

    # Get weather
    weather = get_game_weather(stadium['lat'], stadium['lon'], game_datetime)

    if not weather:
        # Can't get weather - use altitude only
        altitude = stadium.get('alt', 0)
        return {
            'is_dome': 0,
            'temperature': 70,
            'wind_speed': 0,
            'precipitation_prob': 0,
            'is_cold': 0,
            'is_windy': 0,
            'is_rainy': 0,
            'is_snowy': 0,
            'high_altitude': 1 if altitude >= WEATHER_THRESHOLDS['high_altitude'] else 0,
            'weather_home_advantage': 1 if altitude >= WEATHER_THRESHOLDS['high_altitude'] else 0,
        }

    altitude = stadium.get('alt', 0)
    home_advantage, flags = calculate_weather_impact(weather, altitude)

    return {
        'is_dome': 0,
        'temperature': weather.get('temperature', 70),
        'wind_speed': weather.get('wind_speed', 0),
        'precipitation_prob': weather.get('precipitation_probability', 0),
        'is_cold': 1 if flags.get('cold') or flags.get('extreme_cold') else 0,
        'is_windy': 1 if flags.get('high_wind') or flags.get('extreme_wind') else 0,
        'is_rainy': 1 if flags.get('rain') or flags.get('heavy_rain') else 0,
        'is_snowy': 1 if flags.get('snow') else 0,
        'high_altitude': 1 if flags.get('high_altitude') or flags.get('extreme_altitude') else 0,
        'weather_home_advantage': home_advantage,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Test weather fetching for sample games."""
    print("=" * 70)
    print("WEATHER DATA FETCHER TEST")
    print("=" * 70)

    # Test a few stadiums
    test_teams = ['ohio state', 'colorado', 'alabama', 'michigan', 'byu', 'syracuse']

    for team in test_teams:
        print(f"\n{team.upper()}")
        stadium = get_stadium_info(team)
        if stadium:
            print(f"  Location: ({stadium['lat']}, {stadium['lon']})")
            print(f"  Altitude: {stadium['alt']} ft")
            print(f"  Dome: {stadium['is_dome']}")

            # Get current weather
            weather = get_game_weather(stadium['lat'], stadium['lon'], datetime.now())
            if weather:
                print(f"  Current: {weather['temperature']}F, Wind: {weather['wind_speed']} mph")
                print(f"  Conditions: {weather['conditions']}")
        else:
            print("  Stadium not found!")

    print("\n" + "=" * 70)
    print("Testing generate_weather_features()...")
    print("=" * 70)

    # Test feature generation
    features = generate_weather_features(
        'ohio state', 'michigan',
        datetime.now() + timedelta(days=3)
    )
    print(f"\nOhio State vs Michigan:")
    for key, val in features.items():
        print(f"  {key}: {val}")

    features = generate_weather_features(
        'colorado', 'utah',
        datetime.now() + timedelta(days=3)
    )
    print(f"\nColorado vs Utah (high altitude):")
    for key, val in features.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
