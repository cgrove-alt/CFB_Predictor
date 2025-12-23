"""
Fetch College Football Injury Data from ESPN and Covers.com.

This script fetches injury reports for CFB teams to incorporate
into predictions. QB injuries are worth 5-10 points on the spread.

Sources:
1. ESPN Site API (primary) - http://site.api.espn.com/apis/site/v2/sports/football/college-football/
2. Covers.com Injury Reports (backup) - https://www.covers.com/sport/football/ncaaf/injuries

Usage:
    python fetch_injuries.py
"""

import requests
import pandas as pd
import json
import time
import re
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup

# =============================================================================
# ESPN API CONFIGURATION
# =============================================================================
ESPN_BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/football/college-football"
ESPN_CORE_URL = "http://sports.core.api.espn.com/v2/sports/football/leagues/college-football"

# Team ID mapping - ESPN uses numeric IDs
# This will be built dynamically from the ESPN teams endpoint
TEAM_ID_CACHE = {}

# Injury impact estimates (in points)
INJURY_IMPACT = {
    'QB': {'out': -7, 'doubtful': -5, 'questionable': -2, 'probable': -0.5},
    'RB1': {'out': -2, 'doubtful': -1.5, 'questionable': -0.5, 'probable': 0},
    'WR1': {'out': -1.5, 'doubtful': -1, 'questionable': -0.5, 'probable': 0},
    'OL': {'out': -1, 'doubtful': -0.5, 'questionable': -0.25, 'probable': 0},
    'DL': {'out': -0.5, 'doubtful': -0.25, 'questionable': 0, 'probable': 0},
    'LB': {'out': -0.5, 'doubtful': -0.25, 'questionable': 0, 'probable': 0},
    'DB': {'out': -0.5, 'doubtful': -0.25, 'questionable': 0, 'probable': 0},
    'K': {'out': -0.5, 'doubtful': -0.25, 'questionable': 0, 'probable': 0},
    'DEFAULT': {'out': -0.25, 'doubtful': -0.1, 'questionable': 0, 'probable': 0},
}

# Status normalization
STATUS_MAP = {
    'out': 'out',
    'o': 'out',
    'injured reserve': 'out',
    'ir': 'out',
    'suspension': 'out',
    'doubtful': 'doubtful',
    'd': 'doubtful',
    'questionable': 'questionable',
    'q': 'questionable',
    'probable': 'probable',
    'p': 'probable',
    'day-to-day': 'questionable',
    'limited': 'questionable',
    'full': 'probable',
}


def normalize_status(status_str):
    """Normalize injury status to standard categories."""
    if not status_str:
        return 'questionable'
    status_lower = status_str.lower().strip()
    return STATUS_MAP.get(status_lower, 'questionable')


def get_position_category(position):
    """Map detailed position to category for impact calculation."""
    if not position:
        return 'DEFAULT'
    pos = position.upper()
    if pos in ['QB']:
        return 'QB'
    elif pos in ['RB', 'FB']:
        return 'RB1'
    elif pos in ['WR', 'TE']:
        return 'WR1'
    elif pos in ['OT', 'OG', 'C', 'OL']:
        return 'OL'
    elif pos in ['DT', 'DE', 'DL', 'NT']:
        return 'DL'
    elif pos in ['LB', 'ILB', 'OLB', 'MLB']:
        return 'LB'
    elif pos in ['CB', 'S', 'FS', 'SS', 'DB']:
        return 'DB'
    elif pos in ['K', 'P', 'PK']:
        return 'K'
    return 'DEFAULT'


def calculate_injury_impact(injuries):
    """
    Calculate total injury impact in points for a team.

    Returns negative number (injuries hurt the team).
    """
    total_impact = 0

    for injury in injuries:
        position = injury.get('position', '')
        status = normalize_status(injury.get('status', ''))

        pos_category = get_position_category(position)
        impact = INJURY_IMPACT.get(pos_category, INJURY_IMPACT['DEFAULT'])

        total_impact += impact.get(status, 0)

    return total_impact


# =============================================================================
# ESPN API FETCHING
# =============================================================================
def fetch_espn_teams():
    """Fetch all FBS teams and their ESPN IDs."""
    global TEAM_ID_CACHE

    if TEAM_ID_CACHE:
        return TEAM_ID_CACHE

    print("Fetching ESPN team IDs...")

    # FBS group is 80
    url = f"{ESPN_BASE_URL}/teams?limit=200&groups=80"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            teams = data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])

            for team in teams:
                team_info = team.get('team', {})
                team_id = team_info.get('id')
                team_name = team_info.get('displayName', '')
                abbreviation = team_info.get('abbreviation', '')

                if team_id and team_name:
                    TEAM_ID_CACHE[team_name.lower()] = team_id
                    TEAM_ID_CACHE[abbreviation.lower()] = team_id
                    # Also add variations
                    TEAM_ID_CACHE[team_name.lower().replace(' ', '')] = team_id

            print(f"  Found {len(teams)} teams")
            return TEAM_ID_CACHE
        else:
            print(f"  Error: {resp.status_code}")
    except Exception as e:
        print(f"  Exception: {e}")

    return {}


def get_team_id(team_name):
    """Get ESPN team ID from team name."""
    if not TEAM_ID_CACHE:
        fetch_espn_teams()

    name_lower = team_name.lower().strip()

    # Try exact match first
    if name_lower in TEAM_ID_CACHE:
        return TEAM_ID_CACHE[name_lower]

    # Try without spaces
    name_no_space = name_lower.replace(' ', '')
    if name_no_space in TEAM_ID_CACHE:
        return TEAM_ID_CACHE[name_no_space]

    # Try partial match
    for cached_name, team_id in TEAM_ID_CACHE.items():
        if name_lower in cached_name or cached_name in name_lower:
            return team_id

    return None


def fetch_team_injuries_espn(team_name):
    """
    Fetch injuries for a specific team from ESPN.

    Returns list of injury dicts with player, position, status, injury type.
    """
    team_id = get_team_id(team_name)
    if not team_id:
        return []

    # Try the core API endpoint for injuries
    url = f"{ESPN_CORE_URL}/teams/{team_id}/injuries"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get('items', [])

            injuries = []
            for item in items:
                # Each item might be a reference URL or actual data
                if '$ref' in item:
                    # Need to fetch the actual injury data
                    injury_resp = requests.get(item['$ref'], timeout=5)
                    if injury_resp.status_code == 200:
                        injury_data = injury_resp.json()
                        athlete_ref = injury_data.get('athlete', {}).get('$ref', '')
                        if athlete_ref:
                            athlete_resp = requests.get(athlete_ref, timeout=5)
                            if athlete_resp.status_code == 200:
                                athlete = athlete_resp.json()
                                injuries.append({
                                    'player': athlete.get('displayName', 'Unknown'),
                                    'position': athlete.get('position', {}).get('abbreviation', ''),
                                    'status': injury_data.get('status', 'Unknown'),
                                    'injury': injury_data.get('type', {}).get('description', 'Unknown'),
                                    'team': team_name
                                })
                else:
                    injuries.append({
                        'player': item.get('athlete', {}).get('displayName', 'Unknown'),
                        'position': item.get('athlete', {}).get('position', {}).get('abbreviation', ''),
                        'status': item.get('status', 'Unknown'),
                        'injury': item.get('type', {}).get('description', 'Unknown'),
                        'team': team_name
                    })

            return injuries
        elif resp.status_code == 404:
            # Try alternative endpoint (roster with injury status)
            return fetch_team_roster_injuries(team_id, team_name)
    except Exception as e:
        print(f"  ESPN API error for {team_name}: {e}")

    return []


def fetch_team_roster_injuries(team_id, team_name):
    """Fallback: Check roster for injured players."""
    url = f"{ESPN_BASE_URL}/teams/{team_id}/roster"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            injuries = []

            for group in data.get('athletes', []):
                for athlete in group.get('items', []):
                    # Check for injury status in the athlete data
                    status = athlete.get('injuries', [])
                    if status:
                        injuries.append({
                            'player': athlete.get('displayName', 'Unknown'),
                            'position': athlete.get('position', {}).get('abbreviation', ''),
                            'status': status[0].get('status', 'Unknown'),
                            'injury': status[0].get('type', {}).get('description', 'Unknown'),
                            'team': team_name
                        })

            return injuries
    except Exception as e:
        print(f"  Roster fallback error for {team_name}: {e}")

    return []


# =============================================================================
# COVERS.COM SCRAPING (BACKUP)
# =============================================================================
def fetch_covers_injuries():
    """
    Scrape injury reports from Covers.com as backup source.

    Returns dict mapping team names to list of injuries.
    """
    print("Fetching injuries from Covers.com...")

    url = "https://www.covers.com/sport/football/ncaaf/injuries"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"  Covers.com returned {resp.status_code}")
            return {}

        soup = BeautifulSoup(resp.text, 'html.parser')
        team_injuries = {}

        # Find team sections
        team_sections = soup.find_all('div', class_=re.compile(r'team-injuries|injury-report'))

        for section in team_sections:
            # Get team name
            team_header = section.find(['h2', 'h3', 'div'], class_=re.compile(r'team-name|header'))
            if not team_header:
                continue

            team_name = team_header.get_text(strip=True)
            injuries = []

            # Find player rows
            rows = section.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    player = cells[0].get_text(strip=True)
                    position = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                    status = cells[2].get_text(strip=True) if len(cells) > 2 else ''
                    injury_type = cells[3].get_text(strip=True) if len(cells) > 3 else ''

                    if player and status:
                        injuries.append({
                            'player': player,
                            'position': position,
                            'status': status,
                            'injury': injury_type,
                            'team': team_name
                        })

            if injuries:
                team_injuries[team_name.lower()] = injuries

        print(f"  Found injuries for {len(team_injuries)} teams")
        return team_injuries

    except Exception as e:
        print(f"  Covers.com scraping error: {e}")
        return {}


# =============================================================================
# MAIN FETCH FUNCTION
# =============================================================================
def fetch_all_injuries(teams=None):
    """
    Fetch injury data for all teams or specified teams.

    Args:
        teams: List of team names to fetch. If None, fetches all FBS teams.

    Returns:
        DataFrame with columns: team, player, position, status, injury, impact
    """
    print("=" * 70)
    print("FETCHING COLLEGE FOOTBALL INJURY DATA")
    print("=" * 70)

    all_injuries = []

    # Get team list
    if teams is None:
        fetch_espn_teams()
        teams = list(set(TEAM_ID_CACHE.keys()))

    # Try ESPN first for each team
    print(f"\nFetching from ESPN API for {len(teams)} teams...")

    espn_success = 0
    for i, team in enumerate(teams):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(teams)}")

        injuries = fetch_team_injuries_espn(team)
        if injuries:
            all_injuries.extend(injuries)
            espn_success += 1

        time.sleep(0.1)  # Rate limiting

    print(f"\n  ESPN: Found injuries for {espn_success} teams")

    # Supplement with Covers.com
    covers_injuries = fetch_covers_injuries()

    for team_name, injuries in covers_injuries.items():
        # Only add if not already in ESPN data
        existing_teams = {inj['team'].lower() for inj in all_injuries}
        if team_name.lower() not in existing_teams:
            all_injuries.extend(injuries)

    # Create DataFrame
    if not all_injuries:
        print("\nNo injuries found!")
        return pd.DataFrame()

    df = pd.DataFrame(all_injuries)

    # Normalize status
    df['status_normalized'] = df['status'].apply(normalize_status)

    # Calculate impact
    df['position_category'] = df['position'].apply(get_position_category)
    df['impact'] = df.apply(
        lambda row: INJURY_IMPACT.get(
            row['position_category'],
            INJURY_IMPACT['DEFAULT']
        ).get(row['status_normalized'], 0),
        axis=1
    )

    print(f"\nTotal injuries found: {len(df)}")
    print(f"Teams with injuries: {df['team'].nunique()}")

    return df


def get_team_injury_impact(team_name, injury_df=None):
    """
    Get total injury impact for a specific team.

    Returns:
        float: Total impact (negative = injuries hurt team)
    """
    if injury_df is None:
        injury_df = fetch_all_injuries([team_name])

    team_injuries = injury_df[injury_df['team'].str.lower() == team_name.lower()]

    if team_injuries.empty:
        return 0.0

    return team_injuries['impact'].sum()


def generate_injury_features(home_team, away_team, injury_df=None):
    """
    Generate injury-based features for a matchup.

    Returns:
        dict: Features to add to prediction
    """
    if injury_df is None:
        injury_df = fetch_all_injuries([home_team, away_team])

    home_impact = get_team_injury_impact(home_team, injury_df)
    away_impact = get_team_injury_impact(away_team, injury_df)

    # Check for QB injuries specifically
    home_qb_out = len(injury_df[
        (injury_df['team'].str.lower() == home_team.lower()) &
        (injury_df['position_category'] == 'QB') &
        (injury_df['status_normalized'] == 'out')
    ]) > 0

    away_qb_out = len(injury_df[
        (injury_df['team'].str.lower() == away_team.lower()) &
        (injury_df['position_category'] == 'QB') &
        (injury_df['status_normalized'] == 'out')
    ]) > 0

    return {
        'home_injury_impact': home_impact,
        'away_injury_impact': away_impact,
        'injury_diff': home_impact - away_impact,  # Negative = home more hurt
        'home_qb_out': int(home_qb_out),
        'away_qb_out': int(away_qb_out),
        'qb_advantage': int(away_qb_out) - int(home_qb_out),  # Positive = home has QB advantage
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Fetch all injuries and save to CSV."""
    df = fetch_all_injuries()

    if not df.empty:
        # Save raw injuries
        output_file = 'cfb_injuries.csv'
        df.to_csv(output_file, index=False)
        print(f"\nSaved injuries to {output_file}")

        # Save summary by team
        summary = df.groupby('team').agg({
            'player': 'count',
            'impact': 'sum'
        }).rename(columns={'player': 'num_injuries', 'impact': 'total_impact'})
        summary = summary.sort_values('total_impact')

        summary_file = 'cfb_injury_summary.csv'
        summary.to_csv(summary_file)
        print(f"Saved summary to {summary_file}")

        # Print top injured teams
        print("\n" + "=" * 70)
        print("MOST IMPACTED TEAMS (by injury)")
        print("=" * 70)
        print(summary.head(20).to_string())

        # Save timestamp
        status = {
            'last_fetch': datetime.now().isoformat(),
            'total_injuries': len(df),
            'teams_affected': df['team'].nunique()
        }
        with open('.injury_status.json', 'w') as f:
            json.dump(status, f, indent=2)

    return df


if __name__ == "__main__":
    main()
