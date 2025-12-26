"""
FBS Teams Allowlist for Sharp Sports Predictor

This module provides a utility function to get the list of FBS (Football Bowl Subdivision)
teams for a given season. The model only trains and predicts on FBS vs FBS matchups.

The Golden Rule: If a team is NOT in this allowlist, it is NOT an FBS team and
should be excluded from training data and predictions.

Last updated: December 2024 (2024-2025 season)
"""

from typing import Set, Dict, Optional

# =============================================================================
# FBS TEAMS BY CONFERENCE (2024-2025 Season)
# =============================================================================

# Power 4 Conferences (P4)
SEC_TEAMS = {
    "Alabama", "Arkansas", "Auburn", "Florida", "Georgia",
    "Kentucky", "LSU", "Mississippi State", "Missouri", "Ole Miss",
    "South Carolina", "Tennessee", "Texas", "Texas A&M", "Vanderbilt",
    "Oklahoma"
}

BIG_TEN_TEAMS = {
    "Illinois", "Indiana", "Iowa", "Maryland", "Michigan",
    "Michigan State", "Minnesota", "Nebraska", "Northwestern", "Ohio State",
    "Oregon", "Penn State", "Purdue", "Rutgers", "UCLA",
    "USC", "Washington", "Wisconsin"
}

BIG_12_TEAMS = {
    "Arizona", "Arizona State", "Baylor", "BYU", "Cincinnati",
    "Colorado", "Houston", "Iowa State", "Kansas", "Kansas State",
    "Oklahoma State", "TCU", "Texas Tech", "UCF", "Utah",
    "West Virginia"
}

ACC_TEAMS = {
    "Boston College", "California", "Clemson", "Duke", "Florida State",
    "Georgia Tech", "Louisville", "Miami", "NC State", "North Carolina",
    "Pittsburgh", "SMU", "Stanford", "Syracuse", "Virginia",
    "Virginia Tech", "Wake Forest"
}

# Group of 5 Conferences (G5)
AAC_TEAMS = {
    "Charlotte", "East Carolina", "FAU", "Memphis", "Navy",
    "North Texas", "Rice", "South Florida", "Temple", "Tulane",
    "Tulsa", "UAB", "UTSA"
}

MOUNTAIN_WEST_TEAMS = {
    "Air Force", "Boise State", "Colorado State", "Fresno State", "Hawaii",
    "Nevada", "New Mexico", "San Diego State", "San José State", "UNLV",
    "Utah State", "Wyoming"
}

SUN_BELT_TEAMS = {
    "Appalachian State", "Arkansas State", "Coastal Carolina", "Georgia Southern",
    "Georgia State", "James Madison", "Louisiana", "Marshall",
    "Old Dominion", "South Alabama", "Southern Miss", "Texas State",
    "Troy", "UL Monroe"
}

MAC_TEAMS = {
    "Akron", "Ball State", "Bowling Green", "Buffalo", "Central Michigan",
    "Eastern Michigan", "Kent State", "Miami (OH)", "Northern Illinois",
    "Ohio", "Toledo", "Western Michigan"
}

CUSA_TEAMS = {
    "FIU", "Jacksonville State", "Kennesaw State", "Liberty",
    "Louisiana Tech", "Middle Tennessee", "New Mexico State",
    "Sam Houston State", "UTEP", "Western Kentucky"
}

# Independents
INDEPENDENT_TEAMS = {
    "Army", "Notre Dame", "UConn", "UMass"
}

# =============================================================================
# CONFERENCE GROUPINGS
# =============================================================================

POWER_4_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC"}
GROUP_OF_5_CONFERENCES = {"AAC", "Mountain West", "Sun Belt", "MAC", "CUSA"}

# Conference name to team set mapping
CONFERENCE_TEAMS: Dict[str, Set[str]] = {
    "SEC": SEC_TEAMS,
    "Big Ten": BIG_TEN_TEAMS,
    "Big 12": BIG_12_TEAMS,
    "ACC": ACC_TEAMS,
    "AAC": AAC_TEAMS,
    "Mountain West": MOUNTAIN_WEST_TEAMS,
    "Sun Belt": SUN_BELT_TEAMS,
    "MAC": MAC_TEAMS,
    "CUSA": CUSA_TEAMS,
    "Independent": INDEPENDENT_TEAMS,
}

# Team to conference mapping (reverse lookup)
TEAM_TO_CONFERENCE: Dict[str, str] = {}
for conf, teams in CONFERENCE_TEAMS.items():
    for team in teams:
        TEAM_TO_CONFERENCE[team] = conf

# =============================================================================
# TEAM NAME ALIASES (for matching various data sources)
# =============================================================================

TEAM_ALIASES: Dict[str, str] = {
    # Common variations
    "San Jose State": "San José State",
    "SJSU": "San José State",
    "San Jose St": "San José State",
    "San Jose St.": "San José State",
    "Miami (FL)": "Miami",
    "Miami FL": "Miami",
    "Miami-FL": "Miami",
    "Miami Ohio": "Miami (OH)",
    "Miami-OH": "Miami (OH)",
    "UL-Monroe": "UL Monroe",
    "Louisiana-Monroe": "UL Monroe",
    "ULM": "UL Monroe",
    "Louisiana-Lafayette": "Louisiana",
    "UL Lafayette": "Louisiana",
    "ULL": "Louisiana",
    "Louisiana Lafayette": "Louisiana",
    "Southern Mississippi": "Southern Miss",
    "Southern Miss.": "Southern Miss",
    "USM": "Southern Miss",
    "UT San Antonio": "UTSA",
    "Texas-San Antonio": "UTSA",
    "UT-San Antonio": "UTSA",
    "Florida Atlantic": "FAU",
    "Florida Int'l": "FIU",
    "Florida International": "FIU",
    "Central Florida": "UCF",
    "South Florida Bulls": "South Florida",
    "USF": "South Florida",
    "Brigham Young": "BYU",
    "NC St": "NC State",
    "NC St.": "NC State",
    "North Carolina State": "NC State",
    "N.C. State": "NC State",
    "NCSU": "NC State",
    "Pitt": "Pittsburgh",
    "Ole Miss": "Ole Miss",  # Already correct but explicit
    "Mississippi": "Ole Miss",
    "Miss": "Ole Miss",
    "Miss State": "Mississippi State",
    "Miss St": "Mississippi State",
    "Miss St.": "Mississippi State",
    "MSU": "Mississippi State",  # Context-dependent, usually Miss State in SEC
    "Texas-El Paso": "UTEP",
    "MTSU": "Middle Tennessee",
    "Middle Tennessee State": "Middle Tennessee",
    "NMSU": "New Mexico State",
    "App State": "Appalachian State",
    "App St": "Appalachian State",
    "App St.": "Appalachian State",
    "Coastal": "Coastal Carolina",
    "CCU": "Coastal Carolina",
    "W Kentucky": "Western Kentucky",
    "W. Kentucky": "Western Kentucky",
    "WKU": "Western Kentucky",
    "E Michigan": "Eastern Michigan",
    "E. Michigan": "Eastern Michigan",
    "EMU": "Eastern Michigan",
    "W Michigan": "Western Michigan",
    "W. Michigan": "Western Michigan",
    "WMU": "Western Michigan",
    "C Michigan": "Central Michigan",
    "C. Michigan": "Central Michigan",
    "CMU": "Central Michigan",
    "N Illinois": "Northern Illinois",
    "N. Illinois": "Northern Illinois",
    "NIU": "Northern Illinois",
    "BGSU": "Bowling Green",
    "Bowling Green State": "Bowling Green",
    "Ga Southern": "Georgia Southern",
    "Ga. Southern": "Georgia Southern",
    "GSU": "Georgia State",  # Also could be Ga Southern in some contexts
    "Ga State": "Georgia State",
    "Ga. State": "Georgia State",
    "Texas A&M-Commerce": "Texas A&M",  # Common confusion but this is FCS
    "TAMU": "Texas A&M",
    "La Tech": "Louisiana Tech",
    "LA Tech": "Louisiana Tech",
    "LA. Tech": "Louisiana Tech",
    "Sam Houston": "Sam Houston State",
    "SHSU": "Sam Houston State",
    "Jax State": "Jacksonville State",
    "Jax St": "Jacksonville State",
    "Jacksonville St": "Jacksonville State",
    "Jacksonville St.": "Jacksonville State",
    "KSU": "Kennesaw State",  # Also could be Kansas State context-dependent
    "Kennesaw St": "Kennesaw State",
    "Kennesaw St.": "Kennesaw State",
    "JMU": "James Madison",
    "James Madison University": "James Madison",
}

# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def get_fbs_teams(season: Optional[int] = None) -> Set[str]:
    """
    Get the set of all FBS teams for a given season.

    Args:
        season: The season year (e.g., 2024). Currently returns the same set
                for all seasons, but this allows for future expansion to handle
                conference realignment changes.

    Returns:
        Set of FBS team names (canonical names)

    Example:
        >>> fbs_teams = get_fbs_teams(2024)
        >>> "Alabama" in fbs_teams
        True
        >>> "Alabama A&M" in fbs_teams  # FCS team
        False
    """
    # Combine all conference teams
    all_fbs = set()
    for teams in CONFERENCE_TEAMS.values():
        all_fbs.update(teams)

    return all_fbs


def is_fbs_team(team_name: str, season: Optional[int] = None) -> bool:
    """
    Check if a team is an FBS team.

    Args:
        team_name: The team name to check
        season: The season year (optional)

    Returns:
        True if the team is FBS, False otherwise
    """
    # Try canonical name first
    canonical = normalize_team_name(team_name)
    return canonical in get_fbs_teams(season)


def is_fbs_game(home_team: str, away_team: str, season: Optional[int] = None) -> bool:
    """
    Check if a game is an FBS vs FBS matchup (both teams must be FBS).

    Args:
        home_team: Home team name
        away_team: Away team name
        season: The season year (optional)

    Returns:
        True if BOTH teams are FBS, False if either is FCS/non-FBS
    """
    return is_fbs_team(home_team, season) and is_fbs_team(away_team, season)


def normalize_team_name(team_name: str) -> str:
    """
    Normalize a team name to its canonical form.

    Args:
        team_name: The team name (may be an alias)

    Returns:
        The canonical team name, or the original if no alias found
    """
    if team_name in TEAM_ALIASES:
        return TEAM_ALIASES[team_name]
    return team_name


def get_team_conference(team_name: str) -> Optional[str]:
    """
    Get the conference for a given team.

    Args:
        team_name: The team name

    Returns:
        Conference name or None if team not found
    """
    canonical = normalize_team_name(team_name)
    return TEAM_TO_CONFERENCE.get(canonical)


def get_conference_tier(conference: str) -> str:
    """
    Get the tier (P4 or G5) for a conference.

    Args:
        conference: Conference name

    Returns:
        "P4", "G5", or "Independent"
    """
    if conference in POWER_4_CONFERENCES:
        return "P4"
    elif conference in GROUP_OF_5_CONFERENCES:
        return "G5"
    elif conference == "Independent":
        return "Independent"
    return "Unknown"


def get_conference_matchup(home_team: str, away_team: str) -> str:
    """
    Get the conference matchup type for a game.

    Args:
        home_team: Home team name
        away_team: Away team name

    Returns:
        Conference matchup string (e.g., "SEC vs SEC", "Big Ten vs SEC", "P4 vs G5")
    """
    home_conf = get_team_conference(home_team)
    away_conf = get_team_conference(away_team)

    if home_conf is None or away_conf is None:
        return "Unknown"

    # Same conference
    if home_conf == away_conf:
        return f"{home_conf} vs {home_conf}"

    # Different conferences - check if same tier
    home_tier = get_conference_tier(home_conf)
    away_tier = get_conference_tier(away_conf)

    if home_tier == away_tier:
        # Same tier but different conferences
        return f"{home_conf} vs {away_conf}"
    else:
        # Different tiers - use tier notation
        return f"{home_tier} vs {away_tier}"


def get_teams_by_conference(conference: str) -> Set[str]:
    """
    Get all teams in a conference.

    Args:
        conference: Conference name (e.g., "SEC", "Big Ten")

    Returns:
        Set of team names in that conference
    """
    return CONFERENCE_TEAMS.get(conference, set())


def get_power_4_teams() -> Set[str]:
    """Get all Power 4 conference teams."""
    p4_teams = set()
    for conf in POWER_4_CONFERENCES:
        p4_teams.update(CONFERENCE_TEAMS[conf])
    return p4_teams


def get_group_of_5_teams() -> Set[str]:
    """Get all Group of 5 conference teams."""
    g5_teams = set()
    for conf in GROUP_OF_5_CONFERENCES:
        g5_teams.update(CONFERENCE_TEAMS[conf])
    return g5_teams


# =============================================================================
# VALIDATION & STATISTICS
# =============================================================================

def validate_fbs_count() -> Dict[str, int]:
    """
    Validate and return FBS team counts by conference.

    Expected total: ~134 FBS teams
    """
    counts = {conf: len(teams) for conf, teams in CONFERENCE_TEAMS.items()}
    counts["Total"] = sum(counts.values())
    return counts


if __name__ == "__main__":
    # Print validation stats when run directly
    print("FBS Teams by Conference:")
    print("-" * 40)
    counts = validate_fbs_count()
    for conf, count in counts.items():
        if conf != "Total":
            print(f"  {conf}: {count} teams")
    print("-" * 40)
    print(f"  TOTAL: {counts['Total']} FBS teams")
    print()

    # Test some lookups
    print("Sample lookups:")
    print(f"  Alabama conference: {get_team_conference('Alabama')}")
    print(f"  App State conference: {get_team_conference('Appalachian State')}")
    print(f"  Is Alabama FBS? {is_fbs_team('Alabama')}")
    print(f"  Is 'Alabama A&M' FBS? {is_fbs_team('Alabama A&M')}")
    print(f"  Georgia vs Alabama matchup: {get_conference_matchup('Georgia', 'Alabama')}")
    print(f"  Ohio State vs USC matchup: {get_conference_matchup('Ohio State', 'USC')}")
    print(f"  Alabama vs Troy matchup: {get_conference_matchup('Alabama', 'Troy')}")
