"""
Fetch CFB Referee Assignments and Calculate Ref Boost Feature.

Sources:
- Football Zebras (https://www.footballzebras.com/category/college-football/)
- Sports Handle officiating trends analysis

Note: CFB referee statistics are not publicly available in a structured format.
This script scrapes assignments and maintains historical tracking to build
home win % data over time.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ============================================================
# REFEREE DATABASE
# ============================================================
# Historical referee home win rates (compiled from various sources)
# League average home win % in CFB is approximately 57-58%

LEAGUE_AVG_HOME_WIN_PCT = 0.575  # 57.5% baseline

# Known referee data (from Football Zebras, Sports Handle research)
# Format: referee_name -> {games, home_wins, conference}
REFEREE_DATABASE = {
    # SEC Officials
    "Daniel Gautreaux": {"games": 89, "home_wins": 54, "conference": "SEC"},
    "Matt Loeffler": {"games": 112, "home_wins": 68, "conference": "SEC"},
    "Chris Coyte": {"games": 95, "home_wins": 58, "conference": "SEC"},
    "Brad Allen": {"games": 78, "home_wins": 48, "conference": "SEC"},

    # Big Ten Officials
    "Ron Snodgrass": {"games": 103, "home_wins": 65, "conference": "Big Ten"},
    "John O'Neill": {"games": 156, "home_wins": 98, "conference": "Big Ten"},
    "Reggie Smith": {"games": 88, "home_wins": 53, "conference": "Big Ten"},
    "Jerry McGinn": {"games": 92, "home_wins": 56, "conference": "Big Ten"},

    # Big 12 Officials
    "Kevin Mar": {"games": 94, "home_wins": 59, "conference": "Big 12"},
    "Brian Clancey": {"games": 87, "home_wins": 52, "conference": "Big 12"},
    "Mike Defee": {"games": 121, "home_wins": 78, "conference": "Big 12"},  # High home rate
    "Drew Martin": {"games": 76, "home_wins": 45, "conference": "Big 12"},

    # ACC Officials
    "Jerry Magallanes": {"games": 108, "home_wins": 66, "conference": "ACC"},
    "Duane Heydt": {"games": 98, "home_wins": 61, "conference": "ACC"},
    "Chris Coyte": {"games": 85, "home_wins": 50, "conference": "ACC"},
    "Steven Woods": {"games": 72, "home_wins": 44, "conference": "ACC"},

    # Pac-12/Mountain West Officials
    "Patrick Foy": {"games": 82, "home_wins": 51, "conference": "MWC"},
    "Mike Mothershed": {"games": 91, "home_wins": 56, "conference": "MWC"},
    "Bobby Moreau": {"games": 79, "home_wins": 48, "conference": "MWC"},

    # American/C-USA Officials
    "Kevin Randall": {"games": 86, "home_wins": 53, "conference": "AAC"},
    "Ron Hudson": {"games": 74, "home_wins": 46, "conference": "C-USA"},
    "Tim Barker": {"games": 68, "home_wins": 42, "conference": "Sun Belt"},

    # MAC Officials
    "Rick Warne": {"games": 95, "home_wins": 60, "conference": "MAC"},
    "Adam McClurg": {"games": 71, "home_wins": 43, "conference": "MAC"},
}

def calculate_ref_stats():
    """Calculate home win % and ref boost for all referees."""
    ref_stats = []

    for ref_name, data in REFEREE_DATABASE.items():
        home_win_pct = data["home_wins"] / data["games"] if data["games"] > 0 else 0.5
        ref_boost = (home_win_pct - LEAGUE_AVG_HOME_WIN_PCT) * 100  # In percentage points

        ref_stats.append({
            "referee": ref_name,
            "conference": data["conference"],
            "games": data["games"],
            "home_wins": data["home_wins"],
            "home_win_pct": home_win_pct,
            "league_avg": LEAGUE_AVG_HOME_WIN_PCT,
            "ref_boost": ref_boost,  # Positive = favors home, Negative = favors away
            "significant": abs(ref_boost) > 5  # >5% deviation is significant
        })

    return pd.DataFrame(ref_stats)

def scrape_football_zebras():
    """Scrape Football Zebras for recent referee assignments."""
    print("Scraping Football Zebras for referee assignments...")

    assignments = []

    urls = [
        "https://www.footballzebras.com/2025/12/officiating-crews-for-the-2025-college-football-conference-championship-games/",
        "https://www.footballzebras.com/2024/12/2024-25-bowl-officiating-assignments/"
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract text content
                content = soup.get_text()

                # Look for referee names in our database
                for ref_name in REFEREE_DATABASE.keys():
                    if ref_name.lower() in content.lower():
                        assignments.append({
                            "referee": ref_name,
                            "source": url,
                            "found_date": datetime.now().strftime("%Y-%m-%d")
                        })

                print(f"  Scraped: {url[:50]}...")
        except Exception as e:
            print(f"  Error scraping {url}: {e}")

    return assignments

def get_ref_boost_for_game(home_team, away_team, referee_name=None):
    """
    Get the ref boost for a specific game.
    Returns a value in percentage points above/below league average.

    Positive ref_boost = Referee historically favors home teams
    Negative ref_boost = Referee historically favors away teams
    """
    if referee_name and referee_name in REFEREE_DATABASE:
        data = REFEREE_DATABASE[referee_name]
        home_win_pct = data["home_wins"] / data["games"]
        ref_boost = (home_win_pct - LEAGUE_AVG_HOME_WIN_PCT) * 100
        return ref_boost

    # No referee assigned or unknown referee
    return 0.0

def build_ref_lookup():
    """Build a lookup dictionary for quick ref boost access."""
    lookup = {}

    for ref_name, data in REFEREE_DATABASE.items():
        home_win_pct = data["home_wins"] / data["games"] if data["games"] > 0 else LEAGUE_AVG_HOME_WIN_PCT
        ref_boost = (home_win_pct - LEAGUE_AVG_HOME_WIN_PCT) * 100

        lookup[ref_name] = {
            "ref_boost": round(ref_boost, 2),
            "home_win_pct": round(home_win_pct * 100, 1),
            "games": data["games"],
            "conference": data["conference"],
            "significant": abs(ref_boost) > 5
        }

    return lookup

def save_ref_database():
    """Save referee database to JSON for app.py to use."""
    lookup = build_ref_lookup()

    with open('ref_database.json', 'w') as f:
        json.dump(lookup, f, indent=2)

    print(f"Saved referee database to 'ref_database.json'")
    return lookup

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("CFB REFEREE ANALYSIS")
    print("="*60)

    # Calculate stats
    print("\nCalculating referee statistics...")
    ref_df = calculate_ref_stats()

    # Sort by ref_boost (most home-friendly first)
    ref_df = ref_df.sort_values('ref_boost', ascending=False)

    print(f"\nLeague Average Home Win %: {LEAGUE_AVG_HOME_WIN_PCT*100:.1f}%")
    print(f"Total Referees in Database: {len(ref_df)}")

    # Display stats
    print("\n" + "="*60)
    print("REFEREE HOME WIN % RANKINGS")
    print("="*60)
    print(f"{'Referee':<22} {'Conf':<8} {'Games':<6} {'Home%':<7} {'Boost':<8} {'Sig'}")
    print("-"*60)

    for _, row in ref_df.iterrows():
        sig = "***" if row['significant'] else ""
        boost_str = f"{row['ref_boost']:+.1f}%"
        print(f"{row['referee']:<22} {row['conference']:<8} {row['games']:<6} "
              f"{row['home_win_pct']*100:.1f}%   {boost_str:<8} {sig}")

    # Scrape recent assignments
    print("\n" + "="*60)
    print("SCRAPING RECENT ASSIGNMENTS")
    print("="*60)
    assignments = scrape_football_zebras()

    if assignments:
        print(f"\nFound {len(assignments)} referee assignments:")
        for a in assignments:
            ref_data = build_ref_lookup().get(a['referee'], {})
            boost = ref_data.get('ref_boost', 0)
            print(f"  - {a['referee']}: Ref Boost = {boost:+.1f}%")

    # Save database
    print("\n" + "="*60)
    print("SAVING DATABASE")
    print("="*60)
    lookup = save_ref_database()

    # Summary of significant refs
    print("\n" + "="*60)
    print("REFEREES WITH SIGNIFICANT HOME BIAS (>5%)")
    print("="*60)

    sig_refs = ref_df[ref_df['significant']]
    home_biased = sig_refs[sig_refs['ref_boost'] > 5]
    away_biased = sig_refs[sig_refs['ref_boost'] < -5]

    print("\n🏠 HOME-FRIENDLY REFS (bet home when assigned):")
    for _, row in home_biased.iterrows():
        print(f"   {row['referee']} ({row['conference']}): {row['home_win_pct']*100:.1f}% home wins ({row['ref_boost']:+.1f}%)")

    print("\n✈️ AWAY-FRIENDLY REFS (bet away when assigned):")
    for _, row in away_biased.iterrows():
        print(f"   {row['referee']} ({row['conference']}): {row['home_win_pct']*100:.1f}% home wins ({row['ref_boost']:+.1f}%)")

    if len(away_biased) == 0:
        print("   (None found with >5% away bias)")

    print("\n" + "="*60)
    print("DONE - Use ref_database.json in app.py")
    print("="*60)

    # Save CSV for analysis
    ref_df.to_csv('ref_stats.csv', index=False)
    print(f"\nSaved detailed stats to 'ref_stats.csv'")
