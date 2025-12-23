"""
Monte Carlo Simulation Engine for CFB Betting.

Runs 10,000 simulations per game based on:
- Home Offense vs Away Defense matchups
- Pass and Rush success rates
- PPA (Predicted Points Added) metrics

Outputs:
- Cover Probability
- Over/Under Probability
- Simulated Score
- Edge vs Vegas
"""

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
N_SIMULATIONS = 10000
IMPLIED_PROBABILITY_110 = 110 / (110 + 100)  # 52.38% at -110 odds

# Average CFB scoring (for baseline)
AVG_TEAM_SCORE = 28.0
SCORE_STD_DEV = 10.0  # Standard deviation for score variation

# PPA multipliers (how much PPA affects expected points)
PPA_MULTIPLIER = 7.0  # Each point of PPA is worth ~7 points in expected score

# Success rate impact (how much success rate above/below 0.4 affects scoring)
SUCCESS_RATE_BASELINE = 0.40
SUCCESS_RATE_MULTIPLIER = 20.0  # 10% better success rate = +2 points


# ============================================================
# MATCHUP ANALYSIS
# ============================================================
def calculate_expected_points(offense_rush_sr, offense_pass_sr, offense_ppa,
                               defense_rush_sr, defense_pass_sr, defense_ppa,
                               base_score=AVG_TEAM_SCORE):
    """
    Calculate expected points based on matchup stats.

    Higher offensive success rates = more points
    Lower defensive success rates allowed = opponent scores less
    PPA directly impacts scoring expectation
    """
    expected = base_score

    # Offensive adjustments (good offense = more points)
    if offense_rush_sr is not None and not np.isnan(offense_rush_sr):
        rush_adj = (offense_rush_sr - SUCCESS_RATE_BASELINE) * SUCCESS_RATE_MULTIPLIER
        expected += rush_adj

    if offense_pass_sr is not None and not np.isnan(offense_pass_sr):
        pass_adj = (offense_pass_sr - SUCCESS_RATE_BASELINE) * SUCCESS_RATE_MULTIPLIER
        expected += pass_adj

    if offense_ppa is not None and not np.isnan(offense_ppa):
        expected += offense_ppa * PPA_MULTIPLIER

    # Defensive adjustments from opponent (bad defense faced = more points)
    if defense_rush_sr is not None and not np.isnan(defense_rush_sr):
        # Higher opponent defense success rate = they stop you = fewer points
        def_rush_adj = (SUCCESS_RATE_BASELINE - defense_rush_sr) * SUCCESS_RATE_MULTIPLIER
        expected += def_rush_adj

    if defense_pass_sr is not None and not np.isnan(defense_pass_sr):
        def_pass_adj = (SUCCESS_RATE_BASELINE - defense_pass_sr) * SUCCESS_RATE_MULTIPLIER
        expected += def_pass_adj

    if defense_ppa is not None and not np.isnan(defense_ppa):
        # Negative opponent defense PPA = they're good at stopping you
        expected -= defense_ppa * PPA_MULTIPLIER

    # Clamp to reasonable range
    return max(7, min(60, expected))


def calculate_score_std(elo_diff=0):
    """
    Calculate standard deviation for score simulation.
    Closer games have higher variance.
    """
    base_std = SCORE_STD_DEV
    # Adjust based on expected closeness
    if abs(elo_diff) < 100:
        return base_std * 1.2  # More variance in close matchups
    elif abs(elo_diff) > 300:
        return base_std * 0.8  # Less variance in mismatches
    return base_std


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================
def simulate_game(home_expected, away_expected, home_std, away_std,
                  vegas_spread, vegas_total=None, n_sims=N_SIMULATIONS):
    """
    Run Monte Carlo simulation for a single game.

    Returns:
        dict with simulation results
    """
    # Generate random scores
    np.random.seed(42)  # For reproducibility in same session
    home_scores = np.random.normal(home_expected, home_std, n_sims)
    away_scores = np.random.normal(away_expected, away_std, n_sims)

    # Clamp to realistic scores (min 0, round to integers)
    home_scores = np.maximum(0, np.round(home_scores)).astype(int)
    away_scores = np.maximum(0, np.round(away_scores)).astype(int)

    # Calculate margins
    margins = home_scores - away_scores

    # Cover analysis (home team covering spread)
    # Vegas spread is from home perspective (negative = home favored)
    # Home covers if: margin > -spread (e.g., margin > 7 when spread is -7)
    covers = margins > -vegas_spread
    cover_probability = covers.mean()

    # Win probability (straight up)
    home_wins = margins > 0
    win_probability = home_wins.mean()

    # Over/Under analysis
    totals = home_scores + away_scores
    if vegas_total is not None:
        overs = totals > vegas_total
        over_probability = overs.mean()
    else:
        over_probability = None

    # Average simulated scores
    avg_home_score = home_scores.mean()
    avg_away_score = away_scores.mean()

    # Edge calculation
    edge = (cover_probability - IMPLIED_PROBABILITY_110) * 100  # In percentage points

    return {
        'home_expected': home_expected,
        'away_expected': away_expected,
        'simulated_home_score': round(avg_home_score),
        'simulated_away_score': round(avg_away_score),
        'simulated_total': round(avg_home_score + avg_away_score),
        'cover_probability': cover_probability,
        'win_probability': win_probability,
        'over_probability': over_probability,
        'implied_probability': IMPLIED_PROBABILITY_110,
        'edge': edge,
        'n_simulations': n_sims
    }


def run_simulation(game_data):
    """
    Run Monte Carlo simulation for a game using available stats.

    game_data: dict or Series with game info and stats
    """
    # Extract matchup stats
    # Home offense vs Away defense
    home_off_rush = game_data.get('home_off_rush_success')
    home_off_pass = game_data.get('home_off_pass_success')
    home_off_ppa = game_data.get('home_off_std_downs_ppa')

    away_def_rush = game_data.get('away_def_rush_success')
    away_def_pass = game_data.get('away_def_pass_success')
    away_def_ppa = game_data.get('away_def_pass_downs_ppa')

    # Away offense vs Home defense
    away_off_rush = game_data.get('away_off_rush_success')
    away_off_pass = game_data.get('away_off_pass_success')
    away_off_ppa = game_data.get('away_off_std_downs_ppa')

    home_def_rush = game_data.get('home_def_rush_success')
    home_def_pass = game_data.get('home_def_pass_success')
    home_def_ppa = game_data.get('home_def_pass_downs_ppa')

    # Elo for variance adjustment
    home_elo = game_data.get('home_pregame_elo', 1500)
    away_elo = game_data.get('away_pregame_elo', 1500)
    elo_diff = (home_elo or 1500) - (away_elo or 1500)

    # Calculate expected points for each team
    # Home team: their offense vs opponent (away) defense
    home_expected = calculate_expected_points(
        home_off_rush, home_off_pass, home_off_ppa,
        away_def_rush, away_def_pass, away_def_ppa
    )

    # Away team: their offense vs opponent (home) defense
    away_expected = calculate_expected_points(
        away_off_rush, away_off_pass, away_off_ppa,
        home_def_rush, home_def_pass, home_def_ppa
    )

    # Add home field advantage (~3 points)
    home_expected += 3.0

    # Elo adjustment
    elo_adjustment = elo_diff / 100  # ~1 point per 100 Elo
    home_expected += elo_adjustment / 2
    away_expected -= elo_adjustment / 2

    # Calculate std devs
    home_std = calculate_score_std(elo_diff)
    away_std = calculate_score_std(elo_diff)

    # Get Vegas lines
    vegas_spread = game_data.get('spread_line', 0) or 0
    vegas_total = game_data.get('over_under')

    # Run simulation
    results = simulate_game(
        home_expected, away_expected,
        home_std, away_std,
        vegas_spread, vegas_total
    )

    return results


# ============================================================
# BATCH SIMULATION
# ============================================================
def simulate_games_batch(games_df):
    """
    Run Monte Carlo simulations for a batch of games.

    Returns DataFrame with simulation results.
    """
    results = []

    for idx, game in games_df.iterrows():
        game_dict = game.to_dict()
        sim_result = run_simulation(game_dict)

        sim_result['game_id'] = game.get('id')
        sim_result['home_team'] = game.get('home_team')
        sim_result['away_team'] = game.get('away_team')

        results.append(sim_result)

    return pd.DataFrame(results)


# ============================================================
# DISPLAY HELPERS
# ============================================================
def format_probability(prob):
    """Format probability as percentage."""
    if prob is None:
        return "—"
    return f"{prob * 100:.1f}%"


def format_simulated_score(home_score, away_score):
    """Format simulated score string."""
    return f"{int(home_score)} - {int(away_score)}"


def get_bet_recommendation(cover_prob, edge, vegas_spread):
    """
    Get bet recommendation based on simulation.
    """
    if edge > 5:
        if vegas_spread < 0:
            return f"HOME -{abs(vegas_spread):.1f}"
        else:
            return f"HOME +{vegas_spread:.1f}"
    elif edge < -5:
        if vegas_spread < 0:
            return f"AWAY +{abs(vegas_spread):.1f}"
        else:
            return f"AWAY -{abs(vegas_spread):.1f}"
    else:
        return "NO BET"


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MONTE CARLO ENGINE TEST")
    print("=" * 60)

    # Test game data
    test_game = {
        'home_team': 'Ohio State',
        'away_team': 'Michigan',
        'home_pregame_elo': 1650,
        'away_pregame_elo': 1620,
        'home_off_rush_success': 0.48,
        'home_off_pass_success': 0.45,
        'home_off_std_downs_ppa': 0.35,
        'away_def_rush_success': 0.38,
        'away_def_pass_success': 0.42,
        'away_def_pass_downs_ppa': 0.15,
        'away_off_rush_success': 0.46,
        'away_off_pass_success': 0.43,
        'away_off_std_downs_ppa': 0.30,
        'home_def_rush_success': 0.35,
        'home_def_pass_success': 0.40,
        'home_def_pass_downs_ppa': 0.10,
        'spread_line': -3.5,
        'over_under': 48.5
    }

    print(f"\nTest Game: {test_game['away_team']} @ {test_game['home_team']}")
    print(f"Vegas Spread: {test_game['spread_line']}")
    print(f"O/U: {test_game['over_under']}")

    print("\nRunning 10,000 simulations...")
    result = run_simulation(test_game)

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Home Expected Points: {result['home_expected']:.1f}")
    print(f"Away Expected Points: {result['away_expected']:.1f}")
    print(f"\nSimulated Score: {result['simulated_home_score']} - {result['simulated_away_score']}")
    print(f"Simulated Total: {result['simulated_total']}")
    print(f"\nWin Probability: {format_probability(result['win_probability'])}")
    print(f"Cover Probability: {format_probability(result['cover_probability'])}")
    print(f"Over Probability: {format_probability(result['over_probability'])}")
    print(f"\nImplied Probability: {format_probability(result['implied_probability'])}")
    print(f"EDGE: {result['edge']:+.1f}%")

    recommendation = get_bet_recommendation(
        result['cover_probability'],
        result['edge'],
        test_game['spread_line']
    )
    print(f"\nRecommendation: {recommendation}")

    print("\n" + "=" * 60)
    print("ENGINE READY")
    print("=" * 60)
