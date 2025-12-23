"""
Kelly Criterion Money Management for CFB Betting.

Calculates optimal bet sizing using the Kelly Criterion:
- Kelly % = (Decimal Odds * Win Prob - (1 - Win Prob)) / Decimal Odds
- Uses Fractional Kelly (0.25x) for safety

Full Kelly is mathematically optimal but too volatile for real betting.
Quarter Kelly reduces variance while still capturing edge.
"""

import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
KELLY_FRACTION = 0.25  # Quarter Kelly (conservative)
DEFAULT_BANKROLL = 1000  # Default bankroll size
MIN_BET = 10  # Minimum bet size
MAX_BET = 200  # Maximum bet size (cap for safety)
DEFAULT_ODDS = -110  # Standard American odds


def american_to_decimal(american_odds):
    """
    Convert American odds to Decimal odds.

    Examples:
        -110 -> 1.909
        +150 -> 2.500
        -200 -> 1.500
    """
    if american_odds >= 100:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def decimal_to_implied_prob(decimal_odds):
    """
    Convert decimal odds to implied probability.

    Example:
        1.909 -> 52.4% (standard -110 line)
    """
    return 1 / decimal_odds


def margin_to_win_prob(predicted_margin, spread, std_dev=13.5):
    """
    Convert predicted margin and spread to win probability.

    Uses normal distribution to estimate probability of covering spread.

    Args:
        predicted_margin: Model's predicted home margin
        spread: Vegas spread (negative = home favored)
        std_dev: Standard deviation of prediction errors (default 13.5)

    Returns:
        Probability of the bet winning (0-1)
    """
    from scipy import stats

    # Edge is how much better we think team will do vs spread
    # If we bet HOME: edge = predicted_margin - (-spread) = predicted_margin + spread
    # If we bet AWAY: edge = (-spread) - predicted_margin

    edge = predicted_margin - (-spread)  # Positive = bet home, Negative = bet away

    if edge > 0:
        # Betting HOME to cover
        # Need actual_margin > -spread
        # P(actual > -spread) = P(actual - predicted > -spread - predicted)
        # = P(error > -spread - predicted) = P(error > -(edge))
        # = 1 - CDF(-edge) = CDF(edge)
        win_prob = stats.norm.cdf(edge / std_dev)
    else:
        # Betting AWAY to cover
        # Need actual_margin < -spread
        # = P(error < -spread - predicted) = CDF(-edge)
        win_prob = stats.norm.cdf(-edge / std_dev)

    return win_prob


def kelly_criterion(win_prob, decimal_odds):
    """
    Calculate Kelly Criterion bet fraction.

    Formula: Kelly % = (p * b - q) / b
    Where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = net odds received (decimal_odds - 1)

    Args:
        win_prob: Probability of winning (0-1)
        decimal_odds: Decimal odds (e.g., 1.909 for -110)

    Returns:
        Kelly fraction (0-1), or 0 if no edge
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0

    q = 1 - win_prob
    b = decimal_odds - 1  # Net odds

    kelly = (win_prob * b - q) / b

    # Never bet negative Kelly (no edge)
    return max(0, kelly)


def fractional_kelly(win_prob, decimal_odds, fraction=KELLY_FRACTION):
    """
    Calculate Fractional Kelly bet size.

    Fractional Kelly reduces volatility while capturing edge.
    Quarter Kelly (0.25) is common for sports betting.

    Args:
        win_prob: Probability of winning (0-1)
        decimal_odds: Decimal odds
        fraction: Kelly fraction (default 0.25)

    Returns:
        Fractional Kelly percentage (0-1)
    """
    full_kelly = kelly_criterion(win_prob, decimal_odds)
    return full_kelly * fraction


def calculate_bet_size(win_prob, decimal_odds, bankroll=DEFAULT_BANKROLL,
                       fraction=KELLY_FRACTION, min_bet=MIN_BET, max_bet=MAX_BET):
    """
    Calculate actual bet size in dollars.

    Args:
        win_prob: Probability of winning (0-1)
        decimal_odds: Decimal odds
        bankroll: Total bankroll
        fraction: Kelly fraction
        min_bet: Minimum bet size
        max_bet: Maximum bet size

    Returns:
        Bet size in dollars, or 0 if no edge
    """
    kelly_pct = fractional_kelly(win_prob, decimal_odds, fraction)

    if kelly_pct <= 0:
        return 0

    bet_size = bankroll * kelly_pct

    # Apply min/max constraints
    if bet_size < min_bet:
        return 0  # Edge too small to bet

    return min(bet_size, max_bet)


def get_bet_recommendation(predicted_margin, spread, bankroll=DEFAULT_BANKROLL,
                           american_odds=DEFAULT_ODDS, std_dev=13.5):
    """
    Get complete bet recommendation including size.

    Args:
        predicted_margin: Model's predicted home margin
        spread: Vegas spread
        bankroll: Total bankroll
        american_odds: American odds (default -110)
        std_dev: Model standard deviation

    Returns:
        dict with bet_side, win_prob, kelly_pct, bet_size, edge
    """
    decimal_odds = american_to_decimal(american_odds)
    win_prob = margin_to_win_prob(predicted_margin, spread, std_dev)

    # Determine bet side
    edge = predicted_margin - (-spread)
    if edge > 0:
        bet_side = "HOME"
    else:
        bet_side = "AWAY"
        # Recalculate win prob for away bet
        win_prob = margin_to_win_prob(predicted_margin, spread, std_dev)

    kelly_pct = fractional_kelly(win_prob, decimal_odds)
    bet_size = calculate_bet_size(win_prob, decimal_odds, bankroll)

    # Calculate implied edge
    implied_prob = decimal_to_implied_prob(decimal_odds)
    edge_pct = (win_prob - implied_prob) * 100

    return {
        'bet_side': bet_side,
        'win_prob': win_prob,
        'implied_prob': implied_prob,
        'edge_pct': edge_pct,
        'kelly_pct': kelly_pct * 100,  # As percentage
        'bet_size': bet_size,
        'decimal_odds': decimal_odds
    }


# ============================================================
# TOTALS (OVER/UNDER) KELLY FUNCTIONS
# ============================================================

def totals_margin_to_win_prob(predicted_total, vegas_total, std_dev=12.5):
    """
    Convert predicted total and Vegas O/U to win probability.

    Args:
        predicted_total: Model's predicted total points
        vegas_total: Vegas over/under line
        std_dev: Standard deviation of prediction errors

    Returns:
        Tuple (over_prob, under_prob)
    """
    from scipy import stats

    edge = predicted_total - vegas_total

    # P(actual > vegas) = P(error > vegas - predicted) = 1 - CDF(vegas - predicted)
    over_prob = 1 - stats.norm.cdf((vegas_total - predicted_total) / std_dev)
    under_prob = 1 - over_prob

    return over_prob, under_prob


def get_totals_bet_recommendation(predicted_total, vegas_total, bankroll=DEFAULT_BANKROLL,
                                   american_odds=DEFAULT_ODDS, std_dev=12.5):
    """
    Get complete O/U bet recommendation including size.

    Args:
        predicted_total: Model's predicted total
        vegas_total: Vegas O/U line
        bankroll: Total bankroll
        american_odds: American odds (default -110)
        std_dev: Model standard deviation

    Returns:
        dict with bet_side, win_prob, kelly_pct, bet_size, edge
    """
    decimal_odds = american_to_decimal(american_odds)
    over_prob, under_prob = totals_margin_to_win_prob(predicted_total, vegas_total, std_dev)

    edge = predicted_total - vegas_total

    if edge > 0:
        bet_side = "OVER"
        win_prob = over_prob
    else:
        bet_side = "UNDER"
        win_prob = under_prob

    kelly_pct = fractional_kelly(win_prob, decimal_odds)
    bet_size = calculate_bet_size(win_prob, decimal_odds, bankroll)

    # Calculate implied edge
    implied_prob = decimal_to_implied_prob(decimal_odds)
    edge_pct = (win_prob - implied_prob) * 100

    return {
        'bet_side': bet_side,
        'win_prob': win_prob,
        'implied_prob': implied_prob,
        'edge_pct': edge_pct,
        'kelly_pct': kelly_pct * 100,
        'bet_size': bet_size,
        'decimal_odds': decimal_odds
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def format_bet_size(bet_size):
    """Format bet size for display."""
    if bet_size == 0:
        return "No Bet"
    elif bet_size < 25:
        return f"${bet_size:.0f} (Small)"
    elif bet_size < 75:
        return f"${bet_size:.0f} (Medium)"
    elif bet_size < 150:
        return f"${bet_size:.0f} (Large)"
    else:
        return f"${bet_size:.0f} (MAX)"


def get_confidence_tier(win_prob):
    """Get confidence tier based on win probability."""
    if win_prob >= 0.60:
        return "HIGH"
    elif win_prob >= 0.55:
        return "MEDIUM"
    elif win_prob >= 0.52:
        return "LOW"
    else:
        return "PASS"


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("KELLY CRITERION MONEY MANAGER TEST")
    print("=" * 60)

    # Test cases
    test_cases = [
        {"name": "Small Edge", "pred_margin": 5, "spread": -3},
        {"name": "Medium Edge", "pred_margin": 10, "spread": -3},
        {"name": "Large Edge", "pred_margin": 15, "spread": -3},
        {"name": "Huge Edge", "pred_margin": 20, "spread": -3},
        {"name": "No Edge", "pred_margin": 3, "spread": -3},
        {"name": "Negative Edge", "pred_margin": -5, "spread": -3},
    ]

    print(f"\nBankroll: $1,000 | Kelly Fraction: {KELLY_FRACTION}")
    print(f"Odds: -110 (Decimal: {american_to_decimal(-110):.3f})")
    print(f"Implied Prob: {decimal_to_implied_prob(american_to_decimal(-110))*100:.1f}%")

    print(f"\n{'Case':<15} {'Edge':>8} {'Win%':>8} {'Kelly%':>8} {'Bet':>12}")
    print("-" * 55)

    for case in test_cases:
        rec = get_bet_recommendation(case['pred_margin'], case['spread'])
        edge = case['pred_margin'] - (-case['spread'])
        print(f"{case['name']:<15} {edge:>+8.1f} {rec['win_prob']*100:>7.1f}% "
              f"{rec['kelly_pct']:>7.2f}% {format_bet_size(rec['bet_size']):>12}")

    print("\n" + "=" * 60)
    print("TOTALS TEST")
    print("=" * 60)

    totals_cases = [
        {"name": "OVER 3pt edge", "pred": 53, "vegas": 50},
        {"name": "OVER 5pt edge", "pred": 55, "vegas": 50},
        {"name": "UNDER 4pt edge", "pred": 46, "vegas": 50},
        {"name": "No edge", "pred": 50, "vegas": 50},
    ]

    print(f"\n{'Case':<18} {'Pred':>6} {'Vegas':>6} {'Side':>6} {'Win%':>7} {'Bet':>12}")
    print("-" * 60)

    for case in totals_cases:
        rec = get_totals_bet_recommendation(case['pred'], case['vegas'])
        print(f"{case['name']:<18} {case['pred']:>6.0f} {case['vegas']:>6.0f} "
              f"{rec['bet_side']:>6} {rec['win_prob']*100:>6.1f}% {format_bet_size(rec['bet_size']):>12}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
