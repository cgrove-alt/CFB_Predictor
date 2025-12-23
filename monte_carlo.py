"""
Monte Carlo Simulation for CFB Spread Betting.

Simulates game outcomes to calculate cover probability.
"""

import numpy as np


def simulate_game(predicted_margin, spread_line, std_dev=14.0, simulations=10000):
    """
    Simulate a game to calculate cover probability.

    Args:
        predicted_margin: Model's predicted margin (positive = home team wins by X)
        spread_line: Vegas spread line (negative = home favored, positive = home underdog)
        std_dev: Standard deviation for score distribution (default 14.0 for CFB)
        simulations: Number of Monte Carlo simulations (default 10000)

    Returns:
        dict with:
            - cover_probability: Probability home team covers the spread
            - simulated_margins: Array of simulated margins
            - mean_margin: Mean of simulated margins
            - std_margin: Std dev of simulated margins

    Example:
        If model predicts home wins by 7 (margin=7) and spread is -3 (home favored by 3),
        home covers if they win by MORE than 3. So we check if margin > 3.

        If spread is +3 (home is underdog), home covers if they lose by LESS than 3
        or win outright. So we check if margin > -3.

    The spread from home perspective:
        - Home favored by 3 = spread of -3
        - Home covers if actual_margin > abs(spread_line) when spread < 0
        - Home underdog by 3 = spread of +3
        - Home covers if actual_margin > -spread_line
    """
    # Generate simulated margins
    simulated_margins = np.random.normal(predicted_margin, std_dev, simulations)

    # Calculate cover probability
    # Home covers if their margin beats the spread
    # If spread is -7 (home favored), home needs margin > 7 to cover
    # If spread is +7 (home underdog), home needs margin > -7 to cover
    covers = simulated_margins > (-spread_line)
    cover_probability = covers.mean()

    return {
        'cover_probability': cover_probability,
        'simulated_margins': simulated_margins,
        'mean_margin': simulated_margins.mean(),
        'std_margin': simulated_margins.std()
    }


def get_bet_signal(cover_prob, buy_threshold=0.55, fade_threshold=0.45):
    """
    Get betting signal based on cover probability.

    Args:
        cover_prob: Probability of covering the spread
        buy_threshold: Probability above which to BUY (default 55%)
        fade_threshold: Probability below which to FADE (default 45%)

    Returns:
        tuple: (signal_text, signal_emoji)
    """
    if cover_prob >= buy_threshold:
        return "BUY", "BUY"
    elif cover_prob <= fade_threshold:
        return "FADE", "FADE"
    else:
        return "PASS", "PASS"


def format_win_prob(cover_prob):
    """Format cover probability as percentage string."""
    return f"{cover_prob * 100:.0f}%"


# Test the function
if __name__ == "__main__":
    print("=" * 60)
    print("MONTE CARLO SIMULATION TEST")
    print("=" * 60)

    # Test case 1: Home favored, model agrees
    print("\nTest 1: Home favored by 7, model predicts home wins by 10")
    result = simulate_game(predicted_margin=10, spread_line=-7)
    print(f"  Cover probability: {result['cover_probability']:.1%}")
    print(f"  Mean margin: {result['mean_margin']:.1f}")
    print(f"  Signal: {get_bet_signal(result['cover_probability'])}")

    # Test case 2: Home underdog, model agrees
    print("\nTest 2: Home underdog by 7, model predicts home loses by 3")
    result = simulate_game(predicted_margin=-3, spread_line=7)
    print(f"  Cover probability: {result['cover_probability']:.1%}")
    print(f"  Mean margin: {result['mean_margin']:.1f}")
    print(f"  Signal: {get_bet_signal(result['cover_probability'])}")

    # Test case 3: Close game
    print("\nTest 3: Pick'em (spread 0), model predicts home by 2")
    result = simulate_game(predicted_margin=2, spread_line=0)
    print(f"  Cover probability: {result['cover_probability']:.1%}")
    print(f"  Mean margin: {result['mean_margin']:.1f}")
    print(f"  Signal: {get_bet_signal(result['cover_probability'])}")

    # Test case 4: Model disagrees with line
    print("\nTest 4: Home favored by 14, model predicts home by 3 (value on away)")
    result = simulate_game(predicted_margin=3, spread_line=-14)
    print(f"  Cover probability: {result['cover_probability']:.1%}")
    print(f"  Mean margin: {result['mean_margin']:.1f}")
    print(f"  Signal: {get_bet_signal(result['cover_probability'])}")

    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION TEST COMPLETE")
    print("=" * 60)
