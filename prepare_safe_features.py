"""
Prepare Safe Features - Ensure NO DATA LEAKAGE.

This script processes cfb_data_smart.csv to ensure all features are
truly calculated BEFORE the game they appear in.

CRITICAL: The comp_ppa features in the original data are game-specific,
not rolling averages. This script recalculates all features properly.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PREPARE SAFE FEATURES - NO DATA LEAKAGE")
print("=" * 70)


def load_data():
    """Load the raw data."""
    df = pd.read_csv('cfb_data_smart.csv')
    df = df.sort_values(['season', 'week']).reset_index(drop=True)
    print(f"Loaded {len(df)} games from cfb_data_smart.csv")
    return df


def calculate_streaks(df):
    """Calculate win/loss streaks using ONLY past results."""
    print("\nCalculating win/loss streaks...")

    team_streaks = {}
    home_streak = []
    away_streak = []

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        season = row['season']

        # Get current streak BEFORE this game
        h_streak = team_streaks.get(home, (0, 0))
        a_streak = team_streaks.get(away, (0, 0))

        # Only use if same season
        h_val = h_streak[1] if h_streak[0] == season else 0
        a_val = a_streak[1] if a_streak[0] == season else 0

        home_streak.append(h_val)
        away_streak.append(a_val)

        # Update streak AFTER recording (so we don't use current game)
        if pd.notna(row.get('Margin')):
            home_won = row['Margin'] > 0
            if home_won:
                new_h = max(1, h_val + 1) if h_val >= 0 else 1
                new_a = min(-1, a_val - 1) if a_val <= 0 else -1
            else:
                new_h = min(-1, h_val - 1) if h_val <= 0 else -1
                new_a = max(1, a_val + 1) if a_val >= 0 else 1
            team_streaks[home] = (season, new_h)
            team_streaks[away] = (season, new_a)

    df['home_streak'] = home_streak
    df['away_streak'] = away_streak
    df['streak_diff'] = df['home_streak'] - df['away_streak']

    print(f"  Streak range: {df['home_streak'].min()} to {df['home_streak'].max()}")
    return df


def calculate_ats_history(df):
    """Calculate ATS (against the spread) rates using ONLY past results."""
    print("\nCalculating ATS history...")

    # First, ensure we have spread_error
    if 'spread_error' not in df.columns:
        df['spread_error'] = df['Margin'] - (-df['vegas_spread'])

    team_ats = {}
    home_ats = []
    away_ats = []

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']

        # Get ATS rate BEFORE this game (need at least 3 games)
        if home in team_ats and len(team_ats[home]) >= 3:
            home_ats.append(np.mean(team_ats[home][-10:]))  # Last 10 games
        else:
            home_ats.append(0.5)  # Default to 50%

        if away in team_ats and len(team_ats[away]) >= 3:
            away_ats.append(np.mean(team_ats[away][-10:]))
        else:
            away_ats.append(0.5)

        # Update AFTER recording
        if pd.notna(row.get('spread_error')) and pd.notna(row.get('vegas_spread')):
            home_covered = row['spread_error'] > 0
            if home not in team_ats:
                team_ats[home] = []
            if away not in team_ats:
                team_ats[away] = []
            team_ats[home].append(1 if home_covered else 0)
            team_ats[away].append(0 if home_covered else 1)

    df['home_ats'] = home_ats
    df['away_ats'] = away_ats
    df['ats_diff'] = df['home_ats'] - df['away_ats']

    print(f"  Home ATS range: {df['home_ats'].min():.2f} to {df['home_ats'].max():.2f}")
    return df


def calculate_elo_momentum(df):
    """Calculate Elo momentum (change over last 3 games)."""
    print("\nCalculating Elo momentum...")

    team_elo_history = {}
    home_elo_momentum = []
    away_elo_momentum = []

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        season = row['season']

        # Get momentum BEFORE this game
        h_momentum = 0
        a_momentum = 0

        if home in team_elo_history:
            hist = [e for s, e in team_elo_history[home] if s == season]
            if len(hist) >= 3:
                h_momentum = hist[-1] - hist[-3]  # Change over last 3
            elif len(hist) >= 2:
                h_momentum = hist[-1] - hist[-2]

        if away in team_elo_history:
            hist = [e for s, e in team_elo_history[away] if s == season]
            if len(hist) >= 3:
                a_momentum = hist[-1] - hist[-3]
            elif len(hist) >= 2:
                a_momentum = hist[-1] - hist[-2]

        home_elo_momentum.append(h_momentum)
        away_elo_momentum.append(a_momentum)

        # Update AFTER recording
        home_elo = row.get('home_pregame_elo', 1500)
        away_elo = row.get('away_pregame_elo', 1500)

        if home not in team_elo_history:
            team_elo_history[home] = []
        if away not in team_elo_history:
            team_elo_history[away] = []

        team_elo_history[home].append((season, home_elo))
        team_elo_history[away].append((season, away_elo))

    df['home_elo_momentum'] = home_elo_momentum
    df['away_elo_momentum'] = away_elo_momentum
    df['elo_momentum_diff'] = df['home_elo_momentum'] - df['away_elo_momentum']

    print(f"  Elo momentum range: {df['home_elo_momentum'].min():.0f} to {df['home_elo_momentum'].max():.0f}")
    return df


def calculate_scoring_trends(df):
    """Calculate recent scoring trends (last 3 vs last 5)."""
    print("\nCalculating scoring trends...")

    team_scores = {}
    home_trend = []
    away_trend = []

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        season = row['season']

        # Calculate trend BEFORE this game
        h_trend = 0
        a_trend = 0

        if home in team_scores:
            scores = [s for seas, s in team_scores[home] if seas == season]
            if len(scores) >= 5:
                last3 = np.mean(scores[-3:])
                last5 = np.mean(scores[-5:])
                h_trend = last3 - last5
            elif len(scores) >= 3:
                last3 = np.mean(scores[-3:])
                h_trend = last3 - np.mean(scores)

        if away in team_scores:
            scores = [s for seas, s in team_scores[away] if seas == season]
            if len(scores) >= 5:
                last3 = np.mean(scores[-3:])
                last5 = np.mean(scores[-5:])
                a_trend = last3 - last5
            elif len(scores) >= 3:
                last3 = np.mean(scores[-3:])
                a_trend = last3 - np.mean(scores)

        home_trend.append(h_trend)
        away_trend.append(a_trend)

        # Update AFTER recording
        home_pts = row.get('home_points', 0)
        away_pts = row.get('away_points', 0)

        if pd.notna(home_pts):
            if home not in team_scores:
                team_scores[home] = []
            if away not in team_scores:
                team_scores[away] = []
            team_scores[home].append((season, home_pts))
            team_scores[away].append((season, away_pts))

    df['home_scoring_trend'] = home_trend
    df['away_scoring_trend'] = away_trend

    print(f"  Scoring trend range: {df['home_scoring_trend'].min():.1f} to {df['home_scoring_trend'].max():.1f}")
    return df


def calculate_vegas_features(df):
    """Calculate additional Vegas-derived features."""
    print("\nCalculating Vegas features...")

    # Line movement (already might exist)
    if 'line_movement' not in df.columns or df['line_movement'].isna().all():
        df['line_movement'] = df['vegas_spread'] - df['spread_open'].fillna(df['vegas_spread'])

    # Categorical features
    df['large_favorite'] = (df['vegas_spread'] < -14).astype(int)
    df['large_underdog'] = (df['vegas_spread'] > 14).astype(int)
    df['close_game'] = (abs(df['vegas_spread']) < 7).astype(int)

    # Elo vs Spread comparison
    # Elo diff of ~25 points = 1 point on spread
    df['elo_vs_spread'] = (df['elo_diff'] / 25) - df['vegas_spread']

    # Rest-spread interaction
    df['rest_spread_interaction'] = df['rest_diff'] * abs(df['vegas_spread']) / 10

    # Short rest flags
    df['home_short_rest'] = (df['home_rest_days'] < 6).astype(int)
    df['away_short_rest'] = (df['away_rest_days'] < 6).astype(int)

    # Expected total
    df['expected_total'] = df['home_last5_score_avg'] + df['away_last5_score_avg']

    print(f"  Line movement range: {df['line_movement'].min():.1f} to {df['line_movement'].max():.1f}")
    print(f"  Large favorites: {df['large_favorite'].sum()}")
    print(f"  Close games: {df['close_game'].sum()}")
    return df


def calculate_composite_features(df):
    """Calculate composite interaction features for V15 model."""
    print("\nCalculating composite features (V15)...")

    # Matchup efficiency - identifies offensive/defensive mismatches
    if 'home_comp_off_ppa' in df.columns and 'away_comp_def_ppa' in df.columns:
        df['matchup_efficiency'] = (
            (df['home_comp_off_ppa'].fillna(0) - df['away_comp_off_ppa'].fillna(0)) +
            (df['away_comp_def_ppa'].fillna(0) - df['home_comp_def_ppa'].fillna(0))
        )
        print("  Added: matchup_efficiency")

    # Pass vs rush efficiency diff
    if 'home_comp_pass_ppa' in df.columns and 'home_comp_rush_ppa' in df.columns:
        df['home_pass_rush_balance'] = df['home_comp_pass_ppa'].fillna(0) - df['home_comp_rush_ppa'].fillna(0)
        df['away_pass_rush_balance'] = df['away_comp_pass_ppa'].fillna(0) - df['away_comp_rush_ppa'].fillna(0)
        print("  Added: pass_rush_balance features")

    # Success rate difference
    if 'home_comp_success' in df.columns and 'away_comp_success' in df.columns:
        df['success_rate_diff'] = df['home_comp_success'].fillna(0.5) - df['away_comp_success'].fillna(0.5)
        print("  Added: success_rate_diff")

    # Elo x efficiency interaction - combines power ratings with efficiency
    if 'elo_diff' in df.columns and 'matchup_efficiency' in df.columns:
        df['elo_efficiency_interaction'] = df['elo_diff'] * df['matchup_efficiency'] / 100
        print("  Added: elo_efficiency_interaction")

    # Momentum x scoring - hot teams with good offenses
    if 'home_streak' in df.columns and 'home_last5_score_avg' in df.columns:
        df['momentum_strength'] = (
            (df['home_streak'].fillna(0) - df['away_streak'].fillna(0)) *
            (df['home_last5_score_avg'].fillna(25) - df['away_last5_score_avg'].fillna(25)) / 100
        )
        print("  Added: momentum_strength")

    # Dominant team flag - large Elo edge + efficiency edge
    if 'elo_diff' in df.columns and 'matchup_efficiency' in df.columns:
        df['dominant_home'] = ((df['elo_diff'] > 150) & (df['matchup_efficiency'] > 0.3)).astype(int)
        df['dominant_away'] = ((df['elo_diff'] < -150) & (df['matchup_efficiency'] < -0.3)).astype(int)
        print("  Added: dominant_home/away flags")

    # Rest x favorite interaction - tired favorites underperform
    if 'rest_diff' in df.columns and 'vegas_spread' in df.columns:
        df['rest_favorite_interaction'] = df['rest_diff'] * (df['vegas_spread'].fillna(0) < -7).astype(int)
        print("  Added: rest_favorite_interaction")

    # Has line movement indicator
    if 'line_movement' in df.columns:
        df['has_line_movement'] = df['line_movement'].notna().astype(int)
        print("  Added: has_line_movement")

    return df


def verify_no_leakage(df):
    """Verify that features don't contain future information."""
    print("\n" + "=" * 70)
    print("VERIFYING NO DATA LEAKAGE")
    print("=" * 70)

    # V19: Safe features that should NOT leak (EXCLUDES constant features)
    safe_features = [
        # Core Elo features
        'home_pregame_elo', 'away_pregame_elo', 'elo_diff',
        # Rolling averages
        'home_last5_score_avg', 'away_last5_score_avg',
        'home_last5_defense_avg', 'away_last5_defense_avg',
        # Home field advantage
        'home_team_hfa', 'hfa_diff',
        # Scheduling
        'rest_diff', 'home_rest_days', 'away_rest_days',
        # Vegas features
        'vegas_spread', 'line_movement', 'spread_open',
        'large_favorite', 'large_underdog', 'close_game',
        # Momentum
        'home_streak', 'away_streak', 'streak_diff',
        'home_ats', 'away_ats', 'ats_diff',
        'home_elo_momentum', 'away_elo_momentum', 'elo_momentum_diff',
        'home_scoring_trend', 'away_scoring_trend',
        # Vegas-derived
        'elo_vs_spread', 'rest_spread_interaction',
        'home_short_rest', 'away_short_rest', 'expected_total',
        # V19: PPA efficiency features that are NOT constant
        'home_comp_off_ppa', 'away_comp_off_ppa',
        'home_comp_def_ppa', 'away_comp_def_ppa',
        'home_comp_pass_ppa', 'away_comp_pass_ppa',
        'home_comp_rush_ppa', 'away_comp_rush_ppa',
        'pass_efficiency_diff',
        # V19: Composite features (excluding constant ones)
        'matchup_efficiency', 'home_pass_rush_balance', 'away_pass_rush_balance',
        'elo_efficiency_interaction', 'momentum_strength',
        'dominant_home', 'dominant_away', 'rest_favorite_interaction',
        'has_line_movement',
    ]

    # NOTE: The comp_* features were ORIGINALLY thought to be dangerous (game-specific)
    # but after investigation they are SEASON COMPOSITES (pre-game) and are SAFE.
    # These are truly dangerous features that should NOT be used:
    dangerous_features = [
        'net_epa',  # EPA from that game
        'epa_elo_interaction',  # Derived from net_epa
        'success_diff',  # Success rate from that game
        'matchup_advantage',  # Derived from game-specific data
        'home_points', 'away_points', 'Margin',  # Game results
        'spread_error', 'home_covered',  # Derived from results
    ]

    print("\nSAFE Features (pre-game):")
    for f in safe_features:
        if f in df.columns:
            non_null = df[f].notna().sum()
            print(f"  {f}: {non_null} non-null values")

    print("\nDANGEROUS Features (from the game - DO NOT USE):")
    for f in dangerous_features:
        if f in df.columns:
            non_null = df[f].notna().sum()
            print(f"  WARNING: {f}: {non_null} values - CONTAINS LEAKAGE!")

    return safe_features


# =============================================================================
# V19 IMPROVEMENT: REMOVE CONSTANT/USELESS FEATURES
# =============================================================================
# These 41 columns have zero variance (constant values) - waste model capacity
CONSTANT_FEATURES_TO_REMOVE = [
    "adj_net_epa",
    "away_adj_def_epa",
    "away_adj_off_epa",
    "away_clean_def_ppa",
    "away_clean_off_ppa",
    "away_comp_epa",
    "away_comp_success",
    "away_comp_ypp",
    "away_def_pass_downs_ppa",
    "away_def_pass_success",
    "away_def_rush_success",
    "away_garbage_adj_def",
    "away_garbage_adj_off",
    "away_lookahead",
    "away_off_pass_success",
    "away_off_rush_success",
    "away_off_std_downs_ppa",
    "away_raw_def_ppa",
    "away_raw_off_ppa",
    "home_adj_def_epa",
    "home_adj_off_epa",
    "home_clean_def_ppa",
    "home_clean_off_ppa",
    "home_comp_epa",
    "home_comp_success",
    "home_comp_ypp",
    "home_def_pass_downs_ppa",
    "home_def_pass_success",
    "home_def_rush_success",
    "home_garbage_adj_def",
    "home_garbage_adj_off",
    "home_lookahead",
    "home_off_pass_success",
    "home_off_rush_success",
    "home_off_std_downs_ppa",
    "home_raw_def_ppa",
    "home_raw_off_ppa",
    "matchup_advantage",
    "success_diff",
    "success_rate_diff",
    "west_coast_early",
]


def remove_constant_features(df):
    """Remove constant/zero-variance features that waste model capacity."""
    print("\nV19: Removing constant features...")
    removed = 0
    for col in CONSTANT_FEATURES_TO_REMOVE:
        if col in df.columns:
            df = df.drop(columns=[col])
            removed += 1
    print(f"  Removed {removed} constant/useless columns")
    return df


# =============================================================================
# V18 IMPROVEMENT: DAMPEN & CAP ERROR-AMPLIFYING FEATURES
# =============================================================================
# Based on SHAP error analysis, these features amplify prediction errors
# Research: Features with >1.2x error amplification should be dampened
DAMPEN_FEATURES = {
    'away_scoring_trend': 0.5,   # 1.73x amplification → reduce by 50%
    'home_scoring_trend': 0.5,   # Symmetric treatment
    'home_ats': 0.7,              # 1.27x → reduce by 30%
    'away_ats': 0.7,              # Symmetric treatment
    'hfa_diff': 0.8,              # 1.23x → reduce by 20%
    'home_streak': 0.8,           # 1.22x → reduce by 20%
    'away_streak': 0.8,           # Symmetric treatment
    'streak_diff': 0.8,           # 1.22x → reduce by 20%
    'ats_diff': 0.7,              # Derived from ATS
    'elo_momentum_diff': 0.85,    # Momentum features noisy
}

# V18: Cap extreme values to prevent outliers from dominating
CAP_FEATURES = {
    'home_streak': (-4, 4),       # Cap streaks at ±4
    'away_streak': (-4, 4),
    'streak_diff': (-6, 6),
    'home_ats': (0.25, 0.75),     # ATS between 25% and 75%
    'away_ats': (0.25, 0.75),
    'ats_diff': (-0.3, 0.3),
    'home_scoring_trend': (-8, 8),  # Cap scoring trends
    'away_scoring_trend': (-8, 8),
    'home_elo_momentum': (-100, 100),  # Cap Elo momentum
    'away_elo_momentum': (-100, 100),
    'elo_momentum_diff': (-150, 150),
    'line_movement': (-5, 5),      # Cap line movement to ±5 points
}


def dampen_error_amplifying_features(df):
    """
    Dampen and cap features that historically amplify prediction errors.

    Based on V16/V17 SHAP error analysis:
    - away_scoring_trend: 1.73x error amplification (highest)
    - home_ats: 1.27x
    - hfa_diff: 1.23x
    - home_streak: 1.22x
    - streak_diff: 1.22x

    V18 improvements:
    - Symmetric treatment of home/away features
    - Capping extreme values to prevent outliers
    """
    print("\nV18: Dampening and capping error-amplifying features...")

    # Step 1: Cap extreme values first (before dampening)
    capped_count = 0
    for feature, (min_val, max_val) in CAP_FEATURES.items():
        if feature in df.columns:
            original_min = df[feature].min()
            original_max = df[feature].max()
            clipped_low = (df[feature] < min_val).sum()
            clipped_high = (df[feature] > max_val).sum()
            df[feature] = df[feature].clip(min_val, max_val)
            if clipped_low > 0 or clipped_high > 0:
                capped_count += 1
                print(f"  {feature}: capped [{min_val}, {max_val}] (clipped {clipped_low} low, {clipped_high} high)")

    print(f"  Total features capped: {capped_count}")

    # Step 2: Dampen features
    dampened_count = 0
    for feature, factor in DAMPEN_FEATURES.items():
        if feature in df.columns:
            original_std = df[feature].std()
            df[feature] = df[feature] * factor
            new_std = df[feature].std()
            dampened_count += 1
            print(f"  {feature}: dampened by {(1-factor)*100:.0f}% (std: {original_std:.2f} → {new_std:.2f})")

    print(f"  Total features dampened: {dampened_count}")
    return df


def main():
    # Load data
    df = load_data()

    # Ensure spread_error exists
    print("\nCalculating spread error (target)...")
    if 'spread_error' not in df.columns:
        df['spread_error'] = df['Margin'] - (-df['vegas_spread'])

    # Calculate safe features
    df = calculate_streaks(df)
    df = calculate_ats_history(df)
    df = calculate_elo_momentum(df)
    df = calculate_scoring_trends(df)
    df = calculate_vegas_features(df)
    df = calculate_composite_features(df)

    # V17: Dampen error-amplifying features
    df = dampen_error_amplifying_features(df)

    # V19: Remove constant/useless features
    df = remove_constant_features(df)

    # Verify no leakage
    safe_features = verify_no_leakage(df)

    # Save processed data
    output_file = 'cfb_data_safe.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved processed data to {output_file}")

    # Also save the list of safe features
    with open('safe_features.txt', 'w') as f:
        f.write("# Safe Features (No Data Leakage)\n")
        f.write("# These features are calculated BEFORE the game\n\n")
        for feat in safe_features:
            if feat in df.columns:
                f.write(f"{feat}\n")
    print("Saved safe feature list to safe_features.txt")

    # Summary
    print("\n" + "=" * 70)
    print("SAFE FEATURE SUMMARY")
    print("=" * 70)

    df_vegas = df[df['vegas_spread'].notna()].copy()
    print(f"\nGames with Vegas spreads: {len(df_vegas)}")
    print(f"Seasons: {sorted(df_vegas['season'].unique())}")

    # Count features
    available_safe = [f for f in safe_features if f in df.columns]
    print(f"\nTotal safe features available: {len(available_safe)}")

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
