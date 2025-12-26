"""
V21 Graph Features - Common Opponent Analysis for Cross-Conference Games.

Instead of a full GNN (which requires complex graph structure maintenance),
this module uses graph-based reasoning to create features for cross-conference games.

Key insight: Cross-conference games lack head-to-head history, but we can:
1. Find common opponents between teams
2. Compare performance against shared opponents
3. Create "transitive" margin predictions

Example: If A beat C by 10 and B lost to C by 7, we estimate A is 17 points better than B.

Usage:
    python train_v21_graph.py
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')


def get_conference_for_team(team_name, conference_map=None):
    """Get conference for a team."""
    if conference_map is None:
        # Default conference mapping (2024 season)
        # This is a simplified version - production would use CFBD API
        conference_map = {
            # SEC
            'Alabama': 'SEC', 'Auburn': 'SEC', 'LSU': 'SEC', 'Georgia': 'SEC',
            'Florida': 'SEC', 'Tennessee': 'SEC', 'Texas A&M': 'SEC', 'Ole Miss': 'SEC',
            'Mississippi State': 'SEC', 'Arkansas': 'SEC', 'Missouri': 'SEC', 'Kentucky': 'SEC',
            'South Carolina': 'SEC', 'Vanderbilt': 'SEC', 'Texas': 'SEC', 'Oklahoma': 'SEC',
            # Big Ten
            'Ohio State': 'Big Ten', 'Michigan': 'Big Ten', 'Penn State': 'Big Ten',
            'Wisconsin': 'Big Ten', 'Iowa': 'Big Ten', 'Minnesota': 'Big Ten',
            'Nebraska': 'Big Ten', 'Northwestern': 'Big Ten', 'Purdue': 'Big Ten',
            'Illinois': 'Big Ten', 'Indiana': 'Big Ten', 'Maryland': 'Big Ten',
            'Rutgers': 'Big Ten', 'Michigan State': 'Big Ten',
            'USC': 'Big Ten', 'UCLA': 'Big Ten', 'Oregon': 'Big Ten', 'Washington': 'Big Ten',
            # ACC
            'Clemson': 'ACC', 'Florida State': 'ACC', 'Miami': 'ACC', 'NC State': 'ACC',
            'North Carolina': 'ACC', 'Duke': 'ACC', 'Wake Forest': 'ACC',
            'Louisville': 'ACC', 'Pittsburgh': 'ACC', 'Syracuse': 'ACC', 'Boston College': 'ACC',
            'Virginia': 'ACC', 'Virginia Tech': 'ACC', 'Georgia Tech': 'ACC',
            'Stanford': 'ACC', 'Cal': 'ACC', 'SMU': 'ACC',
            # Big 12
            'TCU': 'Big 12', 'Oklahoma State': 'Big 12', 'Kansas': 'Big 12',
            'Kansas State': 'Big 12', 'Baylor': 'Big 12', 'Texas Tech': 'Big 12',
            'West Virginia': 'Big 12', 'Iowa State': 'Big 12',
            'BYU': 'Big 12', 'UCF': 'Big 12', 'Cincinnati': 'Big 12', 'Houston': 'Big 12',
            'Arizona': 'Big 12', 'Arizona State': 'Big 12', 'Utah': 'Big 12', 'Colorado': 'Big 12',
            # PAC remnants (now in Big 12/Big Ten)
            'Oregon State': 'Independent', 'Washington State': 'Independent',
        }
    return conference_map.get(team_name, 'Unknown')


def is_cross_conference_game(home_team, away_team, conference_map=None):
    """Check if a game is cross-conference."""
    home_conf = get_conference_for_team(home_team, conference_map)
    away_conf = get_conference_for_team(away_team, conference_map)
    return home_conf != away_conf


class CommonOpponentAnalyzer:
    """
    Analyze teams through their common opponents.

    This creates transitive comparisons between teams that haven't played,
    which is especially valuable for cross-conference matchups.
    """

    def __init__(self):
        self.game_results = defaultdict(list)  # team -> [(opponent, margin, date)]
        self.opponent_graph = defaultdict(set)  # team -> set of opponents

    def build_from_df(self, df):
        """Build opponent graph from historical data."""
        print("Building common opponent graph...")

        df = df.sort_values(['season', 'week']).reset_index(drop=True)

        for idx, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            margin = row.get('Margin', 0)
            season = row['season']
            week = row['week']

            if pd.isna(margin):
                continue

            # Record game from both perspectives
            self.game_results[home].append({
                'opponent': away,
                'margin': margin,
                'season': season,
                'week': week,
                'is_home': True
            })

            self.game_results[away].append({
                'opponent': home,
                'margin': -margin,
                'season': season,
                'week': week,
                'is_home': False
            })

            # Build opponent graph
            self.opponent_graph[home].add(away)
            self.opponent_graph[away].add(home)

        print(f"  Teams in graph: {len(self.game_results)}")
        total_games = sum(len(games) for games in self.game_results.values()) // 2
        print(f"  Total games: {total_games}")

    def find_common_opponents(self, team_a, team_b, season=None, before_week=None):
        """Find common opponents between two teams."""
        if team_a not in self.opponent_graph or team_b not in self.opponent_graph:
            return []

        common = self.opponent_graph[team_a] & self.opponent_graph[team_b]

        if season is not None and before_week is not None:
            # Filter to only include games before a certain week
            valid_common = []
            for opp in common:
                # Check if both teams played this opponent before the specified week
                a_games = [g for g in self.game_results[team_a]
                          if g['opponent'] == opp and
                          (g['season'] < season or (g['season'] == season and g['week'] < before_week))]
                b_games = [g for g in self.game_results[team_b]
                          if g['opponent'] == opp and
                          (g['season'] < season or (g['season'] == season and g['week'] < before_week))]
                if a_games and b_games:
                    valid_common.append(opp)
            return valid_common

        return list(common)

    def calculate_transitive_margin(self, team_a, team_b, season=None, before_week=None):
        """
        Calculate expected margin using transitive comparison through common opponents.

        Returns:
            dict with 'transitive_margin', 'confidence', 'num_common_opponents'
        """
        common_opps = self.find_common_opponents(team_a, team_b, season, before_week)

        if not common_opps:
            return {
                'transitive_margin': 0,
                'confidence': 0,
                'num_common_opponents': 0,
                'margin_variance': 0
            }

        margins = []

        for opp in common_opps:
            # Get most recent game for each team vs this opponent
            a_games = [g for g in self.game_results[team_a]
                      if g['opponent'] == opp and
                      (season is None or g['season'] < season or
                       (g['season'] == season and g['week'] < before_week))]

            b_games = [g for g in self.game_results[team_b]
                      if g['opponent'] == opp and
                      (season is None or g['season'] < season or
                       (g['season'] == season and g['week'] < before_week))]

            if a_games and b_games:
                # Use most recent games
                a_margin = a_games[-1]['margin']
                b_margin = b_games[-1]['margin']

                # Transitive comparison: A vs Common vs B
                # If A beat Common by 10 and B lost to Common by 7,
                # A is estimated to be 17 points better than B
                transitive_edge = a_margin - b_margin
                margins.append(transitive_edge)

        if not margins:
            return {
                'transitive_margin': 0,
                'confidence': 0,
                'num_common_opponents': 0,
                'margin_variance': 0
            }

        # Average across common opponents
        transitive_margin = np.mean(margins)

        # Confidence based on agreement among common opponents
        margin_std = np.std(margins) if len(margins) > 1 else 15
        confidence = max(0, 1 - (margin_std / 30))  # Higher std = lower confidence

        return {
            'transitive_margin': transitive_margin,
            'confidence': confidence,
            'num_common_opponents': len(margins),
            'margin_variance': np.var(margins) if len(margins) > 1 else 0
        }


def generate_graph_features(home_team, away_team, analyzer, season, week):
    """
    Generate graph-based features for a game.

    Returns dict with:
    - transitive_margin: Expected margin from common opponents
    - transitive_confidence: How much we trust this estimate
    - num_common_opponents: Number of shared opponents
    - is_cross_conference: Whether this is a cross-conference game
    - graph_margin_variance: Variance in transitive estimates
    """
    # Check if cross-conference
    is_cross = is_cross_conference_game(home_team, away_team)

    # Get transitive comparison
    result = analyzer.calculate_transitive_margin(home_team, away_team, season, week)

    return {
        'transitive_margin': result['transitive_margin'],
        'transitive_confidence': result['confidence'],
        'num_common_opponents': result['num_common_opponents'],
        'is_cross_conference': int(is_cross),
        'graph_margin_variance': result['margin_variance'],
        # Weight transitive margin more heavily for cross-conference games
        'transitive_weighted': result['transitive_margin'] * (1.5 if is_cross else 1.0)
    }


def add_graph_features_to_df(df, analyzer):
    """Add graph-based features to dataframe."""
    print("\nAdding graph features to dataset...")

    graph_features = []

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        season = row['season']
        week = row['week']

        features = generate_graph_features(home, away, analyzer, season, week)
        graph_features.append(features)

    # Add columns to dataframe
    for key in graph_features[0].keys():
        df[key] = [f[key] for f in graph_features]

    # Summary stats
    cross_conf_games = df['is_cross_conference'].sum()
    with_common_opps = (df['num_common_opponents'] > 0).sum()

    print(f"  Cross-conference games: {cross_conf_games} ({100*cross_conf_games/len(df):.1f}%)")
    print(f"  Games with common opponents: {with_common_opps} ({100*with_common_opps/len(df):.1f}%)")
    print(f"  Avg common opponents: {df['num_common_opponents'].mean():.1f}")
    print(f"  Avg transitive margin: {df['transitive_margin'].mean():.1f}")

    return df


def main():
    print("=" * 70)
    print("V21 GRAPH FEATURES - COMMON OPPONENT ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('cfb_data_safe.csv')
    print(f"Loaded {len(df)} games")

    # Build common opponent analyzer
    analyzer = CommonOpponentAnalyzer()
    analyzer.build_from_df(df)

    # Add features
    df = add_graph_features_to_df(df, analyzer)

    # Check correlation with actual margin
    df_valid = df[df['Margin'].notna() & (df['num_common_opponents'] > 0)].copy()
    if len(df_valid) > 0:
        corr = df_valid['transitive_margin'].corr(df_valid['Margin'])
        print(f"\nTransitive margin correlation with actual: {corr:.3f}")

        # Check for cross-conference specifically
        df_cross = df_valid[df_valid['is_cross_conference'] == 1]
        if len(df_cross) > 0:
            cross_corr = df_cross['transitive_margin'].corr(df_cross['Margin'])
            print(f"Cross-conference correlation: {cross_corr:.3f}")

    # Save enhanced data
    output_file = 'cfb_data_graph.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved enhanced data to {output_file}")

    # Save analyzer for inference
    analyzer_data = {
        'game_results': dict(analyzer.game_results),
        'opponent_graph': {k: list(v) for k, v in analyzer.opponent_graph.items()},
        'created_at': datetime.now().isoformat()
    }
    with open('common_opponent_analyzer.json', 'w') as f:
        json.dump(analyzer_data, f)
    print("Saved analyzer to common_opponent_analyzer.json")

    print("\n" + "=" * 70)
    print("GRAPH FEATURES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
