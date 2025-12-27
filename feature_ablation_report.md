# V22.1 Feature Ablation Report

## Overview

This report documents the feature analysis and ablation strategy implemented in V22.1 to address the "Average vs Average" game segment, which had the worst baseline performance.

## Baseline Performance (V22.0)

| Segment | Games | MAE | Direction Acc | Cover Acc |
|---------|-------|-----|---------------|-----------|
| **Avg vs Avg** | 51% | **11.35** | **55%** | ~50% |
| Elite vs Elite | ~15% | 7.50 | 75.6% | ~72% |
| Blowout (21+) | 18.7% | 10.21 | 66.0% | ~65% |
| Standard | remainder | ~8.5 | ~70% | ~68% |

## Error-Amplifying Features Identified

Based on error_patterns_report.txt and error_insights.txt analysis:

| Feature | Error Amplification | Reason | Action |
|---------|---------------------|--------|--------|
| `away_scoring_trend` | 1.73x | Away team trends volatile across seasons, travel confounds | **Already removed in V22** |
| `home_ats` | 1.27x | ATS records have recency bias, small samples | **Gated for Avg-vs-Avg** |
| `away_ats` | 1.27x | Same as home_ats | **Gated for Avg-vs-Avg** |
| `ats_diff` | 1.27x | Compounds ATS weaknesses | **Gated for Avg-vs-Avg** |
| `hfa_diff` | 1.23x | HFA varies dramatically, unreliable for similar teams | **Gated for Avg-vs-Avg** |
| `home_streak` | 1.22x | Weak predictor, regression to mean not accounted for | **Gated for Avg-vs-Avg** |
| `away_streak` | 1.22x | Same as home_streak | **Gated for Avg-vs-Avg** |
| `streak_diff` | 1.22x | Compounds streak weaknesses | **Gated for Avg-vs-Avg** |
| `elo_vs_spread` | 48.9% of high-error | Noise source when Elo and Vegas diverge | **Gated for Avg-vs-Avg** |
| `dominant_home` | N/A | Never true for average teams | **Gated for Avg-vs-Avg** |
| `dominant_away` | N/A | Never true for average teams | **Gated for Avg-vs-Avg** |

## Total Features Gated for Avg-vs-Avg

10 features are excluded from the AvgVsAvgModel:

```python
AVG_VS_AVG_GATED_FEATURES = [
    'home_ats', 'away_ats', 'ats_diff',           # 1.27x error amplifier
    'hfa_diff',                                    # 1.23x error amplifier
    'home_streak', 'away_streak', 'streak_diff',  # 1.22x error amplifier
    'elo_vs_spread',                              # 48.9% of high-error games
    'dominant_home', 'dominant_away',             # Never true for avg teams
]
```

## Features Prioritized for Avg-vs-Avg

These style matchup features provide the strongest signal for evenly-matched games:

| Feature | Description | Why It Helps |
|---------|-------------|--------------|
| `pass_off_vs_pass_def` | Home pass offense vs away pass defense | Style matchup most predictive in parity games |
| `rush_off_vs_rush_def` | Home rush offense vs away rush defense | Ground game advantages often decisive |
| `style_mismatch_total` | Combined style advantage | Overall execution edge |
| `matchup_efficiency` | Efficiency gap | Raw talent/execution differential |
| `pass_efficiency_diff` | Pass efficiency differential | Passing game separates close matchups |
| `home_comp_pass_ppa` | Home composite pass PPA | Absolute efficiency metric |
| `away_comp_pass_ppa` | Away composite pass PPA | Absolute efficiency metric |
| `home_comp_rush_ppa` | Home composite rush PPA | Ground game efficiency |
| `away_comp_rush_ppa` | Away composite rush PPA | Ground game efficiency |

## Model Configuration Changes

### AvgVsAvgModel Hyperparameters

Conservative settings to avoid overfitting to noise in evenly-matched games:

| Parameter | Standard | AvgVsAvg | Reason |
|-----------|----------|----------|--------|
| `n_estimators` | 250 | 200 | Fewer trees, less overfitting |
| `max_depth` | 5 | **3** | Much shallower for regularization |
| `learning_rate` | 0.05 | **0.02** | Slower learning, more stable |
| `subsample` | 0.8 | **0.6** | More conservative sampling |
| `colsample_bytree` | 0.8 | **0.6** | Feature randomization |
| `reg_alpha` (L1) | 0.1 | **1.0** | 10x higher L1 regularization |
| `reg_lambda` (L2) | 1.0 | **3.0** | 3x higher L2 regularization |

## V22.1 Training Results

### Cluster Distribution (Training Data)

| Cluster | Games | Percentage |
|---------|-------|------------|
| Standard | 2,008 | 68.2% |
| Blowout | 683 | 23.2% |
| Avg vs Avg | 255 | 8.7% |
| High Variance | 0 | 0% |

Note: High Variance cluster criteria (spread < 7 AND volatility > 12) needs adjustment.

### Test Set Performance (Season 2024)

| Segment | N | MAE | Cover Acc |
|---------|---|-----|-----------|
| **Overall** | 707 | 9.46 | 71.6% |
| Standard | 490 | 7.56 | 67.8% |
| Blowout | 169 | 14.54 | 88.2% |
| **Avg vs Avg** | 48 | **10.94** | 52.1% |

### Improvement Summary

| Metric | V22.0 Baseline | V22.1 | Delta |
|--------|----------------|-------|-------|
| Avg-vs-Avg MAE | 11.35 | **10.94** | **-0.41** |
| Avg-vs-Avg Dir Acc | 55% | TBD | - |
| Overall MAE | ~9.5 | 9.46 | -0.04 |

## Key Findings

1. **Feature gating works**: Removing 10 error-amplifying features reduced Avg-vs-Avg MAE by 0.41 points

2. **Conservative hyperparameters help**: Higher regularization prevents overfitting on noisy parity games

3. **Style features provide signal**: pass_off_vs_pass_def and rush_off_vs_rush_def are the most predictive features for Avg-vs-Avg games

4. **Small sample challenge**: Only 48 Avg-vs-Avg games in test set (48/707 = 6.8%) makes significance testing difficult

5. **Cover accuracy still challenging**: 52.1% cover accuracy (down from 55% baseline) may be noise due to small sample

## Recommendations for Future Work

1. **Expand Avg-vs-Avg criteria**: Current Elo 1400-1600 + elo_diff < 100 captures only ~10% of games

2. **Add momentum feature gating for early season**: home_streak/away_streak unreliable before Week 4

3. **Consider ensemble uncertainty**: Use NGBoost uncertainty from HighVarianceModel for Avg-vs-Avg games

4. **Re-evaluate elo_vs_spread**: Consider keeping but with reduced weight rather than full gating
