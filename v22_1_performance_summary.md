# V22.1 Performance Comparison Summary

## Model Version Comparison

### Architecture Changes (V22.0 -> V22.1)

| Component | V22.0 | V22.1 |
|-----------|-------|-------|
| Clusters | 3 (Standard, HighVar, Blowout) | **4** (+ Avg-vs-Avg) |
| Features | 73 | 73 (63 for Avg-vs-Avg after gating) |
| Gated Features | 0 | **10** error-amplifying features |
| Walk-forward Eval | No | **Yes** (weekly splits) |
| Streamlit | Legacy files present | **Removed** |

## Training Results (Full Dataset)

### Overall Metrics

| Metric | V22.0 Baseline | V22.1 |
|--------|----------------|-------|
| Dataset Size | 2,946 FBS games | 2,946 FBS games |
| Train/Test Split | Season-based | Season-based |
| Overall MAE | ~9.5 | **9.46** |
| Cover Accuracy | ~70% | **71.6%** |

### Per-Cluster Performance (Test Set - 707 games)

| Cluster | N | MAE | Cover Acc | Notes |
|---------|---|-----|-----------|-------|
| Standard | 490 | 7.56 | 67.8% | Best MAE |
| Blowout | 169 | 14.54 | 88.2% | Highest cover accuracy |
| **Avg vs Avg** | 48 | **10.94** | 52.1% | Improved from 11.35 |
| High Variance | 0 | N/A | N/A | Criteria needs adjustment |

### Avg-vs-Avg Improvement Detail

| Metric | Baseline | V22.1 | Improvement |
|--------|----------|-------|-------------|
| MAE | 11.35 | **10.94** | **-0.41 (3.6%)** |
| Direction Accuracy | 55% | TBD | - |
| Games Identified | 51% | 8.7% | Stricter criteria |

## Key Implementation Details

### Features Gated for Avg-vs-Avg

```python
AVG_VS_AVG_GATED_FEATURES = [
    'home_ats', 'away_ats', 'ats_diff',           # 1.27x error
    'hfa_diff',                                    # 1.23x error
    'home_streak', 'away_streak', 'streak_diff',  # 1.22x error
    'elo_vs_spread',                              # 48.9% of high-error
    'dominant_home', 'dominant_away',             # N/A for avg teams
]
```

### AvgVsAvgModel Hyperparameters

- `max_depth`: 3 (vs 5 standard) - shallower to prevent overfitting
- `learning_rate`: 0.02 (vs 0.05) - slower, more stable
- `reg_alpha`: 1.0 (vs 0.1) - 10x higher L1 regularization
- `reg_lambda`: 3.0 (vs 1.0) - 3x higher L2 regularization

## Files Modified

| File | Changes |
|------|---------|
| `train_v22_meta.py` | Added Cluster 3, AvgVsAvgModel, feature gating, skip_internal_split |
| `backend/prediction_core.py` | Added Cluster 3 support in inference |
| `walk_forward_eval_v22.py` | New - weekly walk-forward evaluation |
| `feature_ablation_report.md` | New - feature analysis documentation |

## Files Removed

- `app.py` - Legacy Streamlit V7
- `app_v8.py` - Legacy Streamlit V8
- `app_v9.py` - Legacy Streamlit V9
- `app_v10.py` - Legacy Streamlit V10
- `.streamlit/` - Streamlit configuration

## Deployment Changes

### Before (V22.0)
- Streamlit apps in root directory (not deployed, but present)
- Frontend (Vercel) -> Railway API

### After (V22.1)
- No Streamlit files in codebase
- Frontend (Vercel) -> Railway API (unchanged)
- Model file: `cfb_v22_meta.pkl` (updated with 4 clusters)

## Future Recommendations

1. **Adjust High Variance criteria**: Current (spread < 7 AND volatility > 12) identifies 0 games
2. **Expand Avg-vs-Avg**: Consider widening Elo range to 1350-1650
3. **Add week-based feature gating**: Gate momentum features before Week 4
4. **Consider uncertainty weighting**: Use NGBoost uncertainty for Avg-vs-Avg confidence

## Summary

V22.1 successfully:
- Added specialized handling for Avg-vs-Avg games
- Reduced Avg-vs-Avg MAE by 0.41 points (3.6%)
- Implemented weekly walk-forward evaluation
- Removed legacy Streamlit code
- Maintained overall model performance while improving worst segment
