# Sharp Sports Predictor - Setup Guide

## Quick Start

### 1. Set Your API Key

Get your free API key from [College Football Data](https://collegefootballdata.com/key), then set it:

```bash
export CFBD_API_KEY="your_key_here"
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Fetch Data

```bash
python fetch_cfb_data.py
python feature_engineer.py
```

### 4. Train Model

```bash
python train_v8.py   # Recommended (uses all improvements)
# or
python train_pro_stacking.py  # Original stacking model
```

### 5. Run the App

```bash
streamlit run app_v8.py
# or
streamlit run app.py  # Original version
```

## Project Structure

```
sharp_sports_predictor/
├── src/                    # New modular code
│   ├── data/              # Data fetching and feature engineering
│   │   ├── fetcher.py     # API data fetching with caching
│   │   ├── feature_engineer.py  # Feature calculations
│   │   └── momentum.py    # Streak/momentum tracking
│   ├── models/            # Model training and prediction
│   │   ├── ensemble.py    # Ensemble model training
│   │   ├── predictor.py   # Game prediction interface
│   │   └── optimization.py # Weight optimization & compression
│   ├── prediction/        # Betting logic
│   │   ├── monte_carlo.py # Monte Carlo simulation
│   │   └── kelly.py       # Kelly Criterion sizing
│   └── utils/             # Utilities
│       ├── config.py      # Centralized configuration
│       ├── logging_config.py  # Logging setup
│       ├── validation.py  # Data validation
│       └── cache.py       # Caching layer
├── tests/                 # Unit tests
├── app_v8.py             # New Streamlit app
├── app.py                # Original Streamlit app
├── train_v8.py           # New training script
├── config.py             # API configuration
└── requirements.txt      # Dependencies
```

## Key Improvements in V8

### Security
- API keys via environment variables (not in code)
- No hardcoded credentials

### Code Quality
- Centralized configuration in `src/utils/config.py`
- Proper error handling with logging
- Data validation layer
- Modular package structure

### Features
- **Momentum tracking**: Win/loss streaks, ATS streaks
- **Line movement**: Track opening vs current spreads
- **Weight optimization**: Cross-validation for ensemble weights
- **Model compression**: Smaller model files

### Performance
- API response caching
- Efficient feature calculation

## Configuration

All configuration is in `src/utils/config.py`. Key settings:

```python
# Betting thresholds
kelly_fraction = 0.25       # Quarter Kelly
high_variance_threshold = 7.0
buy_threshold = 0.55
fade_threshold = 0.45

# Monte Carlo
monte_carlo_simulations = 10000
monte_carlo_std_dev = 14.0

# Model hyperparameters
hgb_max_iter = 100
hgb_max_depth = 3
hgb_learning_rate = 0.05
```

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

## Troubleshooting

### "CFBD_API_KEY not set"
Set the environment variable:
```bash
export CFBD_API_KEY="your_key_here"
```

### "No model file found"
Run the training script first:
```bash
python train_v8.py
```

### "Historical data not found"
Fetch and process the data:
```bash
python fetch_cfb_data.py
python feature_engineer.py
```

## Model Architecture

The V8 model uses a **StackingRegressor** with:

1. **Base Models (Experts)**:
   - HistGradientBoosting: Handles missing values, fast training
   - RandomForest: Ensemble of trees, robust
   - Ridge: Linear model, prevents overfitting

2. **Meta-Learner (Boss)**:
   - RidgeCV: Cross-validated Ridge that learns optimal expert weights

The "Confusion Filter" uses variance between base model predictions to identify uncertain games (PASS recommendation when variance > 7.0).
