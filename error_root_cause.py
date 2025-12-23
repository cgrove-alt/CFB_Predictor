"""
Root Cause Analysis for High-Error Predictions.

Uses SHAP values to understand WHY specific predictions failed,
identifying features that amplify prediction errors.

NO SHORTCUTS - Deep analysis of every high-error game.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Try to import SHAP, provide fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Installing...")
    import subprocess
    subprocess.run(['pip3', 'install', 'shap'], capture_output=True)
    try:
        import shap
        SHAP_AVAILABLE = True
    except:
        print("Could not install SHAP. Will use feature importance fallback.")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = 'cfb_data_safe.csv'
PREDICTIONS_FILE = 'predictions_2025_comprehensive.csv'
CONFIG_FILE = 'cfb_v15_config.pkl'
HIGH_ERROR_PCT = 0.20  # Top 20% of errors
OUTPUT_ROOT_CAUSES = 'high_error_root_causes.csv'
OUTPUT_FEATURE_AMP = 'feature_error_amplification.csv'

# Load config
V15_CONFIG = joblib.load(CONFIG_FILE)
FEATURE_COLS = V15_CONFIG['features']


def load_data():
    """Load predictions and original data."""
    print("Loading data...")

    # Load predictions
    pred_df = pd.read_csv(PREDICTIONS_FILE)
    print(f"Loaded {len(pred_df)} predictions")

    # Load original data for training
    data_df = pd.read_csv(DATA_FILE)
    data_df = data_df[data_df['vegas_spread'].notna()].copy()

    return pred_df, data_df


def train_analysis_model(data_df):
    """Train model for SHAP analysis."""
    print("\nTraining model for SHAP analysis...")

    # Use all pre-2025 data for training
    train_df = data_df[data_df['season'] < 2025].copy()

    # Prepare features
    X_train = train_df[FEATURE_COLS].fillna(0)

    # Ensure spread_error exists
    if 'spread_error' not in train_df.columns:
        train_df['margin'] = train_df['home_points'] - train_df['away_points']
        train_df['spread_error'] = train_df['margin'] - (-train_df['vegas_spread'])

    y_train = train_df['spread_error']

    # Train model
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print(f"Model trained on {len(X_train)} games")

    return model


def identify_high_error_games(pred_df):
    """Identify the top HIGH_ERROR_PCT of prediction errors."""
    error_threshold = pred_df['prediction_error'].quantile(1 - HIGH_ERROR_PCT)

    high_error_df = pred_df[pred_df['prediction_error'] >= error_threshold].copy()

    print(f"\nHigh-error games identified:")
    print(f"  Threshold: {error_threshold:.2f} points")
    print(f"  Games: {len(high_error_df)} ({len(high_error_df)/len(pred_df)*100:.1f}%)")
    print(f"  Mean error: {high_error_df['prediction_error'].mean():.2f}")
    print(f"  Max error: {high_error_df['prediction_error'].max():.2f}")

    return high_error_df


def analyze_with_shap(model, pred_df, high_error_df):
    """Use SHAP to analyze high-error predictions."""
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("="*60)

    if not SHAP_AVAILABLE:
        print("SHAP not available, using feature importance fallback")
        return analyze_with_importance(model, pred_df, high_error_df)

    # Get feature columns from predictions (they have 'feature_' prefix)
    feature_pred_cols = [f'feature_{f}' for f in FEATURE_COLS if f'feature_{f}' in pred_df.columns]

    # Prepare feature data
    X_all = pred_df[[col for col in feature_pred_cols]].fillna(0)
    X_all.columns = [col.replace('feature_', '') for col in X_all.columns]

    X_high_error = high_error_df[[col for col in feature_pred_cols]].fillna(0)
    X_high_error.columns = [col.replace('feature_', '') for col in X_high_error.columns]

    # Ensure columns match model features
    missing_cols = [f for f in FEATURE_COLS if f not in X_all.columns]
    for col in missing_cols:
        X_all[col] = 0
        X_high_error[col] = 0

    X_all = X_all[FEATURE_COLS]
    X_high_error = X_high_error[FEATURE_COLS]

    print(f"Calculating SHAP values for {len(X_high_error)} high-error games...")

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Get SHAP values for high-error games
    shap_values_high = explainer.shap_values(X_high_error)

    # Get SHAP values for all games (sample for efficiency)
    sample_size = min(500, len(X_all))
    X_sample = X_all.sample(n=sample_size, random_state=42)
    shap_values_all = explainer.shap_values(X_sample)

    # Analyze per-game root causes
    root_causes = analyze_per_game_shap(high_error_df, shap_values_high, FEATURE_COLS)

    # Analyze feature error amplification
    feature_amp = analyze_feature_amplification(shap_values_high, shap_values_all, FEATURE_COLS)

    return root_causes, feature_amp


def analyze_with_importance(model, pred_df, high_error_df):
    """Fallback analysis using feature importance."""
    print("Using feature importance for analysis...")

    # Get feature importances
    importances = model.feature_importances_

    feature_imp = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': importances,
    }).sort_values('importance', ascending=False)

    # For root causes, we'll use feature values
    root_causes = []

    for idx, row in high_error_df.head(50).iterrows():
        game_features = {}
        for feat in FEATURE_COLS:
            col_name = f'feature_{feat}'
            if col_name in row.index:
                game_features[feat] = row[col_name]

        # Get top features by importance
        top_features = feature_imp.head(5)['feature'].tolist()
        feature_values = [game_features.get(f, 0) for f in top_features]

        root_causes.append({
            'game': f"{row['away_team']} @ {row['home_team']}",
            'week': row['week'],
            'prediction_error': row['prediction_error'],
            'predicted': row['predicted_spread_error'],
            'actual': row['actual_spread_error'],
            'top_features': str(top_features),
            'feature_values': str(feature_values),
        })

    feature_amp = feature_imp.copy()
    feature_amp['error_amplification'] = 1.0  # Placeholder

    return pd.DataFrame(root_causes), feature_amp


def analyze_per_game_shap(high_error_df, shap_values, feature_cols):
    """Analyze SHAP values for each high-error game."""
    print("\nAnalyzing per-game root causes...")

    root_causes = []

    for i, (idx, row) in enumerate(high_error_df.iterrows()):
        if i >= len(shap_values):
            break

        game_shap = pd.Series(shap_values[i], index=feature_cols)

        # Get top 5 features by absolute SHAP value
        top_contributors = game_shap.abs().nlargest(5)
        top_features = list(top_contributors.index)
        shap_impacts = [game_shap[f] for f in top_features]

        root_causes.append({
            'game': f"{row['away_team']} @ {row['home_team']}",
            'week': row['week'],
            'prediction_error': row['prediction_error'],
            'predicted': row['predicted_spread_error'],
            'actual': row['actual_spread_error'],
            'top_feature_1': top_features[0] if len(top_features) > 0 else '',
            'shap_1': shap_impacts[0] if len(shap_impacts) > 0 else 0,
            'top_feature_2': top_features[1] if len(top_features) > 1 else '',
            'shap_2': shap_impacts[1] if len(shap_impacts) > 1 else 0,
            'top_feature_3': top_features[2] if len(top_features) > 2 else '',
            'shap_3': shap_impacts[2] if len(shap_impacts) > 2 else 0,
            'top_feature_4': top_features[3] if len(top_features) > 3 else '',
            'shap_4': shap_impacts[3] if len(shap_impacts) > 3 else 0,
            'top_feature_5': top_features[4] if len(top_features) > 4 else '',
            'shap_5': shap_impacts[4] if len(shap_impacts) > 4 else 0,
        })

    return pd.DataFrame(root_causes)


def analyze_feature_amplification(shap_high, shap_all, feature_cols):
    """Identify features that amplify errors."""
    print("\nAnalyzing feature error amplification...")

    # Mean absolute SHAP for high-error games
    mean_shap_high = np.abs(shap_high).mean(axis=0)

    # Mean absolute SHAP for all games
    mean_shap_all = np.abs(shap_all).mean(axis=0)

    # Error amplification ratio
    amplification = mean_shap_high / (mean_shap_all + 1e-10)

    feature_amp = pd.DataFrame({
        'feature': feature_cols,
        'mean_shap_high_error': mean_shap_high,
        'mean_shap_overall': mean_shap_all,
        'error_amplification': amplification,
    }).sort_values('error_amplification', ascending=False)

    return feature_amp


def generate_insights(root_causes, feature_amp, high_error_df):
    """Generate actionable insights from the analysis."""
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    insights = []

    # 1. Most problematic features (high amplification)
    top_problematic = feature_amp.head(10)
    print("\n1. FEATURES THAT AMPLIFY ERRORS:")
    print("-" * 40)
    for _, row in top_problematic.iterrows():
        if row['error_amplification'] > 1.2:
            print(f"   {row['feature']}: {row['error_amplification']:.2f}x amplification")
            insights.append(f"Feature '{row['feature']}' amplifies errors by {row['error_amplification']:.2f}x")

    # 2. Most stable features (low amplification)
    stable_features = feature_amp.tail(10)
    print("\n2. MOST STABLE FEATURES (low error amplification):")
    print("-" * 40)
    for _, row in stable_features.iterrows():
        if row['error_amplification'] < 0.8:
            print(f"   {row['feature']}: {row['error_amplification']:.2f}x")

    # 3. Common patterns in high-error games
    if 'top_feature_1' in root_causes.columns:
        print("\n3. MOST COMMON ROOT CAUSE FEATURES:")
        print("-" * 40)
        top_1_counts = root_causes['top_feature_1'].value_counts().head(5)
        for feat, count in top_1_counts.items():
            pct = count / len(root_causes) * 100
            print(f"   {feat}: {count} games ({pct:.1f}%)")
            insights.append(f"Feature '{feat}' is the top contributor in {pct:.1f}% of high-error games")

    # 4. Week patterns in high errors
    print("\n4. HIGH ERRORS BY WEEK:")
    print("-" * 40)
    week_errors = high_error_df.groupby('week').size()
    for week, count in week_errors.items():
        print(f"   Week {week}: {count} high-error games")

    # 5. Spread magnitude in high errors
    print("\n5. HIGH ERRORS BY SPREAD MAGNITUDE:")
    print("-" * 40)
    high_error_df['spread_bucket'] = pd.cut(
        high_error_df['spread_magnitude'],
        bins=[0, 3, 7, 14, 21, 100],
        labels=['Pick-em', 'Small', 'Medium', 'Large', 'Blowout']
    )
    spread_counts = high_error_df['spread_bucket'].value_counts()
    for bucket, count in spread_counts.items():
        pct = count / len(high_error_df) * 100
        print(f"   {bucket}: {count} ({pct:.1f}%)")

    return insights


def save_results(root_causes, feature_amp, insights):
    """Save analysis results."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save root causes
    root_causes.to_csv(OUTPUT_ROOT_CAUSES, index=False)
    print(f"Saved root causes to: {OUTPUT_ROOT_CAUSES}")

    # Save feature amplification
    feature_amp.to_csv(OUTPUT_FEATURE_AMP, index=False)
    print(f"Saved feature amplification to: {OUTPUT_FEATURE_AMP}")

    # Save insights
    with open('error_insights.txt', 'w') as f:
        f.write("CFB BETTING MODEL - ERROR ROOT CAUSE INSIGHTS\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("RECOMMENDATIONS FOR V16 MODEL:\n")
        f.write("="*60 + "\n")
        f.write("1. Add uncertainty features for high-error situations\n")
        f.write("2. Create meta-model to predict when primary model is unreliable\n")
        f.write("3. Add team-specific calibration features\n")
        f.write("4. Consider ensemble with situation-weighted models\n")

    print("Saved insights to: error_insights.txt")


def main():
    print("="*60)
    print("ROOT CAUSE ANALYSIS FOR HIGH-ERROR PREDICTIONS")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    pred_df, data_df = load_data()

    # Train model for SHAP analysis
    model = train_analysis_model(data_df)

    # Identify high-error games
    high_error_df = identify_high_error_games(pred_df)

    # Analyze with SHAP
    root_causes, feature_amp = analyze_with_shap(model, pred_df, high_error_df)

    # Generate insights
    insights = generate_insights(root_causes, feature_amp, high_error_df)

    # Save results
    save_results(root_causes, feature_amp, insights)

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\nNEXT STEPS:")
    print("1. Review high_error_root_causes.csv for per-game analysis")
    print("2. Review feature_error_amplification.csv for feature issues")
    print("3. Run train_v16_self_learning.py to build improved model")
    print("="*60)


if __name__ == "__main__":
    main()
