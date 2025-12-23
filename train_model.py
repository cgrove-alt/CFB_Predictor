"""
Train a Gradient Boosting model to predict college football game margins.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Load data
print("Loading data...")
df = pd.read_csv('cfb_data.csv')

# Handle missing Elo values - fill with median
df['home_pregame_elo'] = df['home_pregame_elo'].fillna(df['home_pregame_elo'].median())
df['away_pregame_elo'] = df['away_pregame_elo'].fillna(df['away_pregame_elo'].median())

# Encode team names to numbers
print("Encoding team names...")
le_home = LabelEncoder()
le_away = LabelEncoder()

# Fit on all unique teams from both columns
all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
le_home.fit(all_teams)
le_away.fit(all_teams)

df['home_team_encoded'] = le_home.transform(df['home_team'])
df['away_team_encoded'] = le_away.transform(df['away_team'])

# Define features and target
feature_cols = ['home_pregame_elo', 'away_pregame_elo', 'home_team_encoded', 'away_team_encoded']
X = df[feature_cols]
y = df['Margin']

# Time-series split: Train on 2022-2023, Test on 2024
train_mask = df['season'].isin([2022, 2023])
test_mask = df['season'] == 2024

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train Gradient Boosting model
print("\nTraining Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"\n{'='*50}")
print(f"Mean Absolute Error (MAE) on 2024 data: {mae:.2f} points")
print(f"{'='*50}")

# Feature Importance
print("\nFeature Importance:")
print("-" * 40)
importance = model.feature_importances_
feature_importance = list(zip(feature_cols, importance))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for feature, imp in feature_importance:
    bar = '█' * int(imp * 50)
    print(f"{feature:25} {imp:.4f} {bar}")

# Save the model
with open('cfb_predictor.model', 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved to 'cfb_predictor.model'")

# Also save the label encoders for future predictions
with open('team_encoders.pkl', 'wb') as f:
    pickle.dump({'home': le_home, 'away': le_away}, f)
print("Team encoders saved to 'team_encoders.pkl'")
