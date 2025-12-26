"""
V21 LSTM Training - Team Sequence Modeling.

LSTM processes each team's last N games as a sequence rather than
aggregating into rolling averages. This captures:
- Momentum patterns
- Recent form trajectory
- Recovery from losses

Key insight: Rolling averages lose temporal ordering information.
An LSTM can learn patterns like "won 3, lost 1, won 2" vs "won 5, lost 1".

Usage:
    python train_v21_lstm.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# Sequence length (number of past games to consider)
SEQ_LENGTH = 5

# Features to include in sequence (per-game stats)
SEQ_FEATURES = [
    'margin',  # Points won/lost by
    'covered',  # 1 if covered spread, 0 otherwise
    'total_points',  # Total points scored in game
    'spread',  # What the spread was
]

# Static features (team-level, not sequence)
STATIC_FEATURES = [
    'home_pregame_elo', 'away_pregame_elo', 'elo_diff',
    'home_team_hfa', 'hfa_diff',
    'rest_diff', 'home_rest_days', 'away_rest_days',
    'vegas_spread', 'line_movement', 'spread_open',
    'large_favorite', 'large_underdog', 'close_game',
    'home_qb_status', 'away_qb_status', 'qb_advantage',
    'wind_speed', 'temperature', 'is_dome',
]


class CFBSequenceDataset(Dataset):
    """PyTorch dataset for team sequence data."""

    def __init__(self, home_seqs, away_seqs, static_features, targets):
        self.home_seqs = torch.FloatTensor(home_seqs)
        self.away_seqs = torch.FloatTensor(away_seqs)
        self.static_features = torch.FloatTensor(static_features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.home_seqs[idx],
            self.away_seqs[idx],
            self.static_features[idx],
            self.targets[idx]
        )


class TeamLSTM(nn.Module):
    """LSTM model for team sequence + static features."""

    def __init__(self, seq_input_dim, static_input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()

        # Separate LSTMs for home and away team sequences
        self.home_lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.away_lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Combine LSTM outputs with static features
        combined_dim = hidden_dim * 2 + static_input_dim

        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, home_seq, away_seq, static_feat):
        # Process sequences through LSTMs
        _, (home_hidden, _) = self.home_lstm(home_seq)
        _, (away_hidden, _) = self.away_lstm(away_seq)

        # Take last layer's hidden state
        home_out = home_hidden[-1]  # [batch, hidden_dim]
        away_out = away_hidden[-1]  # [batch, hidden_dim]

        # Combine with static features
        combined = torch.cat([home_out, away_out, static_feat], dim=1)

        # Final prediction
        return self.fc(combined).squeeze(-1)


class V21LSTMModel:
    """LSTM model wrapper for team sequence modeling."""

    def __init__(self, hidden_dim=64, num_layers=2, dropout=0.2):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.scaler = StandardScaler()
        self.config = None

    def build_team_history(self, df):
        """Build historical game data for each team."""
        print("Building team history...")

        # Sort by time
        df = df.sort_values(['season', 'week']).reset_index(drop=True)

        # Build history for each team
        team_history = {}

        for idx, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']

            # Initialize if needed
            if home not in team_history:
                team_history[home] = []
            if away not in team_history:
                team_history[away] = []

            # Get margin
            margin = row.get('Margin', 0) if pd.notna(row.get('Margin')) else 0
            spread = row.get('vegas_spread', 0) if pd.notna(row.get('vegas_spread')) else 0
            covered = 1 if margin > -spread else 0

            home_points = row.get('home_points', 0) if pd.notna(row.get('home_points')) else 28
            away_points = row.get('away_points', 0) if pd.notna(row.get('away_points')) else 24

            # Store game data for home team
            team_history[home].append({
                'season': row['season'],
                'week': row['week'],
                'is_home': 1,
                'margin': margin,  # Positive = won
                'covered': covered,
                'total_points': home_points + away_points,
                'spread': spread,
                'opponent_elo': row.get('away_pregame_elo', 1500),
            })

            # Store for away team (flip margin)
            team_history[away].append({
                'season': row['season'],
                'week': row['week'],
                'is_home': 0,
                'margin': -margin,  # From away perspective
                'covered': 1 - covered,  # Flip coverage
                'total_points': home_points + away_points,
                'spread': -spread,  # Flip spread
                'opponent_elo': row.get('home_pregame_elo', 1500),
            })

        return team_history

    def get_team_sequence(self, team, season, week, team_history, seq_length=SEQ_LENGTH):
        """Get last N games for a team before this game."""
        if team not in team_history:
            return np.zeros((seq_length, len(SEQ_FEATURES)))

        # Filter to games before this one
        past_games = [
            g for g in team_history[team]
            if (g['season'] < season) or (g['season'] == season and g['week'] < week)
        ]

        # Take last seq_length games
        past_games = past_games[-seq_length:]

        # Build sequence array
        seq = np.zeros((seq_length, len(SEQ_FEATURES)))

        for i, game in enumerate(past_games):
            start_idx = seq_length - len(past_games) + i
            for j, feat in enumerate(SEQ_FEATURES):
                seq[start_idx, j] = game.get(feat, 0)

        return seq

    def prepare_data(self, df):
        """Prepare sequences and static features for training."""
        print("Preparing data for LSTM...")

        # Filter to games with Vegas spread and results
        df_valid = df[(df['vegas_spread'].notna()) & (df['Margin'].notna())].copy()

        # Build team histories
        team_history = self.build_team_history(df_valid)

        # Prepare arrays
        home_seqs = []
        away_seqs = []
        static_feats = []
        targets = []

        # Get available static features
        available_static = [f for f in STATIC_FEATURES if f in df_valid.columns]

        for idx, row in df_valid.iterrows():
            home = row['home_team']
            away = row['away_team']
            season = row['season']
            week = row['week']

            # Get sequences
            home_seq = self.get_team_sequence(home, season, week, team_history)
            away_seq = self.get_team_sequence(away, season, week, team_history)

            home_seqs.append(home_seq)
            away_seqs.append(away_seq)

            # Get static features
            static = [row.get(f, 0) for f in available_static]
            static = [0 if pd.isna(x) else x for x in static]
            static_feats.append(static)

            # Target
            targets.append(row['Margin'])

        # Convert to arrays
        home_seqs = np.array(home_seqs)
        away_seqs = np.array(away_seqs)
        static_feats = np.array(static_feats)
        targets = np.array(targets)

        print(f"  Prepared {len(targets)} samples")
        print(f"  Sequence shape: {home_seqs.shape}")
        print(f"  Static features: {len(available_static)}")

        return home_seqs, away_seqs, static_feats, targets, available_static

    def train(self, df, epochs=50, batch_size=64, lr=0.001):
        """Train the LSTM model."""
        print("=" * 70)
        print("V21 LSTM MODEL TRAINING")
        print("=" * 70)

        # Prepare data
        home_seqs, away_seqs, static_feats, targets, static_feat_names = self.prepare_data(df)

        # Scale static features
        static_feats_scaled = self.scaler.fit_transform(static_feats)

        # Train/test split (time-based)
        split_idx = int(len(targets) * 0.8)

        train_dataset = CFBSequenceDataset(
            home_seqs[:split_idx],
            away_seqs[:split_idx],
            static_feats_scaled[:split_idx],
            targets[:split_idx]
        )

        test_dataset = CFBSequenceDataset(
            home_seqs[split_idx:],
            away_seqs[split_idx:],
            static_feats_scaled[split_idx:],
            targets[split_idx:]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        seq_input_dim = len(SEQ_FEATURES)
        static_input_dim = len(static_feat_names)

        self.model = TeamLSTM(
            seq_input_dim=seq_input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        print(f"\nTraining for {epochs} epochs...")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")

        best_test_loss = float('inf')
        best_epoch = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for home_seq, away_seq, static, target in train_loader:
                optimizer.zero_grad()
                pred = self.model(home_seq, away_seq, static)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Evaluation
            self.model.eval()
            test_loss = 0
            test_preds = []
            test_targets = []

            with torch.no_grad():
                for home_seq, away_seq, static, target in test_loader:
                    pred = self.model(home_seq, away_seq, static)
                    loss = criterion(pred, target)
                    test_loss += loss.item()
                    test_preds.extend(pred.numpy())
                    test_targets.extend(target.numpy())

            test_loss /= len(test_loader)
            test_mae = mean_absolute_error(test_targets, test_preds)

            scheduler.step(test_loss)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test MAE={test_mae:.2f}")

        print(f"\nBest epoch: {best_epoch+1} with test loss: {best_test_loss:.4f}")

        # Final evaluation
        self.model.eval()
        test_preds = []
        test_targets = []

        with torch.no_grad():
            for home_seq, away_seq, static, target in test_loader:
                pred = self.model(home_seq, away_seq, static)
                test_preds.extend(pred.numpy())
                test_targets.extend(target.numpy())

        test_mae = mean_absolute_error(test_targets, test_preds)
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))

        print("\nFinal Model Performance:")
        print(f"  Test MAE: {test_mae:.2f}")
        print(f"  Test RMSE: {test_rmse:.2f}")

        # Save config
        self.config = {
            'version': 'V21_LSTM',
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(targets),
            'seq_length': SEQ_LENGTH,
            'seq_features': SEQ_FEATURES,
            'static_features': static_feat_names,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'epochs': epochs,
        }

        return self

    def save(self, path_prefix='cfb_v21_lstm'):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': self.config,
        }, f'{path_prefix}.pt')
        print(f"Saved model to {path_prefix}.pt")

    @classmethod
    def load(cls, path_prefix='cfb_v21_lstm'):
        """Load model from file."""
        checkpoint = torch.load(f'{path_prefix}.pt')

        model = cls()
        model.scaler = checkpoint['scaler']
        model.config = checkpoint['config']

        # Rebuild model architecture
        model.model = TeamLSTM(
            seq_input_dim=len(checkpoint['config']['seq_features']),
            static_input_dim=len(checkpoint['config']['static_features']),
            hidden_dim=checkpoint['config']['hidden_dim'],
            num_layers=checkpoint['config']['num_layers'],
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])

        return model


def main():
    print("=" * 70)
    print("V21 LSTM TRAINING")
    print("=" * 70)

    # Load data
    print("\nLoading training data...")
    df = pd.read_csv('cfb_data_safe.csv')
    print(f"Loaded {len(df)} games")

    # Train model
    model = V21LSTMModel(hidden_dim=64, num_layers=2, dropout=0.2)
    model.train(df, epochs=50, batch_size=64, lr=0.001)

    # Save
    model.save('cfb_v21_lstm')

    print("\n" + "=" * 70)
    print("V21 LSTM TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
