"""LSTM model for cognitive state inference from behavioral sequences.

Architecture: Input -> Linear projection -> Stacked LSTM -> Dropout -> Dense -> Softmax
Outputs probability distribution over 5 cognitive states.
"""

import os
import numpy as np
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import config


class LSTMClassifier(nn.Module):
    """Stacked LSTM for sequence classification of cognitive states."""

    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = None,
        num_classes: int = None,
        dropout: float = None,
    ):
        super().__init__()
        self.input_size = input_size or config.features.num_features
        self.hidden_size = hidden_size or config.model.lstm_hidden_size
        self.num_layers = num_layers or config.model.lstm_num_layers
        self.num_classes = num_classes or config.model.num_states
        self.dropout_rate = dropout or config.model.lstm_dropout

        # Input projection
        self.input_proj = nn.Linear(self.input_size, self.hidden_size)

        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size) tensor.

        Returns:
            (batch, num_classes) log-probabilities.
        """
        # Project input
        x = F.relu(self.input_proj(x))  # (batch, seq, hidden)

        # LSTM
        output, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = output[:, -1, :]  # (batch, hidden)
        last_hidden = self.dropout(last_hidden)

        logits = self.classifier(last_hidden)  # (batch, num_classes)
        return logits


class CognitiveLSTM:
    """Wrapper for LSTM-based cognitive state inference."""

    def __init__(self):
        self.model: Optional[LSTMClassifier] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_names = config.model.cognitive_states

    def build(self):
        self.model = LSTMClassifier().to(self.device)

    def predict_proba(self, feature_sequence: np.ndarray) -> np.ndarray:
        """Predict cognitive state probabilities from a feature sequence.

        Args:
            feature_sequence: (seq_len, n_features) or (1, seq_len, n_features).

        Returns:
            (num_classes,) probability vector.
        """
        if self.model is None:
            self.build()

        self.model.eval()

        if feature_sequence.ndim == 1:
            feature_sequence = feature_sequence.reshape(1, -1)
        if feature_sequence.ndim == 2:
            feature_sequence = feature_sequence[np.newaxis, :]  # (1, seq, features)

        x = torch.FloatTensor(feature_sequence).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

        return probs.cpu().numpy()[0]

    def predict_latest(self, feature_sequence: np.ndarray) -> Dict[str, float]:
        """Get cognitive state probabilities as a named dict."""
        probs = self.predict_proba(feature_sequence)
        return {name: float(p) for name, p in zip(self.state_names, probs)}

    def save(self, path: str):
        if self.model is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        if self.model is None:
            self.build()
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
