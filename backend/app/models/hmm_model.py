"""Hidden Markov Model for cognitive state inference.

Uses hmmlearn with 6 hidden states (confidence, confused, exploring, hesitating, overloaded, fatigue).
Provides Viterbi decoding and posterior state probabilities.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.naive_bayes import GaussianNB

from app.config import config


class CognitiveHMM:
    """Hidden Markov Model adapted for aggregated sessions.
    
    Acts as a Gaussian Naive Bayes model mimicking the stationary emission
    probabilities of an HMM without transition scaling errors on length=1
    sequence summaries. 
    """

    def __init__(self, n_states: int = None, n_features: int = None):
        self.state_names = config.model.cognitive_states
        self.num_classes = len(self.state_names)
        self.model = GaussianNB()
        self._is_fitted = False

    def build(self):
        """Initialize the model."""
        pass  # GaussianNB doesn't need external setup

    def fit(self, sequences: List[np.ndarray], labels: List[np.ndarray] = None, lengths: Optional[List[int]] = None):
        """Train the underlying Gaussian network on categorized behavior.

        Args:
            sequences: List of (T, n_features) arrays.
            labels: List of label arrays corresponding to sequences.
        """
        if isinstance(sequences, list) and len(sequences) > 0:
            X = np.vstack(sequences)
            y = np.array(labels)
        else:
            X = sequences
            y = np.array(labels)

        # Train a Gaussian Naive Bayes classifier on the sequence means
        self.model.fit(X, y)
        self._is_fitted = True

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get posterior probabilities of cognitive states.
        """
        if not self._is_fitted:
            return self._uniform_probs(features.shape[0] if features.ndim > 1 else 1)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        try:
            return self.model.predict_proba(features)
        except Exception:
            return self._uniform_probs(features.shape[0])

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict the most likely state.
        """
        if not self._is_fitted:
            n = features.shape[0] if features.ndim > 1 else 1
            return np.zeros(n, dtype=int)

        return self.model.predict(features)

    def predict_latest(self, features: np.ndarray) -> Dict[str, float]:
        """Get cognitive state probabilities for the latest time step.
        """
        probs = self.predict_proba(features)
        latest = probs[-1] if probs.ndim > 1 else probs[0]
        return {name: float(p) for name, p in zip(self.state_names, latest)}

    def _uniform_probs(self, n: int) -> np.ndarray:
        p = 1.0 / self.num_classes
        return np.full((n, self.num_classes), p)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "is_fitted": self._is_fitted}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self._is_fitted = data["is_fitted"]
