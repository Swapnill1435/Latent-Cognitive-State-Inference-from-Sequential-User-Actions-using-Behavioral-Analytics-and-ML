"""Bayesian AR-ARCH Model for cognitive state inference.

Uses an Autoregressive Conditional Heteroskedasticity (AR-ARCH) structure
to capture rapid fluctuations in cognitive capacity based on behavioral features.
"""

import os
import pickle
import numpy as np
import warnings
from typing import Dict, List, Optional
from arch import arch_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.config import config


class CognitiveARARCH:
    """AR-ARCH model predicting cognitive states.
    
    Fits an AR-ARCH model to the primary temporal feature (e.g., mean reaction time)
    to extract conditional volatility, which is then used alongside other features
    to classify the cognitive state.
    """

    def __init__(self):
        self.state_names = config.model.cognitive_states
        self.n_states = config.model.num_states
        self.classifier = LogisticRegression(max_iter=1000, multi_class='multinomial')
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _extract_volatility(self, sequence: np.ndarray) -> np.ndarray:
        """Extracts conditional volatility from the time series using AR-ARCH."""
        # Using the first feature (typically mean_rt) as the proxy for cognitive load
        ts = sequence[:, 0]
        if len(ts) < 10:
            # Not enough data for AR-ARCH, return zeros
            return np.zeros_like(ts)
            
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                # Basic AR(1)-ARCH(1) model
                am = arch_model(ts, vol='ARCH', p=1, q=0, rescale=False)
                res = am.fit(disp='off', show_warning=False)
                volatility = res.conditional_volatility
                return volatility
            except Exception:
                return np.zeros_like(ts)

    def fit(self, sequences: List[np.ndarray], labels: List[int]):
        """Train the AR-ARCH classifier on featured sequences.

        Args:
            sequences: List of (T, n_features) arrays
            labels: List of integer state labels corresponding to each sequence.
        """
        X_train = []
        y_train = []

        for seq, label in zip(sequences, labels):
            volatility = self._extract_volatility(seq)
            # Use the mean volatility and the mean of the original features for the final state
            mean_vol = np.mean(volatility) if len(volatility) > 0 else 0
            mean_feats = np.mean(seq, axis=0)
            
            combined_features = np.append(mean_feats, mean_vol)
            X_train.append(combined_features)
            y_train.append(label)

        if X_train:
            X_scaled = self.scaler.fit_transform(X_train)
            self.classifier.fit(X_scaled, y_train)
            self._is_fitted = True

    def predict_proba(self, feature_sequence: np.ndarray) -> np.ndarray:
        """Predict cognitive state probabilities from a feature sequence."""
        if not self._is_fitted:
            return self._uniform_probs()

        if feature_sequence.ndim == 1:
            feature_sequence = feature_sequence.reshape(1, -1)

        volatility = self._extract_volatility(feature_sequence)
        mean_vol = np.mean(volatility) if len(volatility) > 0 else 0
        mean_feats = np.mean(feature_sequence, axis=0)
        
        combined_features = np.append(mean_feats, mean_vol).reshape(1, -1)
        
        try:
            scaled_features = self.scaler.transform(combined_features)
            return self.classifier.predict_proba(scaled_features)[0]
        except Exception:
            return self._uniform_probs()

    def predict_latest(self, feature_sequence: np.ndarray) -> Dict[str, float]:
        """Get cognitive state probabilities as a named dict."""
        probs = self.predict_proba(feature_sequence)
        return {name: float(p) for name, p in zip(self.state_names, probs)}

    def _uniform_probs(self) -> np.ndarray:
        p = 1.0 / self.n_states
        return np.full(self.n_states, p)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"classifier": self.classifier, "scaler": self.scaler, "is_fitted": self._is_fitted}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.scaler = data.get("scaler", StandardScaler())
        self._is_fitted = data["is_fitted"]
