"""Random Forest and Gradient Boosting models for cognitive state inference.

Utilizes tree-based ensemble methods for rapid classification of aggregated behavioral features.
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from app.config import config


class CognitiveTreeEnsemble:
    """Wrapper for Random Forest and Gradient Boosting inference."""

    def __init__(self, model_type: str = "random_forest"):
        self.state_names = config.model.cognitive_states
        self.n_states = config.model.num_states
        self.model_type = model_type
        
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the ensemble model."""
        self.model.fit(X, y)
        self._is_fitted = True

    def predict_proba(self, feature_vector: np.ndarray) -> np.ndarray:
        """Predict probabilities for a single session vector.
        
        feature_vector: (n_features,) or (1, n_features)
        """
        if not self._is_fitted:
            return self._uniform_probs()
            
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
            
        # If sequence is provided instead of aggregated vector, take the mean
        if feature_vector.shape[0] > 1:
            feature_vector = np.mean(feature_vector, axis=0).reshape(1, -1)
            
        try:
            return self.model.predict_proba(feature_vector)[0]
        except Exception:
            return self._uniform_probs()

    def predict_latest(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Get cognitive state probabilities as a named dict."""
        probs = self.predict_proba(feature_vector)
        # Handle cases where model might not have seen all classes during training
        if len(probs) < self.n_states:
            extended_probs = np.zeros(self.n_states)
            for i, cls in enumerate(self.model.classes_):
                if cls < self.n_states:
                    extended_probs[cls] = probs[i]
            probs = extended_probs
            
        return {name: float(p) for name, p in zip(self.state_names, probs)}

    def _uniform_probs(self) -> np.ndarray:
        p = 1.0 / self.n_states
        return np.full(self.n_states, p)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "is_fitted": self._is_fitted}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self._is_fitted = data["is_fitted"]
