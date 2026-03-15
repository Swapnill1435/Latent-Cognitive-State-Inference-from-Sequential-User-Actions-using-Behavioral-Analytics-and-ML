"""Differential privacy implementation for behavioral data protection."""

import numpy as np
from typing import Dict

from app.config import config


class DifferentialPrivacy:
    """ε-Differential privacy via Laplace noise injection on behavioral features."""

    def __init__(self, epsilon: float = None):
        self.epsilon = epsilon or config.privacy.epsilon

    def add_noise(self, features: Dict[str, float], sensitivity: float = 1.0) -> Dict[str, float]:
        """Add Laplace noise to features for differential privacy.

        Args:
            features: Dict of feature name -> value.
            sensitivity: Maximum difference a single user's data can cause.

        Returns:
            Noised features dict.
        """
        scale = sensitivity / self.epsilon
        noised = {}
        for name, value in features.items():
            noise = np.random.laplace(0, scale)
            noised[name] = value + noise
        return noised

    def add_noise_to_array(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Add Laplace noise to a numpy array."""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, size=data.shape)
        return data + noise

    def anonymize_coordinates(self, x: float, y: float, grid_size: int = 50) -> tuple:
        """Snap coordinates to a grid for anonymization."""
        return (round(x / grid_size) * grid_size, round(y / grid_size) * grid_size)

    def anonymize_session(self, session_data: dict) -> dict:
        """Anonymize a full session's data for export."""
        anonymized = {
            "session_id": "anon_" + session_data.get("session_id", "")[-8:],
            "user_id": "anonymous",
            "event_count": len(session_data.get("events", [])),
            "predictions": session_data.get("predictions", []),
        }

        # Anonymize event coordinates if configured
        if config.privacy.anonymize_coordinates:
            anon_events = []
            for e in session_data.get("events", []):
                ae = dict(e)
                if "x" in ae and "y" in ae:
                    ae["x"], ae["y"] = self.anonymize_coordinates(ae["x"], ae["y"])
                anon_events.append(ae)
            anonymized["events"] = anon_events
        else:
            anonymized["events"] = session_data.get("events", [])

        return anonymized


# Global singleton
dp = DifferentialPrivacy()
