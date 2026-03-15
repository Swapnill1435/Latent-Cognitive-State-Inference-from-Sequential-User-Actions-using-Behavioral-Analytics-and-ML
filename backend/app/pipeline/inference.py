"""Inference orchestrator — runs all 3 models and produces ensemble predictions.

Includes attention map extraction from Transformer for explainability.
"""

import os
import time
import numpy as np
from typing import Any, Dict, List, Optional

from app.config import config
from app.models.hmm_model import CognitiveHMM
from app.models.lstm_model import CognitiveLSTM
from app.models.transformer_model import CognitiveTransformerInference
from app.models.ar_arch_model import CognitiveARARCH
from app.models.rf_gb_model import CognitiveTreeEnsemble


class InferenceOrchestrator:
    """Manages all ML models and produces ensemble cognitive state predictions."""

    def __init__(self):
        self.hmm = CognitiveHMM()
        self.lstm = CognitiveLSTM()
        self.transformer = CognitiveTransformerInference()
        self.ar_arch = CognitiveARARCH()
        self.rf = CognitiveTreeEnsemble(model_type="random_forest")
        self.gb = CognitiveTreeEnsemble(model_type="gradient_boosting")

        # Ensemble weights (can be tuned)
        # Using GB over RF for the ensemble contribution
        self.weights = {
            "hmm": 0.10, 
            "lstm": 0.25, 
            "transformer": 0.35,
            "ar_arch": 0.10,
            "gb": 0.20
        }
        self.state_names = config.model.cognitive_states
        self._models_loaded = False
        self._last_attention_maps = None

    def load_models(self):
        """Load pre-trained model weights if available."""
        models_dir = config.models_dir
        os.makedirs(models_dir, exist_ok=True)

        hmm_path = os.path.join(models_dir, "hmm.pkl")
        lstm_path = os.path.join(models_dir, "lstm.pt")
        transformer_path = os.path.join(models_dir, "transformer.pt")
        ar_arch_path = os.path.join(models_dir, "ar_arch.pkl")
        rf_path = os.path.join(models_dir, "random_forest.pkl")
        gb_path = os.path.join(models_dir, "gradient_boosting.pkl")

        if os.path.exists(hmm_path):
            try:
                self.hmm.load(hmm_path)
            except Exception as e:
                print(f"Warning: Could not load HMM model: {e}")

        self.lstm.build()
        if os.path.exists(lstm_path):
            try:
                self.lstm.load(lstm_path)
            except Exception as e:
                print(f"Warning: Could not load LSTM model: {e}")

        self.transformer.build()
        if os.path.exists(transformer_path):
            try:
                self.transformer.load(transformer_path)
            except Exception as e:
                print(f"Warning: Could not load Transformer model: {e}")

        if os.path.exists(ar_arch_path):
            try:
                self.ar_arch.load(ar_arch_path)
            except Exception as e:
                print(f"Warning: Could not load AR-ARCH model: {e}")

        if os.path.exists(rf_path):
            try:
                self.rf.load(rf_path)
            except Exception as e:
                print(f"Warning: Could not load Random Forest model: {e}")

        if os.path.exists(gb_path):
            try:
                self.gb.load(gb_path)
            except Exception as e:
                print(f"Warning: Could not load Gradient Boosting model: {e}")

        self._models_loaded = True

    def predict(self, feature_vector: np.ndarray, feature_sequence: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run ensemble inference with attention map extraction.

        Args:
            feature_vector: (n_features,) single feature vector for HMM.
            feature_sequence: (seq_len, n_features) for LSTM/Transformer.

        Returns:
            Dict with ensemble probabilities, individual model outputs, attention maps,
            predicted state, and timestamp.
        """
        if not self._models_loaded:
            self.load_models()

        if feature_sequence is None:
            feature_sequence = feature_vector.reshape(1, -1)

        # Run each model
        hmm_probs = self.hmm.predict_latest(feature_vector.reshape(1, -1))
        lstm_probs = self.lstm.predict_latest(feature_sequence)
        ar_arch_probs = self.ar_arch.predict_latest(feature_sequence)
        gb_probs = self.gb.predict_latest(feature_vector.reshape(1, -1))

        # Transformer with attention maps
        transformer_probs, attention_maps = self.transformer.predict_with_attention(feature_sequence)
        transformer_probs_dict = {name: float(p) for name, p in zip(self.state_names, transformer_probs)}
        self._last_attention_maps = attention_maps

        # Ensemble: weighted average
        num_states = len(self.state_names)
        uniform_p = 1.0 / num_states
        ensemble = {}
        for state in self.state_names:
            ensemble[state] = (
                self.weights["hmm"] * hmm_probs.get(state, uniform_p)
                + self.weights["lstm"] * lstm_probs.get(state, uniform_p)
                + self.weights["transformer"] * transformer_probs_dict.get(state, uniform_p)
                + self.weights["ar_arch"] * ar_arch_probs.get(state, uniform_p)
                + self.weights["gb"] * gb_probs.get(state, uniform_p)
            )

        # Normalize
        total = sum(ensemble.values())
        if total > 0:
            ensemble = {k: v / total for k, v in ensemble.items()}

        predicted_state = max(ensemble, key=ensemble.get)

        # Serialize attention maps (averaged across heads per layer)
        attention_summary = None
        if attention_maps:
            attention_summary = []
            for layer_idx, attn in enumerate(attention_maps):
                # attn shape: (nhead, seq_len, seq_len) — average across heads
                avg_attn = attn.mean(axis=0).tolist()  # (seq_len, seq_len)
                attention_summary.append({
                    "layer": layer_idx,
                    "attention": avg_attn,
                })

        return {
            "timestamp": time.time(),
            "predicted_state": predicted_state,
            "confidence": ensemble[predicted_state],
            "probabilities": ensemble,
            "model_outputs": {
                "hmm": hmm_probs,
                "lstm": lstm_probs,
                "transformer": transformer_probs_dict,
                "ar_arch": ar_arch_probs,
                "gb": gb_probs,
            },
            "attention_maps": attention_summary,
        }

    def get_last_attention_maps(self):
        """Return raw attention maps from last prediction."""
        return self._last_attention_maps


# Global singleton
orchestrator = InferenceOrchestrator()
