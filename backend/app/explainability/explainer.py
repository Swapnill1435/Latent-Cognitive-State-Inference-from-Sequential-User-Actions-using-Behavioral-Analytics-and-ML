"""SHAP, LIME, and Transformer Attention Map explainability for cognitive state predictions.

Provides:
- SHAP kernel explainer for feature importance (model-agnostic)
- LIME tabular explainer for local explanations
- Transformer attention map visualization data
- Heuristic domain-knowledge fallback when SHAP/LIME unavailable
"""

import numpy as np
from typing import Any, Dict, List, Optional, Callable

from app.pipeline.feature_engine import FEATURE_NAMES
from app.config import config

# Try to import SHAP and LIME
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


class CognitiveExplainer:
    """Provides feature importance explanations using SHAP, LIME, attention maps, and heuristics."""

    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self.state_names = config.model.cognitive_states
        self._shap_explainer = None
        self._lime_explainer = None
        self._background_data = None

    def initialize_shap(self, predict_fn: Callable, background_data: np.ndarray):
        """Initialize SHAP KernelExplainer with a prediction function and background data.

        Args:
            predict_fn: Function (n_samples, n_features) -> (n_samples, n_classes) probabilities.
            background_data: (n_background, n_features) samples for SHAP background.
        """
        if not HAS_SHAP:
            print("Warning: SHAP not installed. Using heuristic explainer.")
            return
        self._background_data = background_data
        # Use a small subsample for efficiency
        bg = shap.sample(background_data, min(50, len(background_data)))
        self._shap_explainer = shap.KernelExplainer(predict_fn, bg)

    def initialize_lime(self, training_data: np.ndarray):
        """Initialize LIME explainer with training data statistics.

        Args:
            training_data: (n_samples, n_features) for distribution estimation.
        """
        if not HAS_LIME:
            print("Warning: LIME not installed. Using heuristic explainer.")
            return
        self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=self.feature_names,
            class_names=self.state_names,
            mode="classification",
        )

    def explain_shap(self, feature_vector: np.ndarray) -> Optional[Dict[str, Any]]:
        """Compute SHAP values for a single feature vector.

        Returns dict with per-feature SHAP values for each cognitive state.
        """
        if self._shap_explainer is None:
            return None

        fv = feature_vector.reshape(1, -1)
        try:
            shap_values = self._shap_explainer.shap_values(fv, nsamples=100)
            # shap_values is a list of (1, n_features) arrays, one per class
            result = {}
            for class_idx, state_name in enumerate(self.state_names):
                if class_idx < len(shap_values):
                    values = shap_values[class_idx][0]  # (n_features,)
                    result[state_name] = {
                        name: float(val) for name, val in zip(self.feature_names, values)
                    }
            return result
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return None

    def explain_lime(self, feature_vector: np.ndarray, predict_fn: Callable) -> Optional[Dict[str, Any]]:
        """Compute LIME explanation for a single feature vector.

        Args:
            feature_vector: (n_features,) vector to explain.
            predict_fn: Prediction function (n_samples, n_features) -> (n_samples, n_classes).

        Returns dict with per-feature importance and local model explanation.
        """
        if self._lime_explainer is None:
            return None

        try:
            exp = self._lime_explainer.explain_instance(
                feature_vector, predict_fn, num_features=len(self.feature_names), top_labels=len(self.state_names)
            )
            result = {}
            for class_idx in exp.available_labels():
                state_name = self.state_names[class_idx] if class_idx < len(self.state_names) else f"class_{class_idx}"
                feature_weights = exp.as_list(label=class_idx)
                result[state_name] = {
                    "features": [{"feature": fw[0], "weight": float(fw[1])} for fw in feature_weights],
                    "intercept": float(exp.intercept[class_idx]) if class_idx in exp.intercept else 0.0,
                    "score": float(exp.score) if hasattr(exp, 'score') else 0.0,
                }
            return result
        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return None

    def explain_prediction(
        self,
        features: Dict[str, float],
        prediction: Dict[str, Any],
        predict_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive explanation for a prediction.

        Uses SHAP and LIME when available, falls back to heuristic domain knowledge.

        Args:
            features: Feature dict (name -> value).
            prediction: Prediction dict from inference orchestrator.
            predict_fn: Optional model prediction function for SHAP/LIME.
        """
        predicted_state = prediction.get("predicted_state", "unknown")
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])

        # Try SHAP
        shap_result = None
        if self._shap_explainer is not None:
            shap_result = self.explain_shap(feature_vector)

        # Try LIME
        lime_result = None
        if self._lime_explainer is not None and predict_fn is not None:
            lime_result = self.explain_lime(feature_vector, predict_fn)

        # Heuristic importance (always available)
        heuristic = self._heuristic_importance(features, predicted_state)

        # Attention maps from prediction
        attention_maps = prediction.get("attention_maps", None)

        # Combine into feature importance: prefer SHAP > LIME > heuristic
        if shap_result and predicted_state in shap_result:
            importance_source = "shap"
            importance_values = shap_result[predicted_state]
        elif lime_result and predicted_state in lime_result:
            importance_source = "lime"
            importance_values = {
                fw["feature"]: fw["weight"]
                for fw in lime_result[predicted_state].get("features", [])
            }
        else:
            importance_source = "heuristic"
            importance_values = heuristic

        # Sort by absolute importance
        sorted_features = sorted(importance_values.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "predicted_state": predicted_state,
            "confidence": prediction.get("confidence", 0),
            "importance_source": importance_source,
            "feature_importance": [
                {"feature": name, "importance": float(score), "value": features.get(name, 0)}
                for name, score in sorted_features
            ],
            "shap_available": shap_result is not None,
            "lime_available": lime_result is not None,
            "shap_values": shap_result,
            "lime_values": lime_result,
            "attention_maps": attention_maps,
            "explanation": self._generate_text_explanation(predicted_state, sorted_features[:5], features),
        }

    def _heuristic_importance(self, features: Dict[str, float], predicted_state: str) -> Dict[str, float]:
        """Compute heuristic feature importance based on cognitive state indicators (domain knowledge)."""
        state_indicators = {
            "confused": {
                "loop_count": 0.9, "backtrack_count": 0.85, "navigation_entropy": 0.8,
                "direction_changes_x": 0.7, "direction_changes_y": 0.7,
                "mean_reaction_time": 0.6, "answer_change_count": 0.75,
                "std_reaction_time": 0.5, "mpp_burstiness": 0.6,
            },
            "confidence": {
                "mean_velocity": -0.3, "trajectory_curvature": -0.5, "action_rate": 0.7,
                "std_reaction_time": -0.6, "midline_deviation": -0.5,
                "unique_action_ratio": -0.4, "mpp_regularity": 0.8,
                "mpp_intensity_rate": 0.5,
            },
            "exploring": {
                "navigation_entropy": 0.85, "unique_action_ratio": 0.8,
                "action_rate": 0.6, "backtrack_count": 0.3,
                "mean_velocity": 0.5, "mpp_burstiness": 0.4,
            },
            "hesitating": {
                "mean_hesitation_time": 0.9, "max_pause_duration": 0.8,
                "mean_pause_duration": 0.7, "answer_change_count": 0.6,
                "mean_reaction_time": 0.5, "std_reaction_time": 0.65,
                "mpp_log_rt_mean": 0.7,
            },
            "overloaded": {
                "action_entropy": 0.85, "std_reaction_time": 0.8,
                "direction_changes_x": 0.7, "direction_changes_y": 0.7,
                "trajectory_curvature": 0.75, "max_pause_duration": 0.6,
                "loop_count": 0.5, "mean_velocity": 0.4,
                "mpp_burstiness": 0.7,
            },
            "fatigue": {
                "mean_reaction_time": 0.85, "mpp_log_rt_mean": 0.9,
                "mpp_regularity": -0.7, "mpp_intensity_rate": -0.8,
                "action_rate": -0.75, "max_pause_duration": 0.7,
                "mean_velocity": -0.6, "std_reaction_time": 0.5,
                "trajectory_curvature": -0.4,
            },
        }

        indicators = state_indicators.get(predicted_state, {})
        importance = {}
        for feature_name in self.feature_names:
            value = features.get(feature_name, 0)
            weight = indicators.get(feature_name, 0.1)
            importance[feature_name] = weight * (1 + min(abs(value), 10) / 10)
        return importance

    def _generate_text_explanation(
        self, predicted_state: str, top_features: List, features: Dict[str, float]
    ) -> str:
        """Generate a human-readable text explanation."""
        state_descriptions = {
            "confused": "The user appears confused based on",
            "confidence": "The user shows confident engagement, indicated by",
            "exploring": "The user is in an exploratory state, evidenced by",
            "hesitating": "The user is hesitating, suggested by",
            "overloaded": "The user may be experiencing cognitive overload, shown by",
            "fatigue": "The user shows signs of fatigue, indicated by",
        }

        desc = state_descriptions.get(predicted_state, f"Predicted state: {predicted_state}, based on")

        if not top_features:
            return f"{desc} overall behavioral patterns."

        feature_descriptions = []
        for name, score in top_features[:3]:
            value = features.get(name, 0)
            readable_name = name.replace("_", " ")
            feature_descriptions.append(f"{readable_name} ({value:.2f})")

        return f"{desc} {', '.join(feature_descriptions)}."


# Global singleton
explainer = CognitiveExplainer()
