"""Ground truth label generator for supervised learning.

Generates labels using:
- Self-reported cognitive states (from frontend form)
- Task difficulty levels
- Expert behavioral annotation
- NASA-TLX inspired workload rating
"""

import time
from typing import Any, Dict, List, Optional

from app.config import config


class LabelGenerator:
    """Generates and manages ground-truth cognitive state labels."""

    def __init__(self):
        self.state_names = config.model.cognitive_states
        self.nasa_tlx_dimensions = [
            "mental_demand", "physical_demand", "temporal_demand",
            "performance", "effort", "frustration"
        ]

    def from_self_report(self, state: str, confidence: float = 1.0) -> Dict[str, Any]:
        """Create label from user self-report."""
        return {
            "state": state if state in self.state_names else "unknown",
            "confidence": confidence,
            "source": "self_report",
            "timestamp": time.time(),
        }

    def from_task_difficulty(self, difficulty_level: int, completion_time: float,
                              expected_time: float) -> Dict[str, Any]:
        """Infer cognitive state from task difficulty and completion metrics.

        Args:
            difficulty_level: 1-5 difficulty rating
            completion_time: actual time taken (seconds)
            expected_time: expected time for the task (seconds)
        """
        time_ratio = completion_time / max(expected_time, 0.1)

        if difficulty_level <= 2 and time_ratio < 1.2:
            state = "focused"
        elif difficulty_level <= 2 and time_ratio > 2.0:
            state = "exploring"
        elif difficulty_level >= 4 and time_ratio > 2.0:
            state = "overloaded"
        elif difficulty_level >= 3 and time_ratio > 1.5:
            state = "confused"
        elif time_ratio > 1.8:
            state = "hesitating"
        else:
            state = "focused"

        confidence = min(0.9, 0.5 + abs(time_ratio - 1.0) * 0.2)

        return {
            "state": state,
            "confidence": confidence,
            "source": "task_difficulty",
            "difficulty_level": difficulty_level,
            "time_ratio": time_ratio,
            "timestamp": time.time(),
        }

    def from_nasa_tlx(self, ratings: Dict[str, int]) -> Dict[str, Any]:
        """Infer cognitive state from NASA-TLX workload ratings.

        Args:
            ratings: Dict with keys from nasa_tlx_dimensions, values 1-20
        """
        mental = ratings.get("mental_demand", 10)
        effort = ratings.get("effort", 10)
        frustration = ratings.get("frustration", 10)
        performance = ratings.get("performance", 10)

        overall_load = (mental + effort + frustration) / 3.0

        if overall_load > 15:
            state = "overloaded"
        elif frustration > 14:
            state = "confused"
        elif mental > 14 and performance < 8:
            state = "hesitating"
        elif mental < 8 and performance > 14:
            state = "focused"
        else:
            state = "exploring"

        confidence = min(0.85, overall_load / 20.0)

        return {
            "state": state,
            "confidence": confidence,
            "source": "nasa_tlx",
            "ratings": ratings,
            "overall_load": overall_load,
            "timestamp": time.time(),
        }

    def triangulate(self, labels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple label sources into a single triangulated label.

        Uses weighted voting based on source confidence.
        """
        if not labels:
            return {"state": "unknown", "confidence": 0, "source": "triangulated"}

        state_scores = {s: 0.0 for s in self.state_names}
        total_weight = 0

        source_weights = {
            "self_report": 1.0,
            "expert": 1.5,
            "task_difficulty": 0.7,
            "nasa_tlx": 0.9,
        }

        for label in labels:
            state = label.get("state", "unknown")
            if state in state_scores:
                weight = source_weights.get(label.get("source", ""), 0.5) * label.get("confidence", 0.5)
                state_scores[state] += weight
                total_weight += weight

        if total_weight == 0:
            return {"state": "unknown", "confidence": 0, "source": "triangulated"}

        best_state = max(state_scores, key=state_scores.get)
        confidence = state_scores[best_state] / total_weight

        return {
            "state": best_state,
            "confidence": float(confidence),
            "source": "triangulated",
            "component_labels": labels,
            "timestamp": time.time(),
        }


# Global singleton
label_generator = LabelGenerator()
