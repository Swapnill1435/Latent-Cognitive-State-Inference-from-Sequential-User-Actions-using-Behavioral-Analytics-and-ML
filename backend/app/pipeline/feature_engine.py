"""Behavioral Feature Engineering Engine.

Extracts 24 behavioral features from raw telemetry event sequences:
- Temporal: reaction time, pause duration, task completion latency
- MPP (Marked Point Process): intensity rate, burstiness, inter-event regularity, log-RT statistics
- Sequential: action transition probabilities, navigation entropy, loop detection, backtracking
- Spatial: mouse velocity, acceleration, trajectory curvature, directional changes, midline deviation
- Decision: answer change count, hesitation time
"""

import math
import numpy as np
from collections import Counter
from typing import Any, Dict, List, Optional


def _safe_division(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def extract_temporal_features(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract temporal features from event sequence."""
    if len(events) < 2:
        return {
            "mean_reaction_time": 0.0,
            "std_reaction_time": 0.0,
            "max_pause_duration": 0.0,
            "mean_pause_duration": 0.0,
            "task_completion_latency": 0.0,
            "action_rate": 0.0,
        }

    timestamps = [e.get("timestamp", 0) for e in events]
    deltas = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    deltas = [d for d in deltas if d >= 0]

    if not deltas:
        deltas = [0.0]

    total_duration = max(timestamps) - min(timestamps) if timestamps else 0

    return {
        "mean_reaction_time": float(np.mean(deltas)),
        "std_reaction_time": float(np.std(deltas)),
        "max_pause_duration": float(max(deltas)),
        "mean_pause_duration": float(np.median(deltas)),
        "task_completion_latency": float(total_duration),
        "action_rate": _safe_division(len(events), total_duration / 1000.0),
    }


def extract_mpp_features(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract Marked Point Process (MPP) features.

    Models event sequences as a temporal point process to capture:
    - Event intensity (rate)
    - Burstiness (deviation from Poisson process)
    - Inter-event regularity
    - Log-RT statistics (reaction time in log-space for normality)
    """
    if len(events) < 3:
        return {
            "mpp_intensity_rate": 0.0,
            "mpp_burstiness": 0.0,
            "mpp_regularity": 0.0,
            "mpp_log_rt_mean": 0.0,
        }

    timestamps = [e.get("timestamp", 0) for e in events]
    deltas = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    deltas = [max(d, 1) for d in deltas]  # avoid log(0)

    total_duration = max(timestamps) - min(timestamps)

    # Intensity rate: events per second
    intensity = _safe_division(len(events), total_duration / 1000.0)

    # Burstiness B = (σ - μ) / (σ + μ), ranges from -1 (regular) to 1 (bursty)
    mu = float(np.mean(deltas))
    sigma = float(np.std(deltas))
    burstiness = _safe_division(sigma - mu, sigma + mu)

    # Regularity: coefficient of variation (lower = more regular)
    regularity = 1.0 - min(_safe_division(sigma, mu), 2.0) / 2.0  # normalized 0-1

    # Log-RT: mean of log-transformed reaction times (more normally distributed)
    log_rts = [math.log(d) for d in deltas if d > 0]
    log_rt_mean = float(np.mean(log_rts)) if log_rts else 0.0

    return {
        "mpp_intensity_rate": float(intensity),
        "mpp_burstiness": float(burstiness),
        "mpp_regularity": float(regularity),
        "mpp_log_rt_mean": float(log_rt_mean),
    }


def extract_sequential_features(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract sequential features: transition probabilities, entropy, loops, backtracking."""
    if len(events) < 2:
        return {
            "navigation_entropy": 0.0,
            "action_entropy": 0.0,
            "loop_count": 0.0,
            "backtrack_count": 0.0,
            "unique_action_ratio": 0.0,
        }

    event_types = [e.get("type", "unknown") for e in events]

    # Action entropy (Shannon)
    type_counts = Counter(event_types)
    total = len(event_types)
    probs = [c / total for c in type_counts.values()]
    action_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    # Navigation entropy (based on page/path transitions)
    paths = [e.get("path", e.get("page", "")) for e in events if e.get("path") or e.get("page")]
    if len(paths) > 1:
        path_counts = Counter(paths)
        path_total = len(paths)
        path_probs = [c / path_total for c in path_counts.values()]
        nav_entropy = -sum(p * math.log2(p) for p in path_probs if p > 0)
    else:
        nav_entropy = 0.0

    # Loop detection (repeated consecutive action patterns of length 2-3)
    loop_count = 0
    for window_size in [2, 3]:
        for i in range(len(event_types) - 2 * window_size + 1):
            pattern = tuple(event_types[i:i + window_size])
            next_pattern = tuple(event_types[i + window_size:i + 2 * window_size])
            if pattern == next_pattern:
                loop_count += 1

    # Backtracking (navigating to previously visited pages)
    backtrack_count = 0
    visited = set()
    for p in paths:
        if p in visited:
            backtrack_count += 1
        visited.add(p)

    unique_types = len(type_counts)

    return {
        "navigation_entropy": float(nav_entropy),
        "action_entropy": float(action_entropy),
        "loop_count": float(loop_count),
        "backtrack_count": float(backtrack_count),
        "unique_action_ratio": _safe_division(unique_types, total),
    }


def extract_spatial_features(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract spatial features from mouse movement events."""
    mouse_events = [e for e in events if e.get("x") is not None and e.get("y") is not None]

    if len(mouse_events) < 2:
        return {
            "mean_velocity": 0.0,
            "max_velocity": 0.0,
            "mean_acceleration": 0.0,
            "direction_changes_x": 0.0,
            "direction_changes_y": 0.0,
            "trajectory_curvature": 0.0,
            "midline_deviation": 0.0,
        }

    xs = [e["x"] for e in mouse_events]
    ys = [e["y"] for e in mouse_events]
    ts = [e.get("timestamp", i) for i, e in enumerate(mouse_events)]

    # Velocity
    velocities = []
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dt = max(ts[i] - ts[i - 1], 1)
        dist = math.sqrt(dx ** 2 + dy ** 2)
        velocities.append(dist / dt * 1000)  # pixels per second

    # Acceleration
    accelerations = []
    for i in range(1, len(velocities)):
        dt = max(ts[i + 1] - ts[i], 1)
        accelerations.append(abs(velocities[i] - velocities[i - 1]) / dt * 1000)

    # Direction changes
    dx_signs = [np.sign(xs[i] - xs[i - 1]) for i in range(1, len(xs))]
    dy_signs = [np.sign(ys[i] - ys[i - 1]) for i in range(1, len(ys))]

    dir_changes_x = sum(1 for i in range(1, len(dx_signs)) if dx_signs[i] != dx_signs[i - 1] and dx_signs[i] != 0)
    dir_changes_y = sum(1 for i in range(1, len(dy_signs)) if dy_signs[i] != dy_signs[i - 1] and dy_signs[i] != 0)

    # Trajectory curvature (total angular change)
    angles = []
    for i in range(1, len(xs) - 1):
        v1 = (xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        v2 = (xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 > 0 and mag2 > 0:
            cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
            angles.append(math.acos(cos_angle))

    curvature = sum(angles) if angles else 0.0

    # Midline deviation (AUC from straight path between first and last point)
    if len(xs) >= 2:
        start = np.array([xs[0], ys[0]])
        end = np.array([xs[-1], ys[-1]])
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len > 0:
            line_unit = line_vec / line_len
            deviations = []
            for i in range(len(xs)):
                point = np.array([xs[i], ys[i]])
                proj = np.dot(point - start, line_unit)
                closest = start + proj * line_unit
                dev = np.linalg.norm(point - closest)
                deviations.append(dev)
            midline_dev = float(np.mean(deviations))
        else:
            midline_dev = 0.0
    else:
        midline_dev = 0.0

    return {
        "mean_velocity": float(np.mean(velocities)) if velocities else 0.0,
        "max_velocity": float(max(velocities)) if velocities else 0.0,
        "mean_acceleration": float(np.mean(accelerations)) if accelerations else 0.0,
        "direction_changes_x": float(dir_changes_x),
        "direction_changes_y": float(dir_changes_y),
        "trajectory_curvature": float(curvature),
        "midline_deviation": float(midline_dev),
    }


def extract_decision_features(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract decision features: answer changes, hesitation time."""
    answer_events = [e for e in events if e.get("type") in ("answer_select", "answer_change", "decision")]
    answer_changes = sum(1 for e in events if e.get("type") == "answer_change")

    # Hesitation: time between first viewing a decision and making it
    hesitation_times = []
    decision_start = None
    for e in events:
        if e.get("type") in ("decision_view", "question_view"):
            decision_start = e.get("timestamp", 0)
        elif e.get("type") in ("answer_select", "decision") and decision_start is not None:
            hesitation_times.append(e.get("timestamp", 0) - decision_start)
            decision_start = None

    return {
        "answer_change_count": float(answer_changes),
        "mean_hesitation_time": float(np.mean(hesitation_times)) if hesitation_times else 0.0,
    }


def extract_all_features(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract all behavioral features from an event sequence.

    Returns a dict of 24 named features including MPP features.
    """
    features = {}
    features.update(extract_temporal_features(events))
    features.update(extract_mpp_features(events))
    features.update(extract_sequential_features(events))
    features.update(extract_spatial_features(events))
    features.update(extract_decision_features(events))
    return features


# Canonical feature order for model input (24 features)
FEATURE_NAMES = [
    # Temporal (6)
    "mean_reaction_time", "std_reaction_time", "max_pause_duration",
    "mean_pause_duration", "task_completion_latency", "action_rate",
    # MPP (4)
    "mpp_intensity_rate", "mpp_burstiness", "mpp_regularity", "mpp_log_rt_mean",
    # Sequential (5)
    "navigation_entropy", "action_entropy", "loop_count",
    "backtrack_count", "unique_action_ratio",
    # Spatial (7)
    "mean_velocity", "max_velocity", "mean_acceleration",
    "direction_changes_x", "direction_changes_y",
    "trajectory_curvature", "midline_deviation",
    # Decision (2)
    "answer_change_count", "mean_hesitation_time",
]

NUM_FEATURES = len(FEATURE_NAMES)


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to numpy vector in canonical order."""
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
