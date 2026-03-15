"""Synthetic behavioral data generator for training ML models.

Generates realistic behavioral sessions with known cognitive state labels
by simulating different behavioral patterns per state.
Now includes 6 states: confidence, confused, exploring, hesitating, overloaded, fatigue.
"""

import os
import json
import random
import numpy as np
from typing import Any, Dict, List, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline.feature_engine import extract_all_features, features_to_vector, FEATURE_NAMES


COGNITIVE_STATES = ["confidence", "confused", "exploring", "hesitating", "overloaded", "fatigue"]
STATE_TO_IDX = {s: i for i, s in enumerate(COGNITIVE_STATES)}


def generate_confidence_events(n_events: int = 50, start_time: float = 0) -> List[Dict[str, Any]]:
    """Confidence: smooth trajectories, consistent timing, low entropy, decisive actions."""
    events = []
    t = start_time
    x, y = 500, 400

    for i in range(n_events):
        dt = random.gauss(280, 40)  # very consistent timing ~280ms
        dt = max(50, dt)
        t += dt

        # Smooth deliberate movement toward target
        x += random.gauss(6, 1.5)
        y += random.gauss(4, 1)
        x = max(0, min(1920, x))
        y = max(0, min(1080, y))

        event_type = random.choice(["mousemove", "mousemove", "mousemove", "click"])
        events.append({
            "type": event_type, "timestamp": t, "x": x, "y": y,
            "path": "/task/puzzle",
        })

    return events


def generate_confused_events(n_events: int = 50, start_time: float = 0) -> List[Dict[str, Any]]:
    """Confused: navigation loops, high pause variance, backtracking, direction reversals."""
    events = []
    t = start_time
    x, y = 500, 400
    pages = ["/task/puzzle", "/help", "/task/puzzle", "/help", "/task/decision", "/task/puzzle"]

    for i in range(n_events):
        dt = random.gauss(800, 400)
        dt = max(50, dt)
        t += dt

        x += random.gauss(0, 30)
        y += random.gauss(0, 30)
        x = max(0, min(1920, x))
        y = max(0, min(1080, y))

        event_type = random.choice(["mousemove", "click", "scroll", "answer_change"])
        events.append({
            "type": event_type, "timestamp": t, "x": x, "y": y,
            "path": pages[i % len(pages)],
        })

    return events


def generate_exploring_events(n_events: int = 50, start_time: float = 0) -> List[Dict[str, Any]]:
    """Exploring: broad navigation, many unique pages, varied actions."""
    events = []
    t = start_time
    x, y = 500, 400
    pages = [f"/section/{i}" for i in range(10)]

    for i in range(n_events):
        dt = random.gauss(400, 100)
        dt = max(50, dt)
        t += dt

        x += random.gauss(10, 15)
        y += random.gauss(5, 10)
        x = max(0, min(1920, x))
        y = max(0, min(1080, y))

        event_type = random.choice(["mousemove", "click", "scroll", "hover"])
        events.append({
            "type": event_type, "timestamp": t, "x": x, "y": y,
            "path": random.choice(pages),
        })

    return events


def generate_hesitating_events(n_events: int = 50, start_time: float = 0) -> List[Dict[str, Any]]:
    """Hesitating: long pauses before decisions, answer changes."""
    events = []
    t = start_time
    x, y = 500, 400

    for i in range(n_events):
        if i % 10 < 3:
            dt = random.gauss(2000, 500)
        else:
            dt = random.gauss(300, 100)
        dt = max(50, dt)
        t += dt

        x += random.gauss(2, 3)
        y += random.gauss(1, 2)
        x = max(0, min(1920, x))
        y = max(0, min(1080, y))

        if i % 8 == 0:
            events.append({"type": "decision_view", "timestamp": t, "x": x, "y": y, "path": "/task/decision"})
        elif i % 8 == 5:
            events.append({"type": "answer_select", "timestamp": t, "x": x, "y": y, "path": "/task/decision"})
        elif i % 8 == 7:
            events.append({"type": "answer_change", "timestamp": t, "x": x, "y": y, "path": "/task/decision"})
        else:
            events.append({"type": "mousemove", "timestamp": t, "x": x, "y": y, "path": "/task/decision"})

    return events


def generate_overloaded_events(n_events: int = 50, start_time: float = 0) -> List[Dict[str, Any]]:
    """Overloaded: erratic high-entropy behavior, rapid switching, high velocity."""
    events = []
    t = start_time
    x, y = 500, 400
    pages = ["/task/puzzle", "/task/decision", "/help", "/settings", "/task/navigation"]

    for i in range(n_events):
        dt = random.gauss(200, 300)
        dt = max(30, dt)
        t += dt

        x += random.gauss(0, 50)
        y += random.gauss(0, 40)
        x = max(0, min(1920, x))
        y = max(0, min(1080, y))

        event_type = random.choice(["mousemove", "click", "scroll", "keystroke", "hover", "answer_change"])
        events.append({
            "type": event_type, "timestamp": t, "x": x, "y": y,
            "path": random.choice(pages),
        })

    return events


def generate_fatigue_events(n_events: int = 50, start_time: float = 0) -> List[Dict[str, Any]]:
    """Fatigue: progressively slower reactions, decreasing velocity, longer pauses, reduced interaction rate."""
    events = []
    t = start_time
    x, y = 500, 400

    for i in range(n_events):
        # Progressively slower — fatigue accumulates
        fatigue_factor = 1.0 + (i / n_events) * 2.0  # 1x to 3x slowdown
        dt = random.gauss(400 * fatigue_factor, 100 * fatigue_factor)
        dt = max(100, dt)
        t += dt

        # Progressively less precise movement
        x += random.gauss(2 / fatigue_factor, 2)
        y += random.gauss(1 / fatigue_factor, 1.5)
        x = max(0, min(1920, x))
        y = max(0, min(1080, y))

        # Fewer varied actions as fatigue sets in
        if i < n_events * 0.5:
            event_type = random.choice(["mousemove", "click", "scroll"])
        else:
            event_type = random.choice(["mousemove", "mousemove", "mousemove", "click"])

        events.append({
            "type": event_type, "timestamp": t, "x": x, "y": y,
            "path": "/task/puzzle",
        })

    return events


GENERATORS = {
    "confidence": generate_confidence_events,
    "confused": generate_confused_events,
    "exploring": generate_exploring_events,
    "hesitating": generate_hesitating_events,
    "overloaded": generate_overloaded_events,
    "fatigue": generate_fatigue_events,
}


def generate_session(state: str, n_events: int = 50) -> Tuple[List[Dict], str, np.ndarray]:
    """Generate a single session with features and label."""
    gen = GENERATORS.get(state, generate_confidence_events)
    events = gen(n_events=n_events)
    features = extract_all_features(events)
    vector = features_to_vector(features)
    return events, state, vector


def generate_dataset(
    n_sessions_per_state: int = 100,
    events_per_session: int = 50,
    output_dir: str = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate a full training dataset.

    Returns:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) label indices
        sessions_metadata: list of metadata dicts
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "training_data")
    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    all_labels = []
    metadata = []

    for state in COGNITIVE_STATES:
        for i in range(n_sessions_per_state):
            n_events = events_per_session + random.randint(-10, 10)
            events, label, vector = generate_session(state, max(20, n_events))
            all_features.append(vector)
            all_labels.append(STATE_TO_IDX[label])
            metadata.append({"state": state, "n_events": len(events)})

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    # Save
    np.save(os.path.join(output_dir, "X_train.npy"), X)
    np.save(os.path.join(output_dir, "y_train.npy"), y)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "states": COGNITIVE_STATES,
            "n_samples": len(X),
            "n_per_state": n_sessions_per_state,
            "n_features": len(FEATURE_NAMES),
        }, f, indent=2)

    print(f"✅ Generated {len(X)} samples ({n_sessions_per_state} per state, {len(COGNITIVE_STATES)} states)")
    print(f"📂 Saved to {output_dir}")
    print(f"📊 Feature shape: {X.shape}")

    return X, y, metadata


if __name__ == "__main__":
    generate_dataset(n_sessions_per_state=200, events_per_session=60)
