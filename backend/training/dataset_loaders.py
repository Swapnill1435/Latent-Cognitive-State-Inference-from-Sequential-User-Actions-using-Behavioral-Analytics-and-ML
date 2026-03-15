"""Dataset Loaders for external behavioral datasets referenced in the project PDF.

Provides loaders for:
- EdNet (AI-based educational platform interactions)
- OULAD (Open University Learning Analytics Dataset)
- Junyi Academy (K-12 e-learning platform)
- SENSE-42 (sensor/behavioral dataset for cognitive states)
- UIC HCI Logs (Human-Computer Interaction clickstream data)

Each loader downloads (if needed), preprocesses, and converts the dataset
into the platform's canonical 24-feature format for model training.
"""

import os
import json
import zipfile
import csv
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline.feature_engine import FEATURE_NAMES

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
COGNITIVE_STATES = ["confidence", "confused", "exploring", "hesitating", "overloaded", "fatigue"]
STATE_TO_IDX = {s: i for i, s in enumerate(COGNITIVE_STATES)}
NUM_FEATURES = len(FEATURE_NAMES)


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────
# Utility: map proxy signals to cognitive state labels
# ────────────────────────────────────────────────────────────────

def infer_state_from_learning_metrics(
    correct: bool,
    response_time_ms: float,
    attempts: int = 1,
    hint_used: bool = False,
    avg_response_time: float = 5000,
) -> str:
    """Infer cognitive state from learning-task metrics using proxy heuristics.

    This mapping is based on the behavioral indicators described in the PDF:
    - Fast + correct → confidence
    - Slow + incorrect + hints → confused
    - Many attempts + answer changes → hesitating
    - Very fast + error-prone → overloaded
    - Progressively slower over session → fatigue
    - Broad exploration without deep engagement → exploring
    """
    if correct and response_time_ms < avg_response_time * 0.7:
        return "confidence"
    elif not correct and hint_used:
        return "confused"
    elif attempts > 2:
        return "hesitating"
    elif not correct and response_time_ms < avg_response_time * 0.4:
        return "overloaded"
    elif response_time_ms > avg_response_time * 2.0:
        return "fatigue"
    else:
        return "exploring"


def compute_session_features(
    timestamps_ms: List[float],
    correct_sequence: List[bool],
    response_times_ms: List[float],
    hint_counts: Optional[List[int]] = None,
) -> np.ndarray:
    """Convert a learning session into the canonical 24-feature vector.

    Maps learning analytics metrics to behavioral features:
    - Response times → temporal + MPP features
    - Correctness sequences → sequential features (entropy, loops)
    - Hint usage → decision features
    """
    n = len(timestamps_ms)
    if n < 2:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    rt = np.array(response_times_ms, dtype=np.float64)
    rt_clean = np.clip(rt, 1, None)

    # Temporal (6)
    mean_rt = float(np.mean(rt_clean))
    std_rt = float(np.std(rt_clean))
    max_pause = float(np.max(rt_clean))
    median_rt = float(np.median(rt_clean))
    total_duration = float(max(timestamps_ms) - min(timestamps_ms))
    action_rate = n / max(total_duration / 1000, 1)

    # MPP (4)
    deltas = np.diff(timestamps_ms)
    deltas_clean = np.clip(deltas, 1, None)
    intensity = n / max(total_duration / 1000, 1)
    mu_d, sigma_d = float(np.mean(deltas_clean)), float(np.std(deltas_clean))
    burstiness = (sigma_d - mu_d) / (sigma_d + mu_d) if (sigma_d + mu_d) > 0 else 0
    regularity = 1.0 - min(sigma_d / max(mu_d, 1), 2.0) / 2.0
    log_rt_mean = float(np.mean(np.log(rt_clean)))

    # Sequential (5)
    # Action entropy from correctness pattern
    correct_arr = np.array(correct_sequence, dtype=int)
    unique, counts = np.unique(correct_arr, return_counts=True)
    probs = counts / counts.sum()
    nav_entropy = float(-np.sum(probs * np.log2(np.clip(probs, 1e-10, 1))))

    # Action diversity
    action_types = ["correct" if c else "incorrect" for c in correct_sequence]
    from collections import Counter
    type_counts = Counter(action_types)
    type_probs = [c / n for c in type_counts.values()]
    action_entropy = float(-sum(p * math.log2(p) for p in type_probs if p > 0))

    # Loops: consecutive same-result patterns
    loops = sum(1 for i in range(1, n) if correct_sequence[i] == correct_sequence[i - 1] and not correct_sequence[i])
    backtracks = sum(1 for i in range(1, n) if correct_sequence[i] != correct_sequence[i - 1])
    unique_ratio = len(set(action_types)) / max(n, 1)

    # Spatial (7) — approximate from response time patterns
    # In learning datasets, we don't have mouse coords, so we use RT-derived proxies
    velocity_proxy = 1000 / max(mean_rt, 1)  # actions per second
    max_vel_proxy = 1000 / max(float(np.min(rt_clean)), 1)
    accel_proxy = float(np.std(np.diff(rt_clean))) / max(mean_rt, 1) if len(rt_clean) > 1 else 0

    # Direction changes in response time
    rt_diff = np.diff(rt_clean)
    dir_changes_x = float(np.sum(np.diff(np.sign(rt_diff)) != 0)) if len(rt_diff) > 1 else 0
    dir_changes_y = dir_changes_x * 0.8  # proxy

    curvature_proxy = float(np.sum(np.abs(np.diff(rt_clean, n=2)))) / max(n, 1) if n > 2 else 0
    midline_proxy = float(np.std(rt_clean)) / max(mean_rt, 1)

    # Decision (2)
    hint_total = sum(hint_counts) if hint_counts else 0
    answer_changes = hint_total  # proxy
    hesitation = float(np.mean([r for r in rt_clean if r > mean_rt * 1.5])) if any(r > mean_rt * 1.5 for r in rt_clean) else 0

    feature_vector = np.array([
        mean_rt, std_rt, max_pause, median_rt, total_duration, action_rate,
        intensity, burstiness, regularity, log_rt_mean,
        nav_entropy, action_entropy, float(loops), float(backtracks), unique_ratio,
        velocity_proxy, max_vel_proxy, accel_proxy,
        dir_changes_x, dir_changes_y, curvature_proxy, midline_proxy,
        float(answer_changes), hesitation,
    ], dtype=np.float32)

    return feature_vector


# ────────────────────────────────────────────────────────────────
# EdNet Dataset Loader
# ────────────────────────────────────────────────────────────────

class EdNetLoader:
    """Loader for the EdNet dataset (AI-based educational platform).

    EdNet contains interaction logs from an AI tutoring system.
    Download from: https://github.com/riiid/ednet

    Expected structure:
        datasets/ednet/
            KT1/  — question-level interaction data
                u{user_id}.csv — per-user interaction logs
    """

    DATASET_NAME = "ednet"

    def __init__(self, data_path: str = None):
        self.data_path = data_path or os.path.join(DATA_DIR, self.DATASET_NAME)

    def load(self, max_users: int = 100, sessions_per_user: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess EdNet data into feature matrix and labels.

        Returns:
            X: (n_samples, 24) feature matrix
            y: (n_samples,) label indices
        """
        kt1_dir = os.path.join(self.data_path, "KT1")
        if not os.path.exists(kt1_dir):
            print(f"⚠️ EdNet data not found at {kt1_dir}")
            print("  Download from: https://github.com/riiid/ednet")
            print("  Place KT1/ folder in datasets/ednet/")
            return self._generate_proxy_data(max_users * sessions_per_user)

        features_list, labels_list = [], []
        user_files = sorted(os.listdir(kt1_dir))[:max_users]

        for uf in user_files:
            filepath = os.path.join(kt1_dir, uf)
            try:
                df = pd.read_csv(filepath)
                # Split into sessions (gaps > 30min)
                df['timestamp'] = pd.to_numeric(df.get('timestamp', range(len(df))), errors='coerce')
                df = df.dropna(subset=['timestamp']).sort_values('timestamp')

                session_breaks = df['timestamp'].diff() > 1800000
                df['session'] = session_breaks.cumsum()

                for sid, session in df.groupby('session'):
                    if len(session) < 5:
                        continue

                    timestamps = session['timestamp'].tolist()
                    correct = session.get('correct', session.get('user_answer', pd.Series())).tolist()
                    correct = [bool(c) if isinstance(c, (int, float, bool)) else c == '1' for c in correct]

                    rts = np.diff(timestamps).tolist()
                    if not rts:
                        continue
                    rts.append(np.mean(rts))

                    hints = [0] * len(correct)

                    fv = compute_session_features(timestamps, correct, rts, hints)
                    state = infer_state_from_learning_metrics(
                        correct=correct[-1], response_time_ms=rts[-1],
                        attempts=1, avg_response_time=np.mean(rts)
                    )
                    features_list.append(fv)
                    labels_list.append(STATE_TO_IDX[state])

                    if len(features_list) >= max_users * sessions_per_user:
                        break

            except Exception as e:
                print(f"  Warning: Error processing {uf}: {e}")
                continue

            if len(features_list) >= max_users * sessions_per_user:
                break

        if not features_list:
            return self._generate_proxy_data(max_users * sessions_per_user)

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        print(f"✅ EdNet: Loaded {len(X)} sessions")
        return X, y

    def _generate_proxy_data(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate EdNet-style proxy data when real data is unavailable."""
        print("  📦 Generating EdNet-style proxy data...")
        from training.generate_synthetic_data import generate_dataset
        return generate_dataset(n_sessions_per_state=n // 6, output_dir=None)[:2]


# ────────────────────────────────────────────────────────────────
# OULAD Dataset Loader
# ────────────────────────────────────────────────────────────────

class OULADLoader:
    """Loader for the Open University Learning Analytics Dataset (OULAD).

    Download from: https://analyse.kmi.open.ac.uk/open_dataset
    Contains VLE (Virtual Learning Environment) interaction logs.

    Expected structure:
        datasets/oulad/
            studentVle.csv — student interactions with VLE resources
            studentInfo.csv — student demographics and outcomes
    """

    DATASET_NAME = "oulad"

    def __init__(self, data_path: str = None):
        self.data_path = data_path or os.path.join(DATA_DIR, self.DATASET_NAME)

    def load(self, max_sessions: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        vle_file = os.path.join(self.data_path, "studentVle.csv")
        info_file = os.path.join(self.data_path, "studentInfo.csv")

        if not os.path.exists(vle_file):
            print(f"⚠️ OULAD data not found at {vle_file}")
            print("  Download from: https://analyse.kmi.open.ac.uk/open_dataset")
            return self._generate_proxy_data(max_sessions)

        try:
            vle_df = pd.read_csv(vle_file)
            info_df = pd.read_csv(info_file) if os.path.exists(info_file) else None
        except Exception as e:
            print(f"  Error reading OULAD: {e}")
            return self._generate_proxy_data(max_sessions)

        # Merge outcome info if available
        if info_df is not None and 'final_result' in info_df.columns:
            merge_cols = [c for c in ['id_student', 'code_module', 'code_presentation'] if c in vle_df.columns and c in info_df.columns]
            if merge_cols:
                vle_df = vle_df.merge(info_df[merge_cols + ['final_result']], on=merge_cols, how='left')

        features_list, labels_list = [], []
        group_cols = ['id_student', 'code_module'] if 'code_module' in vle_df.columns else ['id_student']

        for name, group in vle_df.groupby(group_cols):
            if len(group) < 5:
                continue

            group = group.sort_values('date') if 'date' in group.columns else group

            # Create pseudo-timestamps from dates
            dates = group['date'].tolist() if 'date' in group.columns else list(range(len(group)))
            timestamps = [float(d) * 86400000 for d in dates]  # days to ms
            clicks = group['sum_click'].tolist() if 'sum_click' in group.columns else [1] * len(group)
            correct = [c > 2 for c in clicks]  # proxy: high clicks = engagement

            rts = np.diff(timestamps).tolist()
            if not rts:
                continue
            rts.append(np.mean(rts))

            fv = compute_session_features(timestamps, correct, rts)

            # Infer state from outcome
            result = group.get('final_result', pd.Series()).iloc[0] if 'final_result' in group.columns else None
            if result == 'Distinction' or result == 'Pass':
                state = "confidence"
            elif result == 'Withdrawn':
                state = "fatigue"
            elif result == 'Fail':
                state = "confused"
            else:
                state = infer_state_from_learning_metrics(
                    correct[-1], rts[-1], avg_response_time=np.mean(rts)
                )

            features_list.append(fv)
            labels_list.append(STATE_TO_IDX[state])

            if len(features_list) >= max_sessions:
                break

        if not features_list:
            return self._generate_proxy_data(max_sessions)

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        print(f"✅ OULAD: Loaded {len(X)} sessions")
        return X, y

    def _generate_proxy_data(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        print("  📦 Generating OULAD-style proxy data...")
        from training.generate_synthetic_data import generate_dataset
        return generate_dataset(n_sessions_per_state=n // 6, output_dir=None)[:2]


# ────────────────────────────────────────────────────────────────
# Junyi Academy Dataset Loader
# ────────────────────────────────────────────────────────────────

class JunyiLoader:
    """Loader for the Junyi Academy dataset (K-12 e-learning platform).

    Download from: https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198
    or: https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy

    Expected structure:
        datasets/junyi/
            junyi_ProblemLog_original.csv — problem-level interaction logs
    """

    DATASET_NAME = "junyi"

    def __init__(self, data_path: str = None):
        self.data_path = data_path or os.path.join(DATA_DIR, self.DATASET_NAME)

    def load(self, max_sessions: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        log_file = os.path.join(self.data_path, "junyi_ProblemLog_original.csv")

        if not os.path.exists(log_file):
            # Try alternative filename
            alt_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')] if os.path.exists(self.data_path) else []
            if alt_files:
                log_file = os.path.join(self.data_path, alt_files[0])
            else:
                print(f"⚠️ Junyi data not found at {self.data_path}")
                print("  Download from: https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy")
                return self._generate_proxy_data(max_sessions)

        try:
            df = pd.read_csv(log_file, nrows=100000)  # limit for memory
        except Exception as e:
            print(f"  Error reading Junyi: {e}")
            return self._generate_proxy_data(max_sessions)

        features_list, labels_list = [], []

        user_col = next((c for c in df.columns if 'user' in c.lower()), None)
        correct_col = next((c for c in df.columns if 'correct' in c.lower()), None)
        time_col = next((c for c in df.columns if 'time' in c.lower() and 'taken' in c.lower()), None)
        hint_col = next((c for c in df.columns if 'hint' in c.lower()), None)
        attempts_col = next((c for c in df.columns if 'attempt' in c.lower()), None)

        if user_col is None:
            return self._generate_proxy_data(max_sessions)

        for uid, group in df.groupby(user_col):
            if len(group) < 5:
                continue

            timestamps = list(range(len(group)))
            timestamps_ms = [t * 5000 for t in timestamps]

            correct = group[correct_col].tolist() if correct_col else [True] * len(group)
            correct = [bool(c) if isinstance(c, (int, float, bool)) else str(c).lower() in ('1', 'true') for c in correct]

            rts = group[time_col].tolist() if time_col else [5000] * len(group)
            rts = [max(float(r) if pd.notna(r) else 5000, 100) for r in rts]

            hints = group[hint_col].tolist() if hint_col else [0] * len(group)
            hints = [int(h) if pd.notna(h) else 0 for h in hints]

            attempts = group[attempts_col].iloc[-1] if attempts_col else 1
            attempts = int(attempts) if pd.notna(attempts) else 1

            fv = compute_session_features(timestamps_ms, correct, rts, hints)
            state = infer_state_from_learning_metrics(
                correct[-1], rts[-1], attempts,
                hint_used=sum(hints) > 0,
                avg_response_time=np.mean(rts)
            )

            features_list.append(fv)
            labels_list.append(STATE_TO_IDX[state])

            if len(features_list) >= max_sessions:
                break

        if not features_list:
            return self._generate_proxy_data(max_sessions)

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        print(f"✅ Junyi: Loaded {len(X)} sessions")
        return X, y

    def _generate_proxy_data(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        print("  📦 Generating Junyi-style proxy data...")
        from training.generate_synthetic_data import generate_dataset
        return generate_dataset(n_sessions_per_state=n // 6, output_dir=None)[:2]


# ────────────────────────────────────────────────────────────────
# SENSE-42 Dataset Loader
# ────────────────────────────────────────────────────────────────

class SENSE42Loader:
    """Loader for the SENSE-42 sensor/behavioral dataset.

    Contains multimodal sensor data including interaction patterns
    and self-reported cognitive/emotional states.

    Expected structure:
        datasets/sense42/
            participants/
                P{id}/
                    interactions.csv
                    self_reports.csv
    """

    DATASET_NAME = "sense42"

    def __init__(self, data_path: str = None):
        self.data_path = data_path or os.path.join(DATA_DIR, self.DATASET_NAME)

    def load(self, max_sessions: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        participants_dir = os.path.join(self.data_path, "participants")

        if not os.path.exists(participants_dir):
            # Try flat CSV structure
            csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')] if os.path.exists(self.data_path) else []
            if csv_files:
                return self._load_flat(csv_files[0], max_sessions)
            print(f"⚠️ SENSE-42 data not found at {self.data_path}")
            return self._generate_proxy_data(max_sessions)

        features_list, labels_list = [], []

        for pid in sorted(os.listdir(participants_dir)):
            p_dir = os.path.join(participants_dir, pid)
            if not os.path.isdir(p_dir):
                continue

            interactions_file = os.path.join(p_dir, "interactions.csv")
            reports_file = os.path.join(p_dir, "self_reports.csv")

            if not os.path.exists(interactions_file):
                continue

            try:
                df = pd.read_csv(interactions_file)
                reports = pd.read_csv(reports_file) if os.path.exists(reports_file) else None

                timestamps = df['timestamp'].tolist() if 'timestamp' in df.columns else list(range(len(df)))
                timestamps_ms = [float(t) * 1000 if float(t) < 1e10 else float(t) for t in timestamps]

                correct = df['correct'].tolist() if 'correct' in df.columns else [True] * len(df)
                correct = [bool(c) for c in correct]

                rts = list(np.diff(timestamps_ms))
                rts.append(np.mean(rts) if rts else 1000)

                fv = compute_session_features(timestamps_ms, correct, rts)

                # Use self-reports for ground truth if available
                if reports is not None and 'state' in reports.columns:
                    reported_state = reports['state'].iloc[-1].lower().strip()
                    if reported_state in STATE_TO_IDX:
                        state = reported_state
                    else:
                        state = infer_state_from_learning_metrics(correct[-1], rts[-1], avg_response_time=np.mean(rts))
                else:
                    state = infer_state_from_learning_metrics(correct[-1], rts[-1], avg_response_time=np.mean(rts))

                features_list.append(fv)
                labels_list.append(STATE_TO_IDX[state])

            except Exception as e:
                print(f"  Warning: Error processing {pid}: {e}")
                continue

            if len(features_list) >= max_sessions:
                break

        if not features_list:
            return self._generate_proxy_data(max_sessions)

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        print(f"✅ SENSE-42: Loaded {len(X)} sessions")
        return X, y

    def _load_flat(self, filename: str, max_sessions: int) -> Tuple[np.ndarray, np.ndarray]:
        filepath = os.path.join(self.data_path, filename)
        try:
            df = pd.read_csv(filepath, nrows=50000)
            # Generic processing
            features_list, labels_list = [], []
            user_col = next((c for c in df.columns if 'user' in c.lower() or 'participant' in c.lower()), df.columns[0])
            for uid, group in df.groupby(user_col):
                if len(group) < 5:
                    continue
                timestamps_ms = list(range(0, len(group) * 1000, 1000))
                correct = [True] * len(group)
                rts = [1000] * len(group)
                fv = compute_session_features(timestamps_ms, correct, rts)
                features_list.append(fv)
                labels_list.append(STATE_TO_IDX["exploring"])
                if len(features_list) >= max_sessions:
                    break
            if features_list:
                return np.array(features_list), np.array(labels_list)
        except Exception as e:
            print(f"  Error: {e}")
        return self._generate_proxy_data(max_sessions)

    def _generate_proxy_data(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        print("  📦 Generating SENSE-42-style proxy data...")
        from training.generate_synthetic_data import generate_dataset
        return generate_dataset(n_sessions_per_state=n // 6, output_dir=None)[:2]


# ────────────────────────────────────────────────────────────────
# UIC HCI Logs Loader
# ────────────────────────────────────────────────────────────────

class UICHCILoader:
    """Loader for the UIC HCI Clickstream / Interaction Logs.

    Contains mouse tracking and clickstream data from HCI experiments.
    Download from UIC HCI research group repositories.

    Expected structure:
        datasets/uic_hci/
            mouse_logs/
                session_{id}.csv — per-session mouse tracking
            click_logs/
                session_{id}.csv — per-session click data
    """

    DATASET_NAME = "uic_hci"

    def __init__(self, data_path: str = None):
        self.data_path = data_path or os.path.join(DATA_DIR, self.DATASET_NAME)

    def load(self, max_sessions: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        mouse_dir = os.path.join(self.data_path, "mouse_logs")
        click_dir = os.path.join(self.data_path, "click_logs")

        # Check for any CSV in main directory
        if not os.path.exists(mouse_dir):
            csv_files = [f for f in os.listdir(self.data_path) if f.endswith('.csv')] if os.path.exists(self.data_path) else []
            if csv_files:
                return self._load_generic_hci(csv_files, max_sessions)
            print(f"⚠️ UIC HCI data not found at {self.data_path}")
            return self._generate_proxy_data(max_sessions)

        features_list, labels_list = [], []

        session_files = sorted([f for f in os.listdir(mouse_dir) if f.endswith('.csv')])[:max_sessions]

        for sf in session_files:
            try:
                mouse_df = pd.read_csv(os.path.join(mouse_dir, sf))

                # Build events from mouse data
                events = []
                for _, row in mouse_df.iterrows():
                    events.append({
                        "type": row.get("event_type", "mousemove"),
                        "timestamp": float(row.get("timestamp", 0)),
                        "x": float(row.get("x", 0)),
                        "y": float(row.get("y", 0)),
                        "path": str(row.get("page", "/")),
                    })

                if len(events) < 5:
                    continue

                # Use our feature engine directly (this is real HCI data with coordinates!)
                from app.pipeline.feature_engine import extract_all_features, features_to_vector
                features = extract_all_features(events)
                fv = features_to_vector(features)

                # Infer state from behavioral metrics
                rt = features.get("mean_reaction_time", 500)
                entropy = features.get("action_entropy", 0)
                loops = features.get("loop_count", 0)
                velocity = features.get("mean_velocity", 0)

                if loops > 3 and entropy > 1.5:
                    state = "confused"
                elif entropy > 2.0:
                    state = "overloaded"
                elif rt > 2000:
                    state = "fatigue"
                elif rt > 1000:
                    state = "hesitating"
                elif velocity > 500 and entropy > 1.0:
                    state = "exploring"
                else:
                    state = "confidence"

                features_list.append(fv)
                labels_list.append(STATE_TO_IDX[state])

            except Exception as e:
                print(f"  Warning: Error processing {sf}: {e}")
                continue

        if not features_list:
            return self._generate_proxy_data(max_sessions)

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        print(f"✅ UIC HCI: Loaded {len(X)} sessions")
        return X, y

    def _load_generic_hci(self, csv_files: List[str], max_sessions: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load from generic CSV files in the HCI dataset directory."""
        features_list, labels_list = [], []
        for cf in csv_files:
            try:
                df = pd.read_csv(os.path.join(self.data_path, cf), nrows=50000)
                # Group by session/user if possible
                session_col = next((c for c in df.columns if 'session' in c.lower() or 'user' in c.lower()), None)
                if session_col:
                    for sid, group in df.groupby(session_col):
                        if len(group) < 5 or len(features_list) >= max_sessions:
                            continue
                        events = [{"type": "click", "timestamp": float(i) * 100, "x": 0, "y": 0, "path": "/"} for i in range(len(group))]
                        from app.pipeline.feature_engine import extract_all_features, features_to_vector
                        fv = features_to_vector(extract_all_features(events))
                        features_list.append(fv)
                        labels_list.append(STATE_TO_IDX["exploring"])
            except Exception as e:
                print(f"  Warning: Error processing {cf}: {e}")

        if features_list:
            return np.array(features_list), np.array(labels_list)
        return self._generate_proxy_data(max_sessions)

    def _generate_proxy_data(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        print("  📦 Generating UIC HCI-style proxy data...")
        from training.generate_synthetic_data import generate_dataset
        return generate_dataset(n_sessions_per_state=n // 6, output_dir=None)[:2]


# ────────────────────────────────────────────────────────────────
# Unified Dataset Pipeline
# ────────────────────────────────────────────────────────────────

def load_all_datasets(
    max_per_dataset: int = 300,
    include_synthetic: bool = True,
    synthetic_per_state: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and combine all available datasets into a unified training set.

    Automatically falls back to synthetic proxy data for unavailable datasets.

    Returns:
        X: (n_total, 24) feature matrix
        y: (n_total,) label indices
    """
    ensure_data_dir()

    all_X, all_y = [], []

    loaders = [
        ("EdNet", EdNetLoader()),
        ("OULAD", OULADLoader()),
        ("Junyi", JunyiLoader()),
        ("SENSE-42", SENSE42Loader()),
        ("UIC HCI", UICHCILoader()),
    ]

    for name, loader in loaders:
        print(f"\n📂 Loading {name}...")
        try:
            X, y = loader.load(max_sessions=max_per_dataset)
            all_X.append(X)
            all_y.append(y)
            print(f"  → {len(X)} samples ({name})")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    if include_synthetic:
        print(f"\n📦 Adding synthetic data ({synthetic_per_state} per state)...")
        from training.generate_synthetic_data import generate_dataset
        X_syn, y_syn, _ = generate_dataset(
            n_sessions_per_state=synthetic_per_state,
            output_dir=None
        )
        all_X.append(X_syn)
        all_y.append(y_syn)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "..", "training_data")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_train.npy"), X)
    np.save(os.path.join(output_dir, "y_train.npy"), y)

    print(f"\n✅ Combined dataset: {len(X)} samples, {NUM_FEATURES} features")
    print(f"📊 State distribution:")
    for state, idx in STATE_TO_IDX.items():
        count = np.sum(y == idx)
        print(f"  {state}: {count} ({count / len(y) * 100:.1f}%)")

    return X, y


if __name__ == "__main__":
    print("=" * 60)
    print("Loading all datasets for Cognitive State Inference Platform")
    print("=" * 60)
    X, y = load_all_datasets()
