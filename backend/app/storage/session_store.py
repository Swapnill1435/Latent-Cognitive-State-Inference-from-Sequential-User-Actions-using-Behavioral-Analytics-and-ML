"""Thread-safe in-memory session and event store."""

import threading
import time
import uuid
from typing import Any, Dict, List, Optional


class SessionStore:
    """In-memory store for sessions, raw events, features, predictions, and labels."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str = "anonymous") -> str:
        """Create a new session, return session_id."""
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": time.time(),
                "events": [],
                "features": [],
                "predictions": [],
                "labels": [],
                "metadata": {},
            }
        return session_id

    def add_event(self, session_id: str, event: Dict[str, Any]):
        """Append a raw telemetry event to a session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["events"].append(event)

    def add_events(self, session_id: str, events: List[Dict[str, Any]]):
        """Append multiple events."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["events"].extend(events)

    def add_features(self, session_id: str, features: Dict[str, Any]):
        """Append computed feature snapshot."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["features"].append(features)

    def add_prediction(self, session_id: str, prediction: Dict[str, Any]):
        """Append a cognitive state prediction."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["predictions"].append(prediction)

    def add_label(self, session_id: str, label: Dict[str, Any]):
        """Append a ground-truth label."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["labels"].append(label)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get full session data."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_events(self, session_id: str, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get events for a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            events = session["events"]
            if last_n is not None:
                return events[-last_n:]
            return list(events)

    def get_predictions(self, session_id: str, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get predictions for a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            preds = session["predictions"]
            if last_n is not None:
                return preds[-last_n:]
            return list(preds)

    def get_features(self, session_id: str, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get features for a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            feats = session["features"]
            if last_n is not None:
                return feats[-last_n:]
            return list(feats)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with summary info."""
        with self._lock:
            summaries = []
            for sid, s in self._sessions.items():
                summaries.append({
                    "session_id": sid,
                    "user_id": s["user_id"],
                    "created_at": s["created_at"],
                    "event_count": len(s["events"]),
                    "prediction_count": len(s["predictions"]),
                })
            return summaries

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions


# Global singleton
store = SessionStore()
