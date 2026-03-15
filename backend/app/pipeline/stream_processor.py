"""Real-time stream processor.

Manages session windows, computes rolling features, and triggers inference
on each window update. Replaces Kafka/Flink with efficient async Python.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from app.pipeline.feature_engine import extract_all_features, features_to_vector
from app.config import config


class SessionWindow:
    """Sliding window over a session's events."""

    def __init__(self, session_id: str, window_size_sec: float = 30.0, slide_sec: float = 5.0):
        self.session_id = session_id
        self.window_size_ms = window_size_sec * 1000
        self.slide_ms = slide_sec * 1000
        self.events: List[Dict[str, Any]] = []
        self.last_inference_time: float = 0

    def add_events(self, new_events: List[Dict[str, Any]]):
        """Add events and trim old ones outside the window."""
        self.events.extend(new_events)
        self._trim()

    def _trim(self):
        """Remove events older than the window size."""
        if not self.events:
            return
        latest = self.events[-1].get("timestamp", 0)
        cutoff = latest - self.window_size_ms
        self.events = [e for e in self.events if e.get("timestamp", 0) >= cutoff]

    def should_infer(self) -> bool:
        """Whether enough time has passed since last inference."""
        if len(self.events) < config.features.min_events_per_window:
            return False
        now = time.time() * 1000
        return (now - self.last_inference_time) >= config.stream.inference_debounce_ms

    def mark_inferred(self):
        self.last_inference_time = time.time() * 1000

    def get_windowed_events(self) -> List[Dict[str, Any]]:
        return list(self.events)


class StreamProcessor:
    """In-process stream processor for real-time behavioral analysis."""

    def __init__(self):
        self.windows: Dict[str, SessionWindow] = {}
        self._inference_callback: Optional[Callable] = None

    def set_inference_callback(self, callback: Callable):
        """Set callback invoked with (session_id, features_dict, features_vector) on each window trigger."""
        self._inference_callback = callback

    def get_or_create_window(self, session_id: str) -> SessionWindow:
        if session_id not in self.windows:
            self.windows[session_id] = SessionWindow(
                session_id,
                window_size_sec=config.features.window_size_seconds,
                slide_sec=config.stream.window_slide_seconds,
            )
        return self.windows[session_id]

    async def process_events(self, session_id: str, events: List[Dict[str, Any]]):
        """Process incoming events for a session."""
        window = self.get_or_create_window(session_id)
        window.add_events(events)

        if window.should_infer() and self._inference_callback:
            windowed = window.get_windowed_events()
            features = extract_all_features(windowed)
            vector = features_to_vector(features)
            window.mark_inferred()
            await self._inference_callback(session_id, features, vector)

    def remove_session(self, session_id: str):
        self.windows.pop(session_id, None)


# Global singleton
stream_processor = StreamProcessor()
