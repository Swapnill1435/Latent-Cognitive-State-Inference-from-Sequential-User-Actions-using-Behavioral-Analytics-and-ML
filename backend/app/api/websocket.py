"""WebSocket endpoint for real-time telemetry streaming and cognitive state broadcast."""

import json
import asyncio
from typing import Set, Dict

from fastapi import WebSocket, WebSocketDisconnect

from app.storage.session_store import store
from app.pipeline.stream_processor import stream_processor
from app.pipeline.inference import orchestrator


class ConnectionManager:
    """Manages WebSocket connections per session."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast(self, session_id: str, message: dict):
        """Broadcast a message to all connections for a session."""
        if session_id in self.active_connections:
            data = json.dumps(message)
            dead = []
            for ws in self.active_connections[session_id]:
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.active_connections[session_id].discard(ws)


manager = ConnectionManager()


async def on_inference_result(session_id: str, features: dict, feature_vector):
    """Callback from stream processor — run inference and broadcast."""
    import numpy as np

    prediction = orchestrator.predict(feature_vector)

    # Store results
    store.add_features(session_id, {
        "timestamp": prediction["timestamp"],
        "features": features,
    })
    store.add_prediction(session_id, prediction)

    # Broadcast to connected clients
    await manager.broadcast(session_id, {
        "type": "cognitive_state",
        "data": prediction,
    })


# Register inference callback
stream_processor.set_inference_callback(on_inference_result)


async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for telemetry ingestion and state broadcast.

    Expects JSON messages with format:
    {
        "type": "telemetry",
        "events": [
            {"type": "click", "timestamp": 12345, "x": 100, "y": 200, ...},
            ...
        ]
    }
    or:
    {
        "type": "create_session",
        "user_id": "optional_user_id"
    }
    """
    await manager.connect(websocket, session_id)

    # Ensure session exists
    if not store.session_exists(session_id):
        store.create_session.__func__(store)  # create with default
        # Manually set the session id
        from app.storage.session_store import store as s
        import threading
        with s._lock:
            if session_id not in s._sessions:
                import time, uuid
                s._sessions[session_id] = {
                    "session_id": session_id,
                    "user_id": "anonymous",
                    "created_at": time.time(),
                    "events": [],
                    "features": [],
                    "predictions": [],
                    "labels": [],
                    "metadata": {},
                }

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = message.get("type", "")

            if msg_type == "telemetry":
                events = message.get("events", [])
                if events:
                    store.add_events(session_id, events)
                    await stream_processor.process_events(session_id, events)

            elif msg_type == "label":
                label = message.get("label", {})
                store.add_label(session_id, label)

    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception:
        manager.disconnect(websocket, session_id)
