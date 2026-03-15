"""REST API routes for sessions, features, predictions, and labels."""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.storage.session_store import store
from app.pipeline.feature_engine import extract_all_features

router = APIRouter(prefix="/api")


class CreateSessionRequest(BaseModel):
    user_id: str = "anonymous"


class LabelRequest(BaseModel):
    state: str
    confidence: float = 1.0
    source: str = "self_report"  # self_report, expert, task_difficulty
    nasa_tlx: Optional[dict] = None
    timestamp: Optional[float] = None


@router.post("/sessions")
async def create_session(req: CreateSessionRequest):
    """Create a new tracking session."""
    session_id = store.create_session(req.user_id)
    return {"session_id": session_id}


@router.get("/sessions")
async def list_sessions():
    """List all sessions with summary info."""
    return {"sessions": store.list_sessions()}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get full session details."""
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return {
        "session_id": session["session_id"],
        "user_id": session["user_id"],
        "created_at": session["created_at"],
        "event_count": len(session["events"]),
        "prediction_count": len(session["predictions"]),
        "label_count": len(session["labels"]),
    }


@router.get("/sessions/{session_id}/events")
async def get_events(session_id: str, last_n: Optional[int] = None):
    """Get raw events for a session."""
    if not store.session_exists(session_id):
        raise HTTPException(404, "Session not found")
    return {"events": store.get_events(session_id, last_n)}


@router.get("/sessions/{session_id}/features")
async def get_features(session_id: str, last_n: Optional[int] = None):
    """Get computed features for a session."""
    if not store.session_exists(session_id):
        raise HTTPException(404, "Session not found")
    features = store.get_features(session_id, last_n)
    if not features:
        # Compute on the fly from current events
        events = store.get_events(session_id)
        if events:
            computed = extract_all_features(events)
            return {"features": [{"timestamp": time.time(), "features": computed}]}
    return {"features": features}


@router.get("/sessions/{session_id}/predictions")
async def get_predictions(session_id: str, last_n: Optional[int] = None):
    """Get cognitive state predictions for a session."""
    if not store.session_exists(session_id):
        raise HTTPException(404, "Session not found")
    return {"predictions": store.get_predictions(session_id, last_n)}


@router.post("/sessions/{session_id}/labels")
async def add_label(session_id: str, req: LabelRequest):
    """Add a ground-truth cognitive state label."""
    if not store.session_exists(session_id):
        raise HTTPException(404, "Session not found")
    label = {
        "state": req.state,
        "confidence": req.confidence,
        "source": req.source,
        "nasa_tlx": req.nasa_tlx,
        "timestamp": req.timestamp or time.time(),
    }
    store.add_label(session_id, label)
    return {"status": "ok", "label": label}


@router.get("/sessions/{session_id}/analytics")
async def get_analytics(session_id: str):
    """Get session analytics summary."""
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    predictions = session["predictions"]
    events = session["events"]

    # Compute state distribution
    state_counts = {}
    for p in predictions:
        state = p.get("predicted_state", "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1

    total_preds = len(predictions)
    state_distribution = {k: v / total_preds for k, v in state_counts.items()} if total_preds > 0 else {}

    # Event type distribution
    event_type_counts = {}
    for e in events:
        etype = e.get("type", "unknown")
        event_type_counts[etype] = event_type_counts.get(etype, 0) + 1

    return {
        "session_id": session_id,
        "total_events": len(events),
        "total_predictions": total_preds,
        "state_distribution": state_distribution,
        "event_type_distribution": event_type_counts,
        "duration_seconds": (events[-1]["timestamp"] - events[0]["timestamp"]) / 1000.0 if len(events) >= 2 else 0,
    }
