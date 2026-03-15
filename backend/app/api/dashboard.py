"""Dashboard data API — aggregated analytics for the visualization dashboard."""

from fastapi import APIRouter, HTTPException
from typing import Optional

from app.storage.session_store import store
from app.pipeline.feature_engine import extract_all_features, FEATURE_NAMES

router = APIRouter(prefix="/api/dashboard")


@router.get("/overview")
async def dashboard_overview():
    """Get high-level platform overview."""
    sessions = store.list_sessions()
    total_events = sum(s["event_count"] for s in sessions)
    total_predictions = sum(s["prediction_count"] for s in sessions)
    return {
        "total_sessions": len(sessions),
        "total_events": total_events,
        "total_predictions": total_predictions,
        "active_sessions": len([s for s in sessions if s["event_count"] > 0]),
    }


@router.get("/sessions/{session_id}/state-timeline")
async def state_timeline(session_id: str):
    """Get cognitive state predictions over time for timeline visualization."""
    if not store.session_exists(session_id):
        raise HTTPException(404, "Session not found")

    predictions = store.get_predictions(session_id)
    timeline = []
    for p in predictions:
        timeline.append({
            "timestamp": p.get("timestamp", 0),
            "state": p.get("predicted_state", "unknown"),
            "confidence": p.get("confidence", 0),
            "probabilities": p.get("probabilities", {}),
        })
    return {"timeline": timeline}


@router.get("/sessions/{session_id}/heatmap")
async def interaction_heatmap(session_id: str):
    """Get mouse position data for heatmap visualization."""
    if not store.session_exists(session_id):
        raise HTTPException(404, "Session not found")

    events = store.get_events(session_id)
    points = []
    for e in events:
        if e.get("x") is not None and e.get("y") is not None:
            points.append({
                "x": e["x"],
                "y": e["y"],
                "type": e.get("type", "move"),
                "timestamp": e.get("timestamp", 0),
            })
    return {"heatmap": points}


@router.get("/sessions/{session_id}/feature-importance")
async def feature_importance(session_id: str):
    """Get feature values for the latest window — used for importance visualization."""
    if not store.session_exists(session_id):
        raise HTTPException(404, "Session not found")

    features_list = store.get_features(session_id, last_n=1)
    if features_list:
        latest = features_list[-1].get("features", {})
    else:
        events = store.get_events(session_id)
        latest = extract_all_features(events) if events else {}

    # Return feature names and values
    importance = []
    for name in FEATURE_NAMES:
        importance.append({
            "feature": name,
            "value": latest.get(name, 0.0),
        })

    return {"features": importance, "feature_names": FEATURE_NAMES}
