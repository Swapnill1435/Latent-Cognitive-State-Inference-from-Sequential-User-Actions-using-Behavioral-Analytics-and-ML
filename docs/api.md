# API Documentation

## REST Endpoints

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions` | Create a new tracking session |
| GET | `/api/sessions` | List all sessions |
| GET | `/api/sessions/{id}` | Get session details |
| GET | `/api/sessions/{id}/events` | Get raw telemetry events |
| GET | `/api/sessions/{id}/features` | Get computed behavioral features |
| GET | `/api/sessions/{id}/predictions` | Get cognitive state predictions |
| POST | `/api/sessions/{id}/labels` | Add ground-truth label |
| GET | `/api/sessions/{id}/analytics` | Get session analytics summary |

### Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dashboard/overview` | Platform overview stats |
| GET | `/api/dashboard/sessions/{id}/state-timeline` | State predictions over time |
| GET | `/api/dashboard/sessions/{id}/heatmap` | Mouse position heatmap data |
| GET | `/api/dashboard/sessions/{id}/feature-importance` | Feature importance scores |

## WebSocket

### Endpoint: `ws://localhost:8000/ws/{session_id}`

**Client → Server messages:**

```json
{
  "type": "telemetry",
  "events": [
    {
      "type": "click",
      "timestamp": 12345.67,
      "x": 500,
      "y": 300,
      "path": "/task/puzzle"
    }
  ]
}
```

```json
{
  "type": "label",
  "label": {
    "state": "confused",
    "confidence": 0.8,
    "source": "self_report"
  }
}
```

**Server → Client messages:**

```json
{
  "type": "cognitive_state",
  "data": {
    "timestamp": 1234567890.123,
    "predicted_state": "focused",
    "confidence": 0.78,
    "probabilities": {
      "focused": 0.78,
      "confused": 0.05,
      "exploring": 0.10,
      "hesitating": 0.04,
      "overloaded": 0.03
    },
    "model_outputs": {
      "hmm": { "focused": 0.80, ... },
      "lstm": { "focused": 0.75, ... },
      "transformer": { "focused": 0.82, ... }
    }
  }
}
```

## Label Request Body

```json
{
  "state": "confused",
  "confidence": 0.9,
  "source": "self_report",
  "nasa_tlx": {
    "mental_demand": 15,
    "physical_demand": 3,
    "temporal_demand": 12,
    "performance": 8,
    "effort": 14,
    "frustration": 16
  }
}
```
