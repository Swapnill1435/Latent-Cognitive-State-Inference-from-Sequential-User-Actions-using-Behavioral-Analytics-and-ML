# User Guide

## Prerequisites

- **Python 3.9+** with pip
- **Node.js 18+** with npm

## Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

## Training ML Models

Before first use, generate synthetic training data and train the models:

```bash
cd backend
python training/generate_synthetic_data.py
python training/train_hmm.py
python training/train_lstm.py
python training/train_transformer.py
python training/evaluate.py
```

This creates `trained_models/` directory with trained weights.

## Running the Platform

### Start Backend

```bash
cd backend
python run.py
```

Backend runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### Start Frontend

```bash
cd frontend
npm run dev
```

Frontend runs at `http://localhost:5173`.

## Using the Platform

1. **Open** `http://localhost:5173` in your browser
2. The sidebar shows available tasks and connection status
3. **Puzzle Task** — Solve the sliding puzzle; your interactions are tracked
4. **Decision Task** — Answer decision scenarios; hesitation and changes are measured
5. **Navigation Task** — Find hidden information by navigating pages; backtracking is detected
6. **Dashboard** — View real-time cognitive state predictions, feature importance, and heatmaps

## Cognitive States

| State | Description | Behavioral Indicators |
|-------|-------------|----------------------|
| Focused | Engaged, productive | Smooth trajectories, consistent timing |
| Confused | Lost, uncertain | Navigation loops, backtracking, erratic movements |
| Exploring | Browsing, learning | Broad navigation, high entropy, varied actions |
| Hesitating | Uncertain about decisions | Long pauses, answer changes |
| Overloaded | Too much information | Rapid switching, high-entropy behavior |

## Configuration

Edit `backend/app/config.py` to modify:

- Model hyperparameters (LSTM layers, Transformer heads, etc.)
- Feature window size (default: 30 seconds)
- Privacy epsilon (default: 1.0)
- Inference debounce (default: 500ms)
