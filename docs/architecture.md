# Architecture Documentation

## Five-Layer Cognitive Inference Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  1. USER INTERACTION LAYER                   │
│  PuzzleTask │ DecisionTask │ NavigationTask                  │
│  React.js + Behavioral Telemetry Tracker                     │
├─────────────────────────────────────────────────────────────┤
│                2. BEHAVIORAL DATA CAPTURE LAYER              │
│  Mouse moves (60Hz) │ Clicks │ Scrolls │ Keystrokes         │
│  WebSocket streaming → FastAPI backend                       │
├─────────────────────────────────────────────────────────────┤
│             3. FEATURE ENGINEERING ENGINE                     │
│  Temporal │ Sequential │ Spatial │ Decision features         │
│  Sliding windows │ Rolling aggregation                       │
├─────────────────────────────────────────────────────────────┤
│             4. ML INFERENCE ENGINE                            │
│  HMM │ LSTM │ Transformer → Ensemble predictions            │
│  5 states: focused, confused, exploring, hesitating, overloaded │
├─────────────────────────────────────────────────────────────┤
│             5. VISUALIZATION & ADAPTATION LAYER              │
│  Dashboard │ Adaptive UI │ Explainability                    │
│  Real-time state display │ Contextual hints                  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. User interacts with tasks in the browser
2. `BehavioralTracker` captures events at ms-precision (performance.now())
3. Events are batched and streamed via WebSocket to FastAPI
4. `StreamProcessor` manages sliding windows and triggers feature extraction
5. `FeatureEngine` extracts 20 behavioral features per window
6. `InferenceOrchestrator` runs HMM + LSTM + Transformer ensemble
7. Predictions are broadcast back via WebSocket
8. Dashboard displays real-time cognitive state analytics
9. `AdaptiveUI` adjusts the interface based on predicted state

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, D3.js, Recharts |
| Telemetry | JavaScript event listeners, WebSocket |
| Backend | Python, FastAPI, Uvicorn |
| ML Models | PyTorch (LSTM, Transformer), hmmlearn (HMM) |
| Data Processing | NumPy, Pandas, scikit-learn |
| Privacy | ε-Differential Privacy (Laplace noise) |
| Explainability | SHAP, LIME, heuristic feature importance |

## ML Model Details

### Hidden Markov Model (HMM)
- 5 hidden states, Gaussian emissions
- Baum-Welch (EM) training
- Viterbi decoding for state sequences
- Posterior probabilities for real-time inference

### LSTM Network
- Input projection → 2 stacked LSTM layers (128 hidden) → Dropout → Dense
- Cross-entropy loss, Adam optimizer, early stopping
- Variable-length sequence support

### Transformer Encoder
- Sinusoidal positional encoding
- 4 multi-head attention layers (4 heads)
- GELU activation, layer normalization
- Global average pooling → classification head

### Ensemble
- Weighted average: HMM 25%, LSTM 35%, Transformer 40%
- Normalized probability output across 5 states
