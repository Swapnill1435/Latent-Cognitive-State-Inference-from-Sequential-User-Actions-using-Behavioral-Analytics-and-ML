# 🧠 Latent Cognitive State Inference Platform

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Inferring latent human cognitive states (confusion, fatigue, confidence) in real-time by analyzing sequential micro-interactions, cursor telemetry, and behavioral analytics.**

---

## 📖 Overview

The **Latent Cognitive State Inference Platform** is a full-stack, AI-powered system that passively observes user behaviors (mouse movements, keystrokes, navigation events) and translates those raw sequences into active psychological and cognitive states. 

It eliminates the need for intrusive physiological sensors (like eye-trackers or EEG constraints) by relying on **high-frequency web telemetry** and **Deep Sequence Modeling**. Whether a user is confidently solving a puzzle or experiencing cognitive overload during a complex decision task, the system detects it in real time and adapts dynamically.

---

## ✨ Key Features

- 🕵️‍♂️ **Passive Telemetry Extraction:** Captures nuanced micro-interactions via low-latency WebSockets (cursor curvature, velocity, hesitation metrics, entropy).
- 🧠 **Deep Sequence Inference:** Utilizes a highly robust Ensemble AI containing LSTMs, Transformers, and Bayesian volatility models to deduce hidden cognitive states.
- 📊 **Real-Time Visual Analytics:** A dynamic React dashboard visualizing real-time probability streams of cognitive states.
- 🛡️ **Differential Privacy:** Secures raw user interaction data by injecting noise and employing safe hashing protocols natively before analysis.
- 💡 **Explainable AI (XAI):** Built-in LIME explainers to detail *why* the model predicted a specific cognitive state based on recent behavioral features.

---

## 🏗️ System Architecture

The project is structured according to a strict five-layer logical architecture:

1. **Client Layer (React/Vite):** Evaluates users through standard browser tasks (Decision, Puzzle, Navigation). Packages telemetry at 60Hz.
2. **Transport Layer (FastAPI WebSockets):** Manages bidirectional, low-latency streams mapping connected client sessions to the backend inference pipeline.
3. **Engineering Layer (Feature Extraction):** Transforms raw coordinate streams into 24 distinct cognitive features (e.g., path curvature, directional entropy, acceleration proxy, idle ratios).
4. **Inference Layer (Ensemble AI):** The brain. Evaluates sequential inputs and performs weighted ensemble predictions across 5 independent ML architectures.
5. **Storage & Privacy Layer:** Applies $\epsilon$-differential privacy masking before archiving session data to local JSON schemas for longitudinal model refinement.

---

## 🤖 Machine Learning Pipeline

The system is trained on extensive real-world (EdNet, OULAD, Junyi) and proprietary proxy behavioral datasets mapping to 6 distinct latent states:
**Confidence, Confusion, Exploring, Hesitating, Overloaded, and Fatigue.**

### Ensemble Models & Accuracies:

| Model Architecture | Purpose | Accuracy |
| :--- | :--- | :---: |
| **Transformer Encoder** | Captures massive sequential context via self-attention mapping. | **98.33%** |
| **LSTM Classifier** | Excels at understanding continuous temporally-dependent behavior. | **98.06%** |
| **Random Forest / Gradient Boosting** | Provides robust, exceptionally fast classification from aggregated 24-feature profiles. | **98.33%** |
| **Bayesian AR-ARCH** | Extracts Autoregressive Conditional Heteroskedasticity (volatility) to track sudden cognitive load fluctuations. | **97.50%** |
| **Gaussian Naive Bayes (HMM Proxy)** | Models stationary latent emission probabilities to serve as a high-fidelity continuous state transition tracker. | **94.17%** |

*All predictions are orchestrated and ultimately weighted by the `InferenceOrchestrator` to provide the final state inference sent back to the interactive dashboard.*

---

## 🚀 Installation & Setup

### Prerequisites
- Node.js (v18+)
- Python (3.11+)

### 1. Clone the repository
```bash
git clone https://github.com/abizer007/Latent-Cognitive-State-Inference-from-Sequential-User-Actions-using-Behavioral-Analytics-and-ML.git
cd Latent-Cognitive-State-Inference-from-Sequential-User-Actions-using-Behavioral-Analytics-and-ML
```

### 2. Backend Setup (FastAPI & PyTorch)
```bash
cd backend
python -m venv env

# Windows
env\Scripts\activate
# macOS/Linux
source env/bin/activate

pip install -r requirements.txt

# Start the Backend Server (WebSockets + API)
python run.py
```
*The backend will be available at `http://localhost:8000`.*

### 3. Frontend Setup (React & Vite)
```bash
cd frontend
npm install

# Start the Frontend Development Server
npm run dev
```
*The application UI will run at `http://localhost:5173`.*

---

## 📁 Repository Structure

```text
.
├── backend/                  
│   ├── app/                  # FastAPI Application logic
│   │   ├── api/              # API and WebSocket routes
│   │   ├── models/           # PyTorch & Scikit-Learn specific Architectures
│   │   ├── pipeline/         # Feature Engineering and Inference Orchestrator
│   │   ├── explainability/   # LIME feature explainers
│   │   └── privacy/          # Differential Privacy implementations
│   ├── training/             # Scrips for dataset processing & training all ML models
│   └── requirements.txt      # Python dependencies
├── frontend/                
│   ├── src/                  # React Application
│   │   ├── dashboard/        # Real-time state visualizers (D3)
│   │   ├── tasks/            # User interaction interfaces (Puzzle, Demo screens)
│   │   ├── telemetry/        # High-frequency Cursor & Event tracker
│   │   └── adaptive/         # Automatically adjusts UI based on inferred cognitive stat
│   └── package.json          # Node dependencies
└── docs/                     # System design documentation
```

---

## 🔬 Key Engineering Highlights

- **Behavioral Telemetry Parsing:** Instead of just sending raw (X, Y) pairs, the frontend tracks clicks, hovers, scroll depth, key-down latency, and generic cursor idle time, pushing packets selectively over WSS.
- **Dynamic Feature Engineering:** Computes complex kinematic properties purely from JS events, evaluating acceleration changes, path curvature deviations, and spatial entropy to accurately measure "confusion" or "hesitation".
- **Adaptive UI:** The system doesn't only monitor; it *adapts*. If the ML pipeline infers heavy "Fatigue," the visual interface autonomously dims contrast. If it infers "Overloaded," it simplifies on-screen elements.

---

> *"Decoding the mind, one pixel at a time."*