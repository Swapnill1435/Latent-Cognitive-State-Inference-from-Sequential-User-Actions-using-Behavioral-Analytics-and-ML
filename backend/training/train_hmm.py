"""Training script for Hidden Markov Model."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.hmm_model import CognitiveHMM
from app.pipeline.feature_engine import NUM_FEATURES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def train_hmm():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "training_data")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "trained_models")
    os.makedirs(models_dir, exist_ok=True)

    X = np.load(os.path.join(data_dir, "X_train.npy"))
    y = np.load(os.path.join(data_dir, "y_train.npy"))

    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Gaussian NB model (acting as stationary HMM emission probabilities)
    print("🔄 Training Gaussian NB (Static HMM equivalent)...")
    hmm = CognitiveHMM()

    # Fit using the standard extracted arrays
    hmm.fit(X_train, y_train)

    # Evaluate using scaled test data
    y_pred = hmm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ HMM Equivalent Accuracy: {acc:.4f}")

    # Save
    save_path = os.path.join(models_dir, "hmm.pkl")
    hmm.save(save_path)
    print(f"💾 Model saved to {save_path}")

    return acc


if __name__ == "__main__":
    train_hmm()
