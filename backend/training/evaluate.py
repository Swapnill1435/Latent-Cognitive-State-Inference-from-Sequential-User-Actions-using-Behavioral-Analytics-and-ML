"""Evaluation script — compare all 3 models on the test set."""

import os
import sys
import numpy as np
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch

from app.models.hmm_model import CognitiveHMM
from app.models.lstm_model import LSTMClassifier
from app.models.transformer_model import CognitiveTransformer
from app.pipeline.feature_engine import NUM_FEATURES


STATES = ["confidence", "confused", "exploring", "hesitating", "overloaded", "fatigue"]


def evaluate():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "training_data")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "trained_models")

    X = np.load(os.path.join(data_dir, "X_train.npy"))
    y = np.load(os.path.join(data_dir, "y_train.npy"))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # HMM
    print("=" * 50)
    print("HMM Evaluation")
    print("=" * 50)
    hmm = CognitiveHMM(n_states=5, n_features=NUM_FEATURES)
    hmm_path = os.path.join(models_dir, "hmm.pkl")
    if os.path.exists(hmm_path):
        hmm.load(hmm_path)
        y_pred_hmm = hmm.predict(X_test)
        # Remap states
        from collections import Counter
        mapping = {}
        for s in range(5):
            mask = y_pred_hmm == s
            if mask.sum() > 0:
                mapping[s] = Counter(y_test[mask]).most_common(1)[0][0]
            else:
                mapping[s] = s
        y_pred_hmm = np.array([mapping.get(s, s) for s in y_pred_hmm])
        acc = accuracy_score(y_test, y_pred_hmm)
        f1 = f1_score(y_test, y_pred_hmm, average="weighted")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
        print(classification_report(y_test, y_pred_hmm, target_names=STATES, zero_division=0))
        results["hmm"] = {"accuracy": acc, "f1": f1}
    else:
        print("  No trained HMM model found")

    # LSTM
    print("=" * 50)
    print("LSTM Evaluation")
    print("=" * 50)
    lstm_path = os.path.join(models_dir, "lstm.pt")
    if os.path.exists(lstm_path):
        model = LSTMClassifier().to(device)
        model.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=True))
        model.eval()
        X_t = torch.FloatTensor(X_test[:, np.newaxis, :]).to(device)
        with torch.no_grad():
            preds = model(X_t).argmax(dim=-1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
        print(classification_report(y_test, preds, target_names=STATES, zero_division=0))
        results["lstm"] = {"accuracy": acc, "f1": f1}
    else:
        print("  No trained LSTM model found")

    # Transformer
    print("=" * 50)
    print("Transformer Evaluation")
    print("=" * 50)
    tx_path = os.path.join(models_dir, "transformer.pt")
    if os.path.exists(tx_path):
        model = CognitiveTransformer().to(device)
        model.load_state_dict(torch.load(tx_path, map_location=device, weights_only=True))
        model.eval()
        X_t = torch.FloatTensor(X_test[:, np.newaxis, :]).to(device)
        with torch.no_grad():
            preds = model(X_t).argmax(dim=-1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
        print(classification_report(y_test, preds, target_names=STATES, zero_division=0))
        results["transformer"] = {"accuracy": acc, "f1": f1}
    else:
        print("  No trained Transformer model found")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, r in results.items():
        print(f"  {name:15s} — Accuracy: {r['accuracy']:.4f}, F1: {r['f1']:.4f}")

    # Save results
    with open(os.path.join(models_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    evaluate()
