import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.ar_arch_model import CognitiveARARCH
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def train_ar_arch():
    print("🔄 Loading dataset for AR-ARCH...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "training_data")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "trained_models")
    os.makedirs(models_dir, exist_ok=True)

    X = np.load(os.path.join(data_dir, "X_train.npy"))
    y = np.load(os.path.join(data_dir, "y_train.npy"))

    print(f"📊 Dataset: {X.shape[0]} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # For AR-ARCH, we simulate short sequences from the summarized features 
    # to extract volatility. The dataset is already aggregated, so we expand it
    # slightly to demonstrate the ARCH volatility extraction pipeline.
    # Typically, AR-ARCH takes a true time series. Here we use the scalar feature
    # and add noise to create a synthetic sequence representing the session.
    
    def expand_to_sequence(agg_features):
        seq = []
        for feat in agg_features:
            # Recreate a sequence of 20 ticks around the feature mean
            s = np.random.normal(feat, scale=np.abs(feat)*0.1 + 1e-4, size=(20, feat.shape[0]))
            seq.append(s)
        return seq

    print("🔄 Training Bayesian AR-ARCH Model...")
    model = CognitiveARARCH()
    
    # Train
    train_seqs = expand_to_sequence(X_train)
    model.fit(train_seqs, y_train)

    # Evaluate
    test_seqs = expand_to_sequence(X_test)
    y_pred = []
    
    # Simple predict_proba gives dict normally, we want argmax
    for seq in test_seqs:
        probs = model.predict_proba(seq)
        y_pred.append(np.argmax(probs))

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ AR-ARCH Accuracy: {acc:.4f}")

    states = ["confidence", "confused", "exploring", "hesitating", "overloaded", "fatigue"]
    print(classification_report(y_test, y_pred, target_names=states, zero_division=0))

    # Save
    save_path = os.path.join(models_dir, "ar_arch.pkl")
    model.save(save_path)
    print(f"💾 Model saved to {save_path}")

    return acc

if __name__ == "__main__":
    train_ar_arch()
