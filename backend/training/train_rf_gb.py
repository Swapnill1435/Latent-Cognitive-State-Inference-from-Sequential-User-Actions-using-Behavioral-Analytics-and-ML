import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.rf_gb_model import CognitiveTreeEnsemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def train_rf_gb():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "training_data")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "trained_models")
    os.makedirs(models_dir, exist_ok=True)

    X = np.load(os.path.join(data_dir, "X_train.npy"))
    y = np.load(os.path.join(data_dir, "y_train.npy"))

    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} aggregated features")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    states = ["confidence", "confused", "exploring", "hesitating", "overloaded", "fatigue"]
    
    # 1. Random Forest
    print("\n🔄 Training Random Forest...")
    rf_model = CognitiveTreeEnsemble(model_type="random_forest")
    rf_model.fit(X_train, y_train)

    y_pred_rf = [np.argmax(rf_model.predict_proba(x)) for x in X_test]
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"✅ Random Forest Accuracy: {acc_rf:.4f}")
    
    rf_save_path = os.path.join(models_dir, "random_forest.pkl")
    rf_model.save(rf_save_path)
    print(f"💾 Random Forest saved to {rf_save_path}")

    # 2. Gradient Boosting
    print("\n🔄 Training Gradient Boosting...")
    gb_model = CognitiveTreeEnsemble(model_type="gradient_boosting")
    gb_model.fit(X_train, y_train)

    y_pred_gb = [np.argmax(gb_model.predict_proba(x)) for x in X_test]
    acc_gb = accuracy_score(y_test, y_pred_gb)
    print(f"✅ Gradient Boosting Accuracy: {acc_gb:.4f}")
    print(classification_report(y_test, y_pred_gb, target_names=states, zero_division=0))

    gb_save_path = os.path.join(models_dir, "gradient_boosting.pkl")
    gb_model.save(gb_save_path)
    print(f"💾 Gradient Boosting saved to {gb_save_path}")

    return acc_gb

if __name__ == "__main__":
    train_rf_gb()
