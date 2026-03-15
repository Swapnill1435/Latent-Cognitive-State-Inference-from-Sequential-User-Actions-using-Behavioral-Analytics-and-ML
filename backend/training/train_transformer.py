"""Training script for Transformer model."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from app.models.transformer_model import CognitiveTransformer
from app.config import config


def train_transformer():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "training_data")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "trained_models")
    os.makedirs(models_dir, exist_ok=True)

    X = np.load(os.path.join(data_dir, "X_train.npy"))
    y = np.load(os.path.join(data_dir, "y_train.npy"))

    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Reshape to (batch, seq_len=1, features)
    X_train_t = torch.FloatTensor(X_train[:, np.newaxis, :])
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test[:, np.newaxis, :])
    y_test_t = torch.LongTensor(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=config.model.transformer_batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CognitiveTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.model.transformer_learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.model.transformer_epochs)

    print("🔄 Training Transformer...")
    best_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(config.model.transformer_epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t.to(device))
            test_preds = test_logits.argmax(dim=-1).cpu().numpy()
            acc = accuracy_score(y_test, test_preds)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{config.model.transformer_epochs} — Loss: {total_loss / len(train_loader):.4f}, Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(models_dir, "transformer.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    print(f"\n✅ Transformer Best Accuracy: {best_acc:.4f}")

    model.load_state_dict(torch.load(os.path.join(models_dir, "transformer.pt"), weights_only=True))
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t.to(device))
        test_preds = test_logits.argmax(dim=-1).cpu().numpy()

    states = ["confidence", "confused", "exploring", "hesitating", "overloaded", "fatigue"]
    print(classification_report(y_test, test_preds, target_names=states, zero_division=0))
    print(f"💾 Model saved to {os.path.join(models_dir, 'transformer.pt')}")

    return best_acc


if __name__ == "__main__":
    train_transformer()
