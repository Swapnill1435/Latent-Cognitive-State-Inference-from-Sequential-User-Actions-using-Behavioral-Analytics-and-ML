"""Transformer-based model for cognitive state inference with attention map extraction.

Architecture: Positional Encoding -> Multi-Head Self-Attention (4 heads, 4 layers) ->
Feed-Forward -> Classification Head

Includes attention weight extraction for explainability (Attention Maps).
"""

import os
import math
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CognitiveTransformer(nn.Module):
    """Transformer encoder for behavioral sequence classification with attention extraction."""

    def __init__(
        self,
        input_size: int = None,
        d_model: int = None,
        nhead: int = None,
        num_layers: int = None,
        dim_feedforward: int = None,
        num_classes: int = None,
        dropout: float = None,
    ):
        super().__init__()
        self.input_size = input_size or config.features.num_features
        self.d_model = d_model or config.model.transformer_d_model
        self.nhead = nhead or config.model.transformer_nhead
        self.num_layers = num_layers or config.model.transformer_num_layers
        self.dim_feedforward = dim_feedforward or config.model.transformer_dim_feedforward
        self.num_classes = num_classes or config.model.num_states
        self.dropout_rate = dropout or config.model.transformer_dropout

        # Input projection
        self.input_proj = nn.Linear(self.input_size, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout_rate)

        # Custom Transformer encoder layers (to extract attention weights)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_rate,
                batch_first=True,
            )
            for _ in range(self.num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.num_classes),
        )

        # Storage for attention weights (populated during forward_with_attention)
        self._attention_weights = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass without attention extraction."""
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Forward pass that also returns attention weights from each layer.

        Returns:
            logits: (batch, num_classes)
            attention_weights: list of (batch, nhead, seq_len, seq_len) tensors per layer.
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        attention_weights = []
        for layer in self.encoder_layers:
            # Extract attention weights by hooking into self_attn
            attn_output, attn_weight = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            # Still pass through the full layer for proper residual + feedforward
            x = layer(x)
            attention_weights.append(attn_weight.detach())

        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits, attention_weights


class CognitiveTransformerInference:
    """Wrapper for Transformer-based cognitive state inference with attention maps."""

    def __init__(self):
        self.model: Optional[CognitiveTransformer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_names = config.model.cognitive_states
        self._last_attention_weights = None

    def build(self):
        self.model = CognitiveTransformer().to(self.device)

    def predict_proba(self, feature_sequence: np.ndarray) -> np.ndarray:
        """Predict cognitive state probabilities."""
        if self.model is None:
            self.build()

        self.model.eval()

        if feature_sequence.ndim == 1:
            feature_sequence = feature_sequence.reshape(1, -1)
        if feature_sequence.ndim == 2:
            feature_sequence = feature_sequence[np.newaxis, :]

        x = torch.FloatTensor(feature_sequence).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

        return probs.cpu().numpy()[0]

    def predict_with_attention(self, feature_sequence: np.ndarray) -> Tuple[np.ndarray, list]:
        """Predict cognitive state probabilities and return attention maps.

        Returns:
            probs: (num_classes,) probability vector.
            attention_maps: list of (nhead, seq_len, seq_len) numpy arrays per layer.
        """
        if self.model is None:
            self.build()

        self.model.eval()

        if feature_sequence.ndim == 1:
            feature_sequence = feature_sequence.reshape(1, -1)
        if feature_sequence.ndim == 2:
            feature_sequence = feature_sequence[np.newaxis, :]

        x = torch.FloatTensor(feature_sequence).to(self.device)

        with torch.no_grad():
            logits, attn_weights = self.model.forward_with_attention(x)
            probs = F.softmax(logits, dim=-1)

        # Convert attention to numpy
        attention_maps = [w.cpu().numpy()[0] for w in attn_weights]  # remove batch dim
        self._last_attention_weights = attention_maps

        return probs.cpu().numpy()[0], attention_maps

    def get_last_attention_maps(self) -> Optional[list]:
        """Return the attention maps from the last prediction."""
        return self._last_attention_weights

    def predict_latest(self, feature_sequence: np.ndarray) -> Dict[str, float]:
        probs = self.predict_proba(feature_sequence)
        return {name: float(p) for name, p in zip(self.state_names, probs)}

    def save(self, path: str):
        if self.model is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        if self.model is None:
            self.build()
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
