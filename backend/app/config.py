"""Application configuration for the Cognitive State Inference Platform."""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """ML model hyperparameters."""
    # Cognitive states
    cognitive_states: List[str] = field(default_factory=lambda: [
        "confidence", "confused", "exploring", "hesitating", "overloaded", "fatigue"
    ])
    num_states: int = 6

    # HMM
    hmm_n_iter: int = 100
    hmm_covariance_type: str = "full"

    # LSTM
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 50
    lstm_batch_size: int = 32

    # Transformer
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 4
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1
    transformer_learning_rate: float = 0.0005
    transformer_epochs: int = 50
    transformer_batch_size: int = 32


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    window_size_seconds: float = 30.0
    mouse_sample_rate_hz: int = 60
    min_events_per_window: int = 5
    num_features: int = 24


@dataclass
class StreamConfig:
    """Streaming pipeline parameters."""
    window_slide_seconds: float = 5.0
    max_session_duration_seconds: float = 3600.0
    inference_debounce_ms: int = 500


@dataclass
class PrivacyConfig:
    """Privacy parameters."""
    epsilon: float = 1.0  # Differential privacy budget
    noise_mechanism: str = "laplace"
    anonymize_coordinates: bool = True


@dataclass
class AppConfig:
    """Root application configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ])
    models_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


config = AppConfig()
