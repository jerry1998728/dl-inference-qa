"""
Inference wrapper around the trained Keras MNIST CNN.

Exposes a uniform `predict(images) -> probabilities` interface so the test
harness can treat both backends identically.
"""
from __future__ import annotations
import os
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- Pin Keras inference to CPU -------------------------------------------
# model.predict() compiles a tf.function that the Apple Metal GPU plugin
# (tensorflow-metal) optimizes with graph rewrites that change numerics.
# Comparing a Metal-GPU source against a CPU ONNX export conflates conversion
# drift with device drift. A conversion-equivalence test must control for the
# device, so we run Keras inference on CPU — matching ONNX Runtime's CPU
# provider and TorchPredictor (already CPU-pinned).
import tensorflow as tf  # noqa: E402
try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass  # GPU already initialized in this process — harmless

from tensorflow import keras  # noqa: E402


class KerasPredictor:
    """Loads the saved Keras CNN once, runs inference on (N, 28, 28) arrays."""

    def __init__(self, model_path: str = "models/mnist_cnn_keras.keras"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Args:
            images: shape (N, 28, 28) or (N, 28, 28, 1), float in [0, 1].
        Returns:
            probabilities: shape (N, 10), float32, each row sums to ~1.0.
        """
        x = self._prepare(images)
        return self.model.predict(x, verbose=0).astype(np.float32)

    @staticmethod
    def _prepare(images: np.ndarray) -> np.ndarray:
        """Normalize input to channels-last (N, 28, 28, 1) float32."""
        if images.ndim == 3:                    # (N, 28, 28)
            images = images[..., np.newaxis]    # -> (N, 28, 28, 1)
        return images.astype(np.float32)

    @property
    def name(self) -> str:
        return "keras"