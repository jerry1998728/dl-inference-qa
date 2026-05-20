"""
Inference wrapper around the trained PyTorch MNIST CNN.

Mirrors KerasPredictor's interface so tests are backend-agnostic.
"""
from __future__ import annotations
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

# Allow importing the model definition from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_torch import MnistCNN  # noqa: E402


class TorchPredictor:
    """Loads the saved PyTorch CNN once, runs inference on (N, 28, 28) arrays."""

    def __init__(self, model_path: str = "models/mnist_cnn_torch.pt", device: str = "cpu"):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = MnistCNN()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()   # critical: disables dropout for deterministic inference

    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Args:
            images: shape (N, 28, 28) or (N, 1, 28, 28), float in [0, 1].
        Returns:
            probabilities: shape (N, 10), float32, each row sums to ~1.0.
        """
        x = self._prepare(images)
        with torch.no_grad():
            logits = self.model(x)                       # raw (N, 10)
            probs = F.softmax(logits, dim=1)             # match Keras output semantics
        return probs.cpu().numpy().astype(np.float32)

    def _prepare(self, images: np.ndarray) -> torch.Tensor:
        """Normalize input to channels-first (N, 1, 28, 28) torch tensor."""
        if images.ndim == 3:                            # (N, 28, 28)
            images = images[:, np.newaxis, :, :]        # -> (N, 1, 28, 28)
        elif images.ndim == 4 and images.shape[-1] == 1:   # channels-last passthrough
            images = np.transpose(images, (0, 3, 1, 2))    # -> (N, 1, 28, 28)
        tensor = torch.from_numpy(images.astype(np.float32)).to(self.device)
        return tensor

    @property
    def name(self) -> str:
        return "torch"