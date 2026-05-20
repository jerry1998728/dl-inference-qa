"""
Shared pytest fixtures for the inference test suite.

Key trick: the `predictor` fixture is parametrized with ["keras", "torch"], so
EVERY test that takes `predictor` as an argument is automatically run twice —
once per backend — with no per-test boilerplate.
"""
import sys
import os
import numpy as np
import pytest
from torchvision import datasets  # add near the other imports

# Make `src/` importable from test files
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.inference_keras import KerasPredictor  # noqa: E402
from src.inference_torch import TorchPredictor  # noqa: E402


# --- Backend predictors, loaded once per session (expensive setup) ---

@pytest.fixture(scope="session")
def keras_predictor():
    """Load the Keras model once for the entire test session."""
    return KerasPredictor("models/mnist_cnn_keras.keras")


@pytest.fixture(scope="session")
def torch_predictor():
    """Load the PyTorch model once for the entire test session."""
    return TorchPredictor("models/mnist_cnn_torch.pt")


# --- The parametrize-by-backend trick ---

@pytest.fixture(params=["keras", "torch"], ids=["keras", "torch"])
def predictor(request, keras_predictor, torch_predictor):
    """
    Yields either KerasPredictor or TorchPredictor based on `params`.
    Any test that takes `predictor` runs TWICE — once per backend — automatically.
    """
    return {"keras": keras_predictor, "torch": torch_predictor}[request.param]


# --- Shared test inputs ---

@pytest.fixture(scope="session")
def random_batch():
    """Deterministic dummy batch for structural/shape tests."""
    rng = np.random.default_rng(seed=42)
    return rng.random(size=(8, 28, 28), dtype=np.float32)


# --- Test data fixture ---

@pytest.fixture(scope="session")
def mnist_test():
    """
    First 100 MNIST test images + labels.
    Reuses the data PyTorch downloaded during training (./data).
    """
    ds = datasets.MNIST(root="./data", train=False, download=False)
    images = np.stack([
        np.array(ds[i][0], dtype=np.float32) / 255.0  # PIL -> normalized array
        for i in range(100)
    ])
    labels = np.array([ds[i][1] for i in range(100)], dtype=np.int64)
    return images, labels