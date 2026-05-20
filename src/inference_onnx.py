"""
Inference wrapper around ONNX models, run through ONNX Runtime.

Same uniform interface as KerasPredictor / TorchPredictor:
    predict(images: (N,28,28) float[0,1]) -> probabilities (N,10) float32

Handles the two real asymmetries discovered during export:
  - Keras ONNX:  input 'input_layer', channels-last,  output already softmax'd
  - PyTorch ONNX: input 'input',      channels-first, output is raw logits
"""
from __future__ import annotations
import numpy as np
import onnxruntime as ort


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class OnnxPredictor:
    """Generic ONNX Runtime inference wrapper with a uniform predict()."""

    def __init__(self, model_path: str, layout: str, apply_softmax: bool, name: str):
        """
        Args:
            model_path: path to the .onnx file.
            layout: 'nhwc' (channels-last) or 'nchw' (channels-first).
            apply_softmax: True if the ONNX graph outputs raw logits.
            name: short identifier for error messages.
        """
        assert layout in ("nhwc", "nchw")
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        # Query names from the graph — never hardcode
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.layout = layout
        self.apply_softmax = apply_softmax
        self._name = name

    def predict(self, images: np.ndarray) -> np.ndarray:
        """(N,28,28) float in [0,1] -> (N,10) float32 probabilities."""
        x = self._prepare(images)
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        if self.apply_softmax:
            out = _softmax(out)
        return out.astype(np.float32)

    def _prepare(self, images: np.ndarray) -> np.ndarray:
        """Reshape (N,28,28) to the layout this ONNX graph expects."""
        if images.ndim == 3:
            if self.layout == "nhwc":
                images = images[..., np.newaxis]        # -> (N,28,28,1)
            else:                                        # nchw
                images = images[:, np.newaxis, :, :]    # -> (N,1,28,28)
        return images.astype(np.float32)

    @property
    def name(self) -> str:
        return self._name


# --- Factory functions: pre-configured for each exported model ---

def keras_onnx_predictor(model_path: str = "models/mnist_cnn_keras.onnx") -> OnnxPredictor:
    # Keras model had softmax as its last layer -> ONNX graph already outputs probs
    return OnnxPredictor(model_path, layout="nhwc", apply_softmax=False, name="keras_onnx")


def torch_onnx_predictor(model_path: str = "models/mnist_cnn_torch.onnx") -> OnnxPredictor:
    # PyTorch model returned raw logits -> apply softmax after ONNX inference
    return OnnxPredictor(model_path, layout="nchw", apply_softmax=True, name="torch_onnx")