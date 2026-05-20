"""
Export the trained PyTorch MNIST CNN to ONNX format.

Run: python scripts/export_torch_to_onnx.py
Output: models/mnist_cnn_torch.onnx
"""
import os
import sys
import torch

# Allow importing the model definition
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_torch import MnistCNN

TORCH_PATH = "models/mnist_cnn_torch.pt"
ONNX_PATH = "models/mnist_cnn_torch.onnx"
OPSET = 17


def main():
    print(f"Loading PyTorch model from {TORCH_PATH}")
    model = MnistCNN()
    model.load_state_dict(torch.load(TORCH_PATH, map_location="cpu", weights_only=True))
    model.eval()

    # A dummy input of any batch size — used only to trace the graph
    dummy = torch.randn(1, 1, 28, 28)

    print(f"Exporting to ONNX (opset {OPSET})")
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        opset_version=OPSET,
        dynamic_axes={
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    print(f"Saved: {ONNX_PATH}")


if __name__ == "__main__":
    main()