"""
Export the trained Keras MNIST CNN to ONNX format.

Keras 3 note: tf2onnx's from_keras() path is broken for Keras 3 models
(it accesses the removed `output_names` attribute). We work around it by
exporting to a TF SavedModel first, then converting from that — the
SavedModel path works at the graph level and is framework-version-agnostic.

Run: python scripts/export_keras_to_onnx.py
Output: models/mnist_cnn_keras.onnx
"""
import os
import sys
import shutil
import subprocess
from tensorflow import keras

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

KERAS_PATH = "models/mnist_cnn_keras.keras"
SAVEDMODEL_DIR = "models/_keras_savedmodel"
ONNX_PATH = "models/mnist_cnn_keras.onnx"
OPSET = 17


def main():
    import tensorflow as tf

    print(f"Loading Keras model from {KERAS_PATH}")
    model = keras.models.load_model(KERAS_PATH)

    # Step 1: export to SavedModel with inference mode EXPLICITLY forced.
    # model.export() does not reliably trace training=False when the model
    # contains Dropout — so we build the serving endpoint by hand.
    if os.path.exists(SAVEDMODEL_DIR):
        shutil.rmtree(SAVEDMODEL_DIR)
    print(f"Exporting to TF SavedModel (training=False forced) at {SAVEDMODEL_DIR}")

    archive = keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint(
        name="serve",
        fn=lambda x: model(x, training=False),   # <-- the fix: inference mode
        input_signature=[
            tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name="input")
        ],
    )
    archive.write_out(SAVEDMODEL_DIR)

    # Step 2: SavedModel -> ONNX via the tf2onnx CLI
    print(f"Converting SavedModel to ONNX (opset {OPSET})")
    subprocess.run(
        [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", SAVEDMODEL_DIR,
            "--output", ONNX_PATH,
            "--opset", str(OPSET),
        ],
        check=True,
    )
    print(f"\nSaved: {ONNX_PATH}")


if __name__ == "__main__":
    main()