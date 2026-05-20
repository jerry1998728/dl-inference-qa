"""
Train a CNN on MNIST using Keras and save the model to disk.

Run: python scripts/train_keras.py
Output: models/mnist_cnn_keras.keras
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# --- Reproducibility ---
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# --- Config ---
MODEL_PATH = "models/mnist_cnn_keras.keras"
EPOCHS = 2
BATCH_SIZE = 512


def load_data():
    """Load MNIST, normalize to [0, 1], reshape to (N, 28, 28, 1)."""
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Hold out 5000 for validation
    x_valid = x_train_full[:5000].astype("float32") / 255.0
    x_train = x_train_full[5000:].astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # One-hot the labels
    y_train_full_cat = keras.utils.to_categorical(y_train_full, 10)
    y_valid = y_train_full_cat[:5000]
    y_train = y_train_full_cat[5000:]
    y_test = keras.utils.to_categorical(y_test, 10)

    # Add channel dim: (N, 28, 28) -> (N, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_valid = x_valid.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def build_model():
    """Same architecture as the original notebook."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, kernel_size=5, strides=1, padding="same",
                      activation="relu", name="conv1"),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.Conv2D(36, kernel_size=5, strides=1, padding="same",
                      activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print(f"TensorFlow {tf.__version__}")
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data()
    print(f"Train: {x_train.shape}, Valid: {x_valid.shape}, Test: {x_test.shape}")

    model = build_model()
    model.summary()

    model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_valid, y_valid),
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()