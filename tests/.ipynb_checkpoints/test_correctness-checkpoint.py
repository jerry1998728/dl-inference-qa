"""
Correctness tests: the model behaves like the trained MNIST classifier we expect.

Functional tests check structure. Correctness tests check substance — if someone
swapped the trained weights with random ones, functional tests would still pass.
These wouldn't.
"""
import numpy as np
import pytest
from torchvision import datasets


# --- Tests ---

def test_deterministic(predictor, random_batch):
    """Same input → byte-identical output across calls."""
    out1 = predictor.predict(random_batch)
    out2 = predictor.predict(random_batch)
    np.testing.assert_array_equal(
        out1, out2,
        err_msg=f"{predictor.name}: predict() is non-deterministic on identical input",
    )


def test_first_test_image_classified_correctly(predictor, mnist_test):
    """The first MNIST test image should be predicted as its true label."""
    images, labels = mnist_test
    out = predictor.predict(images[:1])
    predicted = int(out.argmax(axis=1)[0])
    expected = int(labels[0])
    assert predicted == expected, (
        f"{predictor.name}: first MNIST test image is a {expected}, "
        f"model predicted {predicted}"
    )


def test_accuracy_on_test_sample(predictor, mnist_test):
    """Top-1 accuracy on 100 real test digits should exceed 0.95."""
    images, labels = mnist_test
    out = predictor.predict(images)
    predicted = out.argmax(axis=1)
    accuracy = (predicted == labels).mean()
    assert accuracy > 0.95, (
        f"{predictor.name}: accuracy {accuracy:.4f} below 0.95 threshold — "
        "model may be untrained or corrupted"
    )