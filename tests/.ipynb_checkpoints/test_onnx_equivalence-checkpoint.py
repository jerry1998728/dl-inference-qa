"""
ONNX equivalence tests — the QA pattern at the heart of TensorRT validation.

For each framework: run the SAME input through the source model and its ONNX
export, and assert the outputs match within a numerical tolerance. This is the
simplified version of how NVIDIA validates a TensorRT engine against its source.

Why per-framework and never Keras-vs-PyTorch: the two models have different
weights (different RNG + different initializers), so only same-source pairs
are expected to agree.
"""
import numpy as np
import pytest

from src.inference_onnx import keras_onnx_predictor, torch_onnx_predictor

# Tolerances for fp32 conversion drift (interview-grade: justify these numbers).
ATOL = 1e-4
RTOL = 1e-3


# --- ONNX predictors, loaded once per session ---

@pytest.fixture(scope="session")
def keras_onnx():
    return keras_onnx_predictor("models/mnist_cnn_keras.onnx")


@pytest.fixture(scope="session")
def torch_onnx():
    return torch_onnx_predictor("models/mnist_cnn_torch.onnx")


# --- Pair each source backend with its ONNX export ---

@pytest.fixture(params=["keras", "torch"], ids=["keras", "torch"])
def equivalence_pair(request, keras_predictor, torch_predictor, keras_onnx, torch_onnx):
    """Returns (source_predictor, onnx_predictor, label)."""
    if request.param == "keras":
        return keras_predictor, keras_onnx, "keras"
    return torch_predictor, torch_onnx, "torch"


# --- Tests ---

def test_onnx_output_is_well_formed(equivalence_pair, random_batch):
    """ONNX output has the right shape and is a valid probability distribution."""
    _, onnx, label = equivalence_pair
    out = onnx.predict(random_batch)
    assert out.shape == (8, 10), f"{label}_onnx: bad shape {out.shape}"
    np.testing.assert_allclose(
        out.sum(axis=1), 1.0, atol=1e-5,
        err_msg=f"{label}_onnx: rows don't sum to 1",
    )


def test_onnx_matches_source_within_tolerance(equivalence_pair, mnist_test, capsys):
    """
    THE core test: ONNX output must match the source framework within tolerance.
    Reports the actual max abs diff so we can see headroom vs the threshold.
    """
    source, onnx, label = equivalence_pair
    images, _ = mnist_test

    src_out = source.predict(images)
    onnx_out = onnx.predict(images)

    max_abs_diff = float(np.max(np.abs(src_out - onnx_out)))
    with capsys.disabled():
        print(f"\n  [{label}] source vs ONNX  max|diff|={max_abs_diff:.2e}  "
              f"(threshold atol={ATOL:.0e})")

    np.testing.assert_allclose(
        src_out, onnx_out, atol=ATOL, rtol=RTOL,
        err_msg=f"{label}: ONNX export drifted from source beyond tolerance",
    )


def test_onnx_argmax_agrees_with_source(equivalence_pair, mnist_test):
    """
    Even if probabilities drift slightly, the DECISION (top-1 class) must be
    identical. This is the test that matters most to a customer.
    """
    source, onnx, label = equivalence_pair
    images, _ = mnist_test

    src_pred = source.predict(images).argmax(axis=1)
    onnx_pred = onnx.predict(images).argmax(axis=1)

    n_disagree = int((src_pred != onnx_pred).sum())
    assert n_disagree == 0, (
        f"{label}: {n_disagree}/{len(src_pred)} top-1 predictions differ "
        "between source and ONNX"
    )


@pytest.mark.parametrize("batch_size", [1, 8, 64])
def test_onnx_equivalence_across_batch_sizes(equivalence_pair, batch_size):
    """Equivalence must hold at every batch size — proves dynamic axes work."""
    source, onnx, label = equivalence_pair
    x = np.random.default_rng(seed=batch_size).random((batch_size, 28, 28), dtype=np.float32)

    src_out = source.predict(x)
    onnx_out = onnx.predict(x)

    np.testing.assert_allclose(
        src_out, onnx_out, atol=ATOL, rtol=RTOL,
        err_msg=f"{label} @ batch={batch_size}: ONNX drifted from source",
    )