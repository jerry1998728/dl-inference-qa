"""
Functional tests: structural properties of the model's output.

Every test taking `predictor` runs twice — once per backend — via the
parametrized fixture in conftest.py.
"""
import numpy as np
import pytest


def test_output_shape(predictor, random_batch):
    """Output is (batch_size, num_classes) = (8, 10)."""
    out = predictor.predict(random_batch)
    assert out.shape == (8, 10), \
        f"{predictor.name}: expected (8, 10), got {out.shape}"


def test_output_dtype(predictor, random_batch):
    """Output is float32 — important for downstream ONNX/TRT pipelines."""
    out = predictor.predict(random_batch)
    assert out.dtype == np.float32, \
        f"{predictor.name}: expected float32, got {out.dtype}"


def test_no_nan_or_inf(predictor, random_batch):
    """Output contains no NaN or Inf — catches numerical instability."""
    out = predictor.predict(random_batch)
    assert not np.isnan(out).any(), f"{predictor.name}: output has NaN"
    assert not np.isinf(out).any(), f"{predictor.name}: output has Inf"


def test_probabilities_in_unit_interval(predictor, random_batch):
    """Every probability is in [0, 1]."""
    out = predictor.predict(random_batch)
    assert (out >= 0.0).all() and (out <= 1.0).all(), \
        f"{predictor.name}: values outside [0, 1] — softmax may be missing/broken"


def test_probabilities_sum_to_one(predictor, random_batch):
    """Each row sums to 1.0 within float32 tolerance."""
    out = predictor.predict(random_batch)
    row_sums = out.sum(axis=1)
    np.testing.assert_allclose(
        row_sums, 1.0, atol=1e-5,
        err_msg=f"{predictor.name}: rows don't sum to 1.0",
    )


def test_top_class_in_valid_range(predictor, random_batch):
    """argmax produces a class index in [0, 9]."""
    out = predictor.predict(random_batch)
    top = out.argmax(axis=1)
    assert (top >= 0).all() and (top < 10).all(), \
        f"{predictor.name}: top class index out of [0, 9]"


@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
def test_handles_various_batch_sizes(predictor, batch_size):
    """
    Output shape adapts to input batch size.
    Compound parametrize: predictor (2) × batch_size (4) = 8 test runs from one function.
    """
    rng = np.random.default_rng(seed=batch_size)
    x = rng.random(size=(batch_size, 28, 28), dtype=np.float32)
    out = predictor.predict(x)
    assert out.shape == (batch_size, 10), \
        f"{predictor.name} @ batch={batch_size}: expected ({batch_size}, 10), got {out.shape}"