# dl-inference-qa

A QA test harness for deep learning **inference**, built to validate image-classification
models across two frameworks (Keras, PyTorch) and their ONNX exports.

It mirrors, at small scale, the work of validating NVIDIA's deep learning software:
testing that models produce correct, fast, and consistent results — and that a model
stays numerically equivalent after conversion to a deployment format.

## What it tests

| Category | File | What it validates |
|---|---|---|
| Functional | `tests/test_functional.py` | Output shape, dtype, no NaN/Inf, valid probability distributions |
| Correctness | `tests/test_correctness.py` | Determinism, known-input accuracy, accuracy threshold (regression catch) |
| Performance | `tests/test_performance.py` | Latency (p50/p95) and throughput across batch sizes |
| ONNX equivalence | `tests/test_onnx_equivalence.py` | Source model vs ONNX export match within `1e-4` tolerance |

Every test runs against **both** backends via a parametrized `predictor` fixture —
one test function, two backends, no duplicated code.

## Results

56 tests passing: 20 functional, 6 correctness, 18 performance, 12 ONNX equivalence.

## Quick start

```bashpython3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txtTrain both models (produces models/.keras and models/.pt)
python scripts/train_keras.py
python scripts/train_torch.pyExport both to ONNX
python scripts/export_keras_to_onnx.py
python scripts/export_torch_to_onnx.pyRun the suite
pytest                       # all 56 tests
pytest -m "not performance"  # skip slow perf tests
pytest -k keras              # only the Keras backend

## Engineering notes

- **Unified inference interface.** Each backend is wrapped in a class with an
  identical `predict()` contract, so tests are backend-agnostic. Adding a third
  backend touches only `conftest.py`.
- **Device control.** All inference is pinned to CPU. The ONNX equivalence test
  caught a real CPU-vs-GPU numerical discrepancy: `model.predict()` ran through
  the Apple Metal GPU plugin's graph optimizer and diverged from the CPU ONNX by
  up to 0.48. Pinning the source to CPU isolates true conversion drift (now ~3e-7).
- **Toolchain workaround.** `tf2onnx` is incompatible with Keras 3; the Keras
  export routes through a TensorFlow SavedModel intermediate.
- **Tolerance design.** Continuous outputs (probabilities) are asserted with a
  tolerance; discrete outputs (argmax class) are asserted exact.

## Project structurescripts/   training and ONNX-export scripts
src/       inference wrappers (keras / torch / onnx)
tests/     pytest suite + shared fixtures (conftest.py)
models/    trained + exported artifacts (gitignored)