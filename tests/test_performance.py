"""
Performance tests: latency and throughput across batch sizes, per backend.

Run with `pytest tests/test_performance.py -s` to see printed measurements.
Run with `pytest -m "not performance"` to SKIP these (useful for fast dev loops).
"""
import time
import numpy as np
import pytest


# --- Tunables (interview-relevant pattern: configurable, not magic numbers) ---
WARMUP_ITERS = 3        # discard the first N calls — they include lazy init / JIT
MEASURE_ITERS = 20      # number of measured runs
LATENCY_BUDGET_MS = 500 # soft ceiling: catches catastrophic slowdowns only
MIN_THROUGHPUT_ITEMS_PER_SEC = 10  # soft floor: model not totally broken


def _measure_latency_ms(predict_fn, x, warmup=WARMUP_ITERS, iters=MEASURE_ITERS):
    """Return dict of (mean, p50, p95) latency in milliseconds."""
    for _ in range(warmup):                 # warmup — first runs include lazy init
        predict_fn(x)
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        predict_fn(x)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)
    return {
        "mean": float(np.mean(timings)),
        "p50":  float(np.percentile(timings, 50)),
        "p95":  float(np.percentile(timings, 95)),
        "min":  float(np.min(timings)),
        "max":  float(np.max(timings)),
    }


@pytest.mark.performance
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
def test_latency_within_budget(predictor, batch_size, capsys):
    """
    Latency p95 must be under LATENCY_BUDGET_MS at every batch size.
    Reports per-batch latency stats so we can see the trend.
    """
    rng = np.random.default_rng(seed=batch_size)
    x = rng.random(size=(batch_size, 28, 28), dtype=np.float32)

    stats = _measure_latency_ms(predictor.predict, x)

    # Print measurements (visible with `pytest -s`)
    with capsys.disabled():
        print(f"\n  [{predictor.name:5s} bs={batch_size:>4d}]  "
              f"mean={stats['mean']:6.2f}ms  "
              f"p50={stats['p50']:6.2f}ms  "
              f"p95={stats['p95']:6.2f}ms  "
              f"min={stats['min']:6.2f}ms  "
              f"max={stats['max']:6.2f}ms")

    assert stats["p95"] < LATENCY_BUDGET_MS, (
        f"{predictor.name} batch={batch_size}: "
        f"p95 latency {stats['p95']:.1f}ms exceeds budget {LATENCY_BUDGET_MS}ms"
    )


@pytest.mark.performance
@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
def test_throughput_minimum(predictor, batch_size, capsys):
    """
    Throughput (items/sec) must exceed MIN_THROUGHPUT.
    Throughput should INCREASE with batch size (the GPU saturation curve).
    """
    rng = np.random.default_rng(seed=batch_size + 1000)
    x = rng.random(size=(batch_size, 28, 28), dtype=np.float32)

    # Warmup
    for _ in range(WARMUP_ITERS):
        predictor.predict(x)

    # Measure throughput over a wall-clock window
    iters = MEASURE_ITERS
    t0 = time.perf_counter()
    for _ in range(iters):
        predictor.predict(x)
    t1 = time.perf_counter()

    total_items = iters * batch_size
    elapsed_s = t1 - t0
    throughput = total_items / elapsed_s

    with capsys.disabled():
        print(f"\n  [{predictor.name:5s} bs={batch_size:>4d}]  "
              f"throughput={throughput:>9.1f} items/sec  "
              f"({elapsed_s*1000:.1f}ms total for {total_items} items)")

    assert throughput > MIN_THROUGHPUT_ITEMS_PER_SEC, (
        f"{predictor.name} batch={batch_size}: "
        f"throughput {throughput:.1f} items/sec below min {MIN_THROUGHPUT_ITEMS_PER_SEC}"
    )


@pytest.mark.performance
def test_batching_improves_throughput_at_scale(predictor, capsys):
    """
    The customer-relevant claim: batching helps when applied at a useful scale.
    Strict monotonicity across all batch sizes is NOT guaranteed on GPU backends
    due to dispatch overhead and cache effects — see notes in README.
    """
    batch_sizes = [1, 16, 128]
    throughputs = {}

    for bs in batch_sizes:
        rng = np.random.default_rng(seed=bs + 2000)
        x = rng.random(size=(bs, 28, 28), dtype=np.float32)
        for _ in range(WARMUP_ITERS):
            predictor.predict(x)
        t0 = time.perf_counter()
        for _ in range(MEASURE_ITERS):
            predictor.predict(x)
        t1 = time.perf_counter()
        throughputs[bs] = MEASURE_ITERS * bs / (t1 - t0)

    with capsys.disabled():
        print(f"\n  [{predictor.name}] throughput sweep:")
        for bs, tp in throughputs.items():
            print(f"      batch={bs:>4d}  ->  {tp:>9.1f} items/sec")
        # Optional info line: warn (don't fail) if intermediate dips occur
        if throughputs[16] < throughputs[1]:
            print(f"      NOTE: bs=16 ({throughputs[16]:.0f}) < bs=1 "
                  f"({throughputs[1]:.0f}) — likely dispatch-overhead crossover")

    # Customer-relevant invariant: large-batch throughput must exceed single-call
    assert throughputs[128] > throughputs[1], (
        f"{predictor.name}: bs=128 throughput {throughputs[128]:.0f} not greater than "
        f"bs=1 throughput {throughputs[1]:.0f} — batching not providing any benefit"
    )