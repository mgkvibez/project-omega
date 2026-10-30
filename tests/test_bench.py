import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from omega.vision.bench import benchmark, percentile, summarise


def test_percentile_edges():
    values = [3.0, 1.0, 2.0, 4.0]
    assert percentile(values, 50) == 2.0
    assert percentile(values, 100) == 4.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_rejects_nonsense():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 0)


def test_summarise_math_is_exact():
    result = summarise([0.01, 0.02, 0.03, 0.04], batch_size=2)
    assert result["mean_ms"] == pytest.approx(25.0)
    assert result["p50_ms"] == pytest.approx(20.0)
    assert result["p95_ms"] == pytest.approx(40.0)
    assert result["throughput_images_per_s"] == pytest.approx(2 * 4 / 0.1, rel=1e-3)


def test_benchmark_warms_up_then_measures():
    calls = []
    def predict(batch):
        calls.append(1)
        return np.zeros((len(batch), 4))
    ticks = iter([i * 0.01 for i in range(100)])
    benchmark(predict, np.zeros((2, 1, 28, 28)), runs=5, warmup=2, clock=lambda: next(ticks))
    assert len(calls) == 7          # 2 warmup + 5 measured


def test_benchmark_reports_tens_of_milliseconds_for_a_stubbed_clock():
    def predict(batch):
        return np.zeros((len(batch), 4))
    ticks = iter([i * 0.012 for i in range(100)])
    result = benchmark(predict, np.zeros((1, 1, 28, 28)), runs=10, warmup=0, clock=lambda: next(ticks))
    assert result["p50_ms"] == pytest.approx(12.0, rel=1e-6)
    assert result["runs"] == 10
