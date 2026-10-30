"""Latency benchmark with nearest-rank percentiles: no scipy, no guesswork."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Callable, Dict, List, Sequence

import numpy as np


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. For p=50 and 100 samples this is the 50th smallest."""
    if not values:
        raise ValueError("no samples")
    if not 0 < p <= 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[rank - 1]


def summarise(samples: Sequence[float], batch_size: int = 1) -> Dict[str, float]:
    total = sum(samples)
    return {
        "runs": len(samples),
        "batch_size": batch_size,
        "mean_ms": round(1000 * total / len(samples), 3),
        "p50_ms": round(1000 * percentile(samples, 50), 3),
        "p95_ms": round(1000 * percentile(samples, 95), 3),
        "p99_ms": round(1000 * percentile(samples, 99), 3),
        "throughput_images_per_s": round(batch_size * len(samples) / total, 1) if total else 0.0,
    }


def benchmark(predict: Callable[[np.ndarray], np.ndarray], batch: np.ndarray,
              runs: int = 30, warmup: int = 3, clock: Callable[[], float] = time.perf_counter) -> Dict[str, float]:
    for _ in range(warmup):
        predict(batch)
    samples: List[float] = []
    for _ in range(runs):
        started = clock()
        predict(batch)
        samples.append(clock() - started)
    return summarise(samples, batch_size=int(batch.shape[0]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="omega-bench", description="Latency benchmark for the served predictor.")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--size", type=int, default=28)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    from .model import load_predictor
    predictor, reason = load_predictor(args.checkpoint)
    batch = np.random.default_rng(0).normal(0, 1, (args.batch, 1, args.size, args.size)).astype("float32")
    result = benchmark(predictor.predict, batch, runs=args.runs)
    result.update({"source": predictor.source, "model_version": predictor.version, "reason": reason})
    print(json.dumps(result, indent=2) if args.json else
          "{source} {model_version} batch={batch_size} p50={p50_ms}ms p95={p95_ms}ms "
          "p99={p99_ms}ms throughput={throughput_images_per_s}/s".format(**result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
