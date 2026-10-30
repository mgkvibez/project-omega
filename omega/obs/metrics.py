"""A small, thread-safe metrics registry that renders Prometheus text format.

Written rather than imported so the exposition format, the bucket arithmetic and
the label handling are all reviewable in one file.
"""

from __future__ import annotations

import math
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Labels = Optional[Dict[str, str]]
DEFAULT_BUCKETS: Tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _key(labels: Labels) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))


def _fmt(labels: Labels) -> str:
    if not labels:
        return ""
    inner = ",".join('{}="{}"'.format(k, v) for k, v in sorted(labels.items()))
    return "{" + inner + "}"


def _num(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


class Metric:
    kind = "untyped"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()):
        self.name, self.help, self.labelnames = name, help_text, tuple(labelnames)
        self._lock = threading.Lock()

    def _check(self, labels: Labels) -> None:
        keys = set((labels or {}).keys())
        if keys and keys != set(self.labelnames):
            raise ValueError("{}: labels {} do not match {}".format(self.name, sorted(keys), list(self.labelnames)))

    def expose(self) -> List[str]:
        raise NotImplementedError

    def _header(self) -> List[str]:
        return ["# HELP {} {}".format(self.name, self.help), "# TYPE {} {}".format(self.name, self.kind)]


class Counter(Metric):
    kind = "counter"

    def __init__(self, name, help_text, labelnames=()):
        super().__init__(name, help_text, labelnames)
        self._values: Dict[Tuple, float] = {}

    def inc(self, labels: Labels = None, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("counters only increase")
        self._check(labels)
        with self._lock:
            key = _key(labels)
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, labels: Labels = None) -> float:
        with self._lock:
            return self._values.get(_key(labels), 0.0)

    def expose(self) -> List[str]:
        out = self._header()
        with self._lock:
            items = sorted(self._values.items())
        for key, value in items:
            labels = dict(key) if key else None
            out.append("{}{} {}".format(self.name, _fmt(labels), _num(value)))
        if not items:
            out.append("{} 0".format(self.name))
        return out


class Gauge(Metric):
    kind = "gauge"

    def __init__(self, name, help_text, labelnames=()):
        super().__init__(name, help_text, labelnames)
        self._values: Dict[Tuple, float] = {}

    def set(self, value: float, labels: Labels = None) -> None:
        self._check(labels)
        with self._lock:
            self._values[_key(labels)] = float(value)

    def inc(self, amount: float = 1.0, labels: Labels = None) -> None:
        self._check(labels)
        with self._lock:
            key = _key(labels)
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, labels: Labels = None) -> None:
        self.inc(-amount, labels)

    def value(self, labels: Labels = None) -> float:
        with self._lock:
            return self._values.get(_key(labels), 0.0)

    def expose(self) -> List[str]:
        out = self._header()
        with self._lock:
            items = sorted(self._values.items())
        for key, value in items:
            out.append("{}{} {}".format(self.name, _fmt(dict(key) if key else None), _num(value)))
        if not items:
            out.append("{} 0".format(self.name))
        return out


class Histogram(Metric):
    kind = "histogram"

    def __init__(self, name, help_text, labelnames=(), buckets: Sequence[float] = DEFAULT_BUCKETS):
        super().__init__(name, help_text, labelnames)
        self.buckets = tuple(sorted(float(b) for b in buckets))
        self._counts: Dict[Tuple, List[int]] = {}
        self._sums: Dict[Tuple, float] = {}
        self._totals: Dict[Tuple, int] = {}

    def observe(self, value: float, labels: Labels = None) -> None:
        self._check(labels)
        key = _key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * len(self.buckets))
            for i, edge in enumerate(self.buckets):
                if value <= edge:
                    counts[i] += 1
                    break
            self._sums[key] = self._sums.get(key, 0.0) + float(value)
            self._totals[key] = self._totals.get(key, 0) + 1

    def count(self, labels: Labels = None) -> int:
        with self._lock:
            return self._totals.get(_key(labels), 0)

    def total(self, labels: Labels = None) -> float:
        with self._lock:
            return self._sums.get(_key(labels), 0.0)

    def expose(self) -> List[str]:
        out = self._header()
        with self._lock:
            keys = sorted(set(self._counts) | set(self._totals))
            snapshot = {k: (list(self._counts.get(k, [])), self._sums.get(k, 0.0), self._totals.get(k, 0))
                        for k in keys}
        for key in keys:
            counts, total_sum, total_count = snapshot[key]
            labels = dict(key) if key else {}
            # Prometheus buckets are cumulative, so each line reports all values <= edge.
            running = 0
            for i, edge in enumerate(self.buckets):
                running += counts[i]
                merged = dict(labels, le=_num(edge))
                out.append("{}_bucket{} {}".format(self.name, _fmt(merged), running))
            merged = dict(labels, le="+Inf")
            out.append("{}_bucket{} {}".format(self.name, _fmt(merged), total_count))
            out.append("{}_sum{} {}".format(self.name, _fmt(labels or None), _num(total_sum)))
            out.append("{}_count{} {}".format(self.name, _fmt(labels or None), total_count))
        return out


class Registry:
    def __init__(self):
        self._metrics: List[Metric] = []
        self._by_name: Dict[str, Metric] = {}
        self._lock = threading.Lock()

    def _add(self, metric: Metric) -> Metric:
        with self._lock:
            existing = self._by_name.get(metric.name)
            if existing is not None:
                return existing
            self._metrics.append(metric)
            self._by_name[metric.name] = metric
            return metric

    def counter(self, name, help_text, labelnames=()) -> Counter:
        return self._add(Counter(name, help_text, labelnames))

    def gauge(self, name, help_text, labelnames=()) -> Gauge:
        return self._add(Gauge(name, help_text, labelnames))

    def histogram(self, name, help_text, labelnames=(), buckets=DEFAULT_BUCKETS) -> Histogram:
        return self._add(Histogram(name, help_text, labelnames, buckets))

    def get(self, name: str) -> Optional[Metric]:
        return self._by_name.get(name)

    def render(self) -> str:
        lines: List[str] = []
        for metric in self._metrics:
            lines.extend(metric.expose())
        return "\n".join(lines) + "\n"


def build_default_registry() -> Registry:
    """The metrics this service actually promises to expose."""
    reg = Registry()
    reg.counter("omega_http_requests_total", "HTTP requests handled, by route, method and status.",
                ("route", "method", "status"))
    reg.histogram("omega_http_request_duration_seconds", "HTTP request duration in seconds.",
                  ("route",), buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0))
    reg.counter("omega_predictions_total", "Predictions served, by outcome.", ("outcome",))
    reg.histogram("omega_predict_duration_seconds", "Time spent inside the model call, per batch.",
                  buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
    reg.gauge("omega_predict_batch_size", "Size of the last batch scored.")
    reg.gauge("omega_model_ready", "1 when a predictor is loaded and warm.")
    reg.gauge("omega_model_load_seconds", "Seconds spent loading the predictor at startup.")
    return reg
