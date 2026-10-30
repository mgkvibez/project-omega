import pathlib, sys, threading
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from omega.obs.metrics import Counter, Gauge, Histogram, Registry, build_default_registry


def test_counter_increments_and_reads_back():
    c = Counter("x_total", "x")
    c.inc(); c.inc(amount=2)
    assert c.value() == 3


def test_counter_rejects_negative():
    c = Counter("x_total", "x")
    with pytest.raises(ValueError):
        c.inc(amount=-1)


def test_counter_labels_must_match_declared():
    c = Counter("x_total", "x", ("route",))
    c.inc({"route": "/a"})
    with pytest.raises(ValueError):
        c.inc({"path": "/a"})


def test_gauge_set_inc_dec():
    g = Gauge("g", "g")
    g.set(5); g.inc(); g.dec(2)
    assert g.value() == 4


def test_histogram_buckets_are_cumulative():
    h = Histogram("h", "h", buckets=(1, 2, 5))
    for value in (0.5, 1.5, 4.9, 9.0):
        h.observe(value)
    lines = h.expose()
    buckets = [l for l in lines if "_bucket" in l]
    assert buckets[0].endswith(" 1")          # <=1: one sample
    assert buckets[1].endswith(" 2")          # <=2: two samples
    assert buckets[2].endswith(" 3")          # <=5: three samples
    assert buckets[3].endswith(" 4")          # +Inf: all four
    assert any(l.startswith("h_count 4") for l in lines)


def test_histogram_sum_and_count():
    h = Histogram("h", "h")
    h.observe(0.25); h.observe(0.75)
    assert h.count() == 2 and h.total() == pytest.approx(1.0)


def test_render_has_help_and_type_lines():
    reg = Registry(); reg.counter("a_total", "counter help"); reg.gauge("b", "gauge help")
    text = reg.render()
    assert "# TYPE a_total counter" in text and "# HELP b gauge help" in text


def test_render_formats_labels_sorted():
    c = Counter("x_total", "x", ("b", "a"))
    c.inc({"b": "2", "a": "1"})
    assert 'x_total{a="1",b="2"} 1' in "\n".join(c.expose())


def test_empty_counter_still_exposes_a_zero():
    assert Counter("z_total", "z").expose()[-1] == "z_total 0"


def test_registry_returns_the_same_metric_for_one_name():
    reg = Registry()
    assert reg.counter("dup_total", "d") is reg.counter("dup_total", "d")


def test_counter_is_thread_safe():
    c = Counter("t_total", "t")
    def work():
        for _ in range(500):
            c.inc()
    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert c.value() == 4000


def test_default_registry_promises_the_documented_metrics():
    reg = build_default_registry()
    names = set()
    for line in reg.render().splitlines():
        if line.startswith("# TYPE"):
            names.add(line.split()[2])
    assert {"omega_http_requests_total", "omega_http_request_duration_seconds",
            "omega_predictions_total", "omega_predict_duration_seconds",
            "omega_predict_batch_size", "omega_model_ready"} <= names
