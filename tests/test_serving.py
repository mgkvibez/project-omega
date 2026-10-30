import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from fastapi.testclient import TestClient
from omega.config import Settings
from omega.obs.metrics import build_default_registry
from omega.serving.app import create_app
from omega.vision.model import StubPredictor

SETTINGS = Settings(log_level="CRITICAL", max_batch=4, image_size=28, channels=1)


def client(predictor=None, **kwargs):
    # An explicitly passed predictor must win; defaulting it inside kwargs silently
    # replaced it with the stub, which made a broken-predictor test pass for the wrong reason.
    chosen = predictor if predictor is not None else StubPredictor()
    app = create_app(predictor=chosen, settings=SETTINGS, registry=build_default_registry(), **kwargs)
    return TestClient(app), app


def blank(side=28):
    return [[0] * side for _ in range(side)]


def test_health_is_cheap_and_does_not_claim_the_model_works():
    c, _ = client()
    body = c.get("/health").json()
    assert c.get("/health").status_code == 200 and body["status"] == "ok"
    assert "model" not in body


def test_ready_reports_the_predictor_source():
    c, _ = client()
    body = c.get("/ready").json()
    assert c.get("/ready").status_code == 200 and body["source"] == "stub"


def test_ready_is_503_without_a_predictor():
    app = create_app(predictor=None, settings=SETTINGS, registry=build_default_registry(),
                     load_if_missing=False, load_reason="no predictor was supplied")
    c = TestClient(app)
    assert c.get("/ready").status_code == 503
    assert c.get("/ready").json()["ready"] is False


def test_predict_not_ready_is_503_not_a_fake_answer():
    app = create_app(predictor=None, settings=SETTINGS, registry=build_default_registry(), load_if_missing=False)
    c = TestClient(app)
    assert c.post("/predict", json={"images": [blank()]}).status_code == 503


def test_predict_returns_top_k_sorted():
    c, _ = client()
    body = c.post("/predict", json={"images": [blank()], "top_k": 2}).json()
    scores = [p["score"] for p in body["predictions"][0]]
    assert len(scores) == 2 and scores == sorted(scores, reverse=True)


def test_predict_batches_several_images():
    c, _ = client()
    body = c.post("/predict", json={"images": [blank(), blank()]}).json()
    assert body["batch_size"] == 2 and len(body["predictions"]) == 2


def test_bad_image_is_422_with_a_reason():
    c, _ = client()
    response = c.post("/predict", json={"images": [[1, 2, 3]]})
    assert response.status_code == 422 and "square" in response.json()["detail"]


def test_over_batch_limit_is_422():
    c, _ = client()
    response = c.post("/predict", json={"images": [blank()] * 5})
    assert response.status_code == 422 and "exceeds" in response.json()["detail"]


def test_model_failure_is_500_not_a_silent_wrong_answer():
    class Broken:
        version, source, classes = "broken", "test", ("a",)
        def predict(self, batch):
            raise RuntimeError("weights corrupted")
    c, _ = client(predictor=Broken())
    assert c.post("/predict", json={"images": [blank()]}).status_code == 500


def test_metrics_count_requests_by_route_status():
    c, _ = client()
    c.get("/health"); c.post("/predict", json={"images": [blank()]})
    text = c.get("/metrics").text
    assert 'omega_http_requests_total{method="GET",route="/health",status="200"}' in text
    assert 'omega_http_requests_total{method="POST",route="/predict",status="200"}' in text


def test_unknown_paths_do_not_create_unbounded_labels():
    c, _ = client()
    c.get("/does-not-exist-1234")
    text = c.get("/metrics").text
    assert 'route="unmatched"' in text and "does-not-exist-1234" not in text


def test_scraping_metrics_is_not_counted():
    c, _ = client()
    c.get("/metrics"); c.get("/metrics")
    assert 'route="/metrics"' not in c.get("/metrics").text


def test_predictions_total_counts_images_and_outcomes():
    c, _ = client()
    c.post("/predict", json={"images": [blank(), blank()]})
    c.post("/predict", json={"images": [[1, 2, 3]]})
    text = c.get("/metrics").text
    assert 'omega_predictions_total{outcome="ok"} 2' in text
    assert 'omega_predictions_total{outcome="invalid"} 1' in text


def test_predict_duration_is_observed():
    c, _ = client()
    c.post("/predict", json={"images": [blank()]})
    assert "omega_predict_duration_seconds_count 1" in c.get("/metrics").text


def test_batch_size_gauge_tracks_the_last_request():
    c, _ = client()
    c.post("/predict", json={"images": [blank(), blank(), blank()]})
    assert "omega_predict_batch_size 3" in c.get("/metrics").text


def test_request_id_is_echoed_when_supplied():
    c, _ = client()
    response = c.get("/health", headers={"x-request-id": "trace-42"})
    assert response.headers["x-request-id"] == "trace-42"


def test_request_id_is_generated_when_absent():
    c, _ = client()
    assert c.get("/health").headers["x-request-id"]


def test_model_endpoint_is_honest_about_what_is_loaded():
    c, _ = client()
    body = c.get("/model").json()
    assert body["source"] == "stub" and body["classes"][0] == "circle"
    assert isinstance(body["torch_available"], bool)


def test_metrics_endpoint_is_plain_text():
    c, _ = client()
    assert c.get("/metrics").headers["content-type"].startswith("text/plain")


def test_unhandled_route_exception_is_counted_as_a_500():
    app = create_app(predictor=StubPredictor(), settings=SETTINGS, registry=build_default_registry())

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    c = TestClient(app, raise_server_exceptions=False)
    assert c.get("/boom").status_code == 500
    assert 'omega_http_requests_total{method="GET",route="/boom",status="500"}' in c.get("/metrics").text
