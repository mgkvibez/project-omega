"""The app factory. Everything is injected so tests never load a real model."""

from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from .. import __version__
from ..config import Settings
from ..obs.logging import configure, get_logger, new_request_id, set_request_id
from ..obs.metrics import Registry, build_default_registry
from ..vision.model import Predictor, load_predictor, torch_available
from ..vision.preprocess import to_batch
from .schemas import (HealthResponse, ModelInfo, PredictRequest, PredictResponse,
                      Prediction, ReadyResponse)


def create_app(predictor: Optional[Predictor] = None, settings: Optional[Settings] = None,
               registry: Optional[Registry] = None, load_seconds: float = 0.0,
               load_reason: str = "injected", load_if_missing: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = registry or build_default_registry()
    logger = configure(settings.log_level)

    if predictor is None and load_if_missing:
        started = time.perf_counter()
        predictor, load_reason = load_predictor(settings.checkpoint)
        load_seconds = time.perf_counter() - started
        logger.info("predictor loaded", extra={"model_version": predictor.version, "outcome": predictor.source})
    elif predictor is None:
        load_reason = load_reason if load_reason != "injected" else "no predictor was supplied"

    registry.get("omega_model_ready").set(1 if predictor else 0)
    registry.get("omega_model_load_seconds").set(round(load_seconds, 6))

    app = FastAPI(title=settings.service, version=__version__,
                  description="A small vision model with an instrumented inference path.")
    app.state.predictor = predictor
    app.state.settings = settings
    app.state.registry = registry
    app.state.load_reason = load_reason

    @app.middleware("http")
    async def instrument(request: Request, call_next):
        # Never count the scrape itself: it would inflate every dashboard it feeds.
        if request.url.path == "/metrics":
            return await call_next(request)
        rid = request.headers.get("x-request-id") or new_request_id()
        set_request_id(rid)
        started = time.perf_counter()

        def route_label() -> str:
            # Resolved after routing: scope["route"] is only set once a route matched.
            matched = getattr(request.scope.get("route"), "path", None)
            return matched or "unmatched"

        try:
            response = await call_next(request)
        except Exception:
            # An unhandled exception still happened, so it still gets counted and logged.
            duration = time.perf_counter() - started
            route = route_label()
            registry.get("omega_http_requests_total").inc(
                {"route": route, "method": request.method, "status": "500"})
            registry.get("omega_http_request_duration_seconds").observe(duration, {"route": route})
            logger.error("request failed", extra={"method": request.method, "path": route,
                                                  "status": 500,
                                                  "duration_ms": round(duration * 1000, 3)})
            raise
        duration = time.perf_counter() - started
        route = route_label()
        registry.get("omega_http_requests_total").inc(
            {"route": route, "method": request.method, "status": str(response.status_code)})
        registry.get("omega_http_request_duration_seconds").observe(duration, {"route": route})
        logger.info("request", extra={"method": request.method, "path": route,
                                      "status": response.status_code,
                                      "duration_ms": round(duration * 1000, 3)})
        response.headers["x-request-id"] = rid
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness: the process is up. Says nothing about the model."""
        return HealthResponse(status="ok", service=settings.service, version=__version__)

    @app.get("/ready", response_model=ReadyResponse)
    def ready(request: Request):
        from fastapi.responses import JSONResponse
        predictor = request.app.state.predictor
        body = ReadyResponse(ready=predictor is not None,
                            model_version=getattr(predictor, "version", "none"),
                            source=getattr(predictor, "source", "none"),
                            reason=request.app.state.load_reason)
        return JSONResponse(status_code=200 if body.ready else 503, content=body.model_dump())

    @app.get("/model", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        predictor = request.app.state.predictor
        return ModelInfo(service=settings.service, version=__version__,
                         source=getattr(predictor, "source", "none"),
                         classes=list(getattr(predictor, "classes", ())),
                         image_size=settings.image_size, channels=settings.channels,
                         max_batch=settings.max_batch,
                         load_seconds=registry.get("omega_model_load_seconds").value(),
                         torch_available=torch_available())

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        predictor = request.app.state.predictor
        if predictor is None:
            registry.get("omega_predictions_total").inc({"outcome": "not_ready"})
            raise HTTPException(status_code=503, detail="no predictor is loaded")
        try:
            batch = to_batch(payload.images, size=settings.image_size,
                             channels=settings.channels, max_batch=settings.max_batch)
        except ValueError as exc:
            registry.get("omega_predictions_total").inc({"outcome": "invalid"})
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        started = time.perf_counter()
        try:
            probabilities = predictor.predict(batch)
        except Exception as exc:                        # a model failure is a 500, not a 200 with junk
            registry.get("omega_predictions_total").inc({"outcome": "error"})
            logger.error("prediction failed", exc_info=exc)
            raise HTTPException(status_code=500, detail="prediction failed") from exc
        elapsed = time.perf_counter() - started

        registry.get("omega_predictions_total").inc({"outcome": "ok"}, amount=batch.shape[0])
        registry.get("omega_predict_duration_seconds").observe(elapsed)
        registry.get("omega_predict_batch_size").set(batch.shape[0])

        classes = list(getattr(predictor, "classes", ()))
        rows = []
        for row in probabilities:
            order = row.argsort()[::-1][:payload.top_k]
            rows.append([Prediction(label=classes[i] if i < len(classes) else "class-{}".format(i),
                                    score=round(float(row[i]), 6)) for i in order])
        return PredictResponse(predictions=rows, model_version=predictor.version,
                               source=getattr(predictor, "source", "unknown"),
                               batch_size=int(batch.shape[0]), latency_ms=round(elapsed * 1000, 3))

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request) -> str:
        return request.app.state.registry.render()

    return app
