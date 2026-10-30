"""Request and response shapes. Validation errors become 422, never a traceback."""

from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    images: List[Any] = Field(..., description="One or more images: flat list, (H,W) or (C,H,W) nested lists.")
    top_k: int = Field(default=3, ge=1, le=10, description="How many labels to return per image.")


class Prediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    predictions: List[List[Prediction]]
    model_version: str
    source: str
    batch_size: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    model_version: str
    source: str
    reason: str


class ModelInfo(BaseModel):
    service: str
    version: str
    source: str
    classes: List[str]
    image_size: int
    channels: int
    max_batch: int
    load_seconds: float
    torch_available: bool
