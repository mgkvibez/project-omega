"""Predictor interface plus a deterministic stub, so serving never needs torch.

The serving layer depends on this protocol, not on PyTorch. That is why the API
tests run in a second, in CI, without a 200MB download — and why the torch path
is exercised in its own job rather than assumed.
"""

from __future__ import annotations

import os
from typing import List, Optional, Protocol, Tuple

import numpy as np

SHAPE_CLASSES: Tuple[str, ...] = ("circle", "square", "bar-horizontal", "bar-vertical")


class Predictor(Protocol):
    version: str
    classes: Tuple[str, ...]
    source: str

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """(N, C, H, W) batch in, (N, classes) probabilities out."""


class StubPredictor:
    """Deterministic stand-in: same image in, same distribution out.

    It is not a model and never pretends to be one: `source` says "stub", the
    model endpoint reports it, and the README says so too.
    """

    version = "stub-0"
    source = "stub"
    classes = SHAPE_CLASSES

    def predict(self, batch: np.ndarray) -> np.ndarray:
        x = np.asarray(batch, dtype="float64")
        if x.ndim != 4:
            raise ValueError("expected a 4D batch, got shape {}".format(x.shape))
        flat = x.reshape(x.shape[0], -1)
        mean = flat.mean(axis=1)
        spread = flat.std(axis=1)
        h_energy = np.abs(np.diff(x, axis=2)).mean(axis=(1, 2, 3))
        v_energy = np.abs(np.diff(x, axis=3)).mean(axis=(1, 2, 3))
        logits = np.stack([-np.abs(mean), spread, h_energy, v_energy], axis=1)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits * 4.0)
        return exp / exp.sum(axis=1, keepdims=True)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def load_predictor(checkpoint: str = "") -> Tuple[Predictor, str]:
    """Return (predictor, reason). The reason is shown by /model, never swallowed."""
    if not checkpoint:
        return StubPredictor(), "no checkpoint configured"
    if not os.path.isfile(checkpoint):
        return StubPredictor(), "checkpoint {!r} not found".format(checkpoint)
    if not torch_available():
        return StubPredictor(), "torch is not installed in this environment"
    from .torch_model import TorchPredictor
    return TorchPredictor(checkpoint), "loaded from {}".format(checkpoint)
