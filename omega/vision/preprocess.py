"""Turn JSON pixels into a model-ready batch. Pure numpy, fully predictable.

Validation lives here and raises ValueError; the HTTP layer turns that into a 422.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Union

Image = Union[Sequence, Iterable]


def _as_array(pixels: Image) -> np.ndarray:
    arr = np.asarray(pixels, dtype="float32")
    if arr.ndim == 1:
        side = int(round(arr.size ** 0.5))
        if side * side != arr.size:
            raise ValueError("flat image of {} values is not a square".format(arr.size))
        arr = arr.reshape(side, side)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError("image must be 2D (H,W) or 3D (C,H,W), got shape {}".format(arr.shape))
    if not np.isfinite(arr).all():
        raise ValueError("image contains NaN or infinity")
    return arr


def resize(image: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize: deterministic, dependency-free, good enough at 28x28."""
    c, h, w = image.shape
    if (h, w) == (size, size):
        return image
    rows = (np.arange(size) * (h / size)).astype(int).clip(0, h - 1)
    cols = (np.arange(size) * (w / size)).astype(int).clip(0, w - 1)
    return image[:, rows][:, :, cols]


def normalize(image: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Scale to a predictable range. Input is expected on 0..255 or 0..1."""
    peak = float(image.max()) if image.size else 0.0
    scaled = image / 255.0 if peak > 1.0 else image
    if std == 0:
        raise ValueError("std must be non-zero")
    return ((scaled - mean) / std).astype("float32")


def to_tensor(pixels: Image, size: int = 28, channels: int = 1,
              mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    arr = _as_array(pixels)
    c = arr.shape[0]
    if c != channels:
        if channels == 1 and c in (3, 4):
            arr = arr[:3].mean(axis=0, keepdims=True)      # collapse colour to luminance
        else:
            raise ValueError("expected {} channel(s), got {}".format(channels, c))
    return normalize(resize(arr, size), mean=mean, std=std)


def to_batch(images: List[Image], size: int = 28, channels: int = 1, max_batch: int = 32) -> np.ndarray:
    if not images:
        raise ValueError("batch is empty")
    if len(images) > max_batch:
        raise ValueError("batch of {} exceeds the limit of {}".format(len(images), max_batch))
    # Every image is resized to (size, size) and channel-checked first, so a
    # batch may mix input sizes and still stack cleanly.
    tensors = [to_tensor(img, size=size, channels=channels) for img in images]
    return np.stack(tensors, axis=0)
