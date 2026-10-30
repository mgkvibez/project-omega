"""Settings from the environment, with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    service: str = "omega-vision"
    model_version: str = "stub-0"
    checkpoint: str = ""
    image_size: int = 28
    channels: int = 1
    max_batch: int = 32
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        base = cls(
            service=os.environ.get("OMEGA_SERVICE", "omega-vision"),
            model_version=os.environ.get("OMEGA_MODEL_VERSION", "stub-0"),
            checkpoint=os.environ.get("OMEGA_CHECKPOINT", ""),
            image_size=_int("OMEGA_IMAGE_SIZE", 28),
            channels=_int("OMEGA_CHANNELS", 1),
            max_batch=_int("OMEGA_MAX_BATCH", 32),
            log_level=os.environ.get("OMEGA_LOG_LEVEL", "INFO"),
        )
        return cls(**{**base.__dict__, **overrides}) if overrides else base
