"""Structured JSON logs with a request id that follows the request."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any, Dict, Optional

_REQUEST_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("omega_request_id", default=None)
_EXTRA_FIELDS = ("method", "path", "status", "duration_ms", "request_id", "model_version", "batch_size", "outcome")


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:16]
    _REQUEST_ID.set(rid)
    return rid


def set_request_id(rid: Optional[str]) -> None:
    _REQUEST_ID.set(rid)


def current_request_id() -> Optional[str]:
    return _REQUEST_ID.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": round(record.created, 3),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = current_request_id()
        if rid:
            payload["request_id"] = rid
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure(level: str = "INFO", stream=None) -> logging.Logger:
    logger = logging.getLogger("omega")
    logger.setLevel(level.upper())
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("omega")
