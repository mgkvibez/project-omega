import io, json, logging, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from omega.obs.logging import JsonFormatter, configure, current_request_id, new_request_id, set_request_id


def render(record_fields=None, msg="hello"):
    stream = io.StringIO()
    logger = configure("INFO", stream=stream)
    extra = record_fields or {}
    logger.info(msg, extra=extra)
    return json.loads(stream.getvalue().strip())


def test_log_is_one_json_object_with_core_fields():
    payload = render()
    assert payload["message"] == "hello" and payload["level"] == "INFO"
    assert "ts" in payload and "iso" in payload


def test_extra_fields_are_promoted():
    payload = render({"status": 200, "duration_ms": 1.5, "method": "POST"})
    assert payload["status"] == 200 and payload["method"] == "POST"


def test_request_id_follows_the_context():
    rid = new_request_id()
    assert current_request_id() == rid
    assert render()["request_id"] == rid
    set_request_id(None)
    assert current_request_id() is None


def test_formatter_handles_exceptions():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord("omega", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert "RuntimeError" in payload["exception"]
