#!/usr/bin/env bash
# Release gate for project-omega: run before every push that claims "done".
set -uo pipefail
fail() { echo "GATE FAILED: $1"; exit 1; }

echo "== check 1/5: install"
python3 -m pip install -q -e ".[dev]" || fail "install"

echo "== check 2/5: version consistency (pyproject vs module)"
pv=$(python3 - <<'PY'
import pathlib, re
try:
    import tomllib
    print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])
except ModuleNotFoundError:
    print(re.search(r'(?m)^version\s*=\s*"([^"]+)"', pathlib.Path("pyproject.toml").read_text()).group(1))
PY
)
mv=$(python3 -c "import omega;print(omega.__version__)")
[ "$pv" = "$mv" ] || fail "pyproject says $pv, module says $mv"

echo "== check 3/5: test suite"
python3 -m pytest tests -q || fail "tests"

echo "== check 4/5: API and CLI contract"
python3 - <<'PY' || fail "api contract"
import sys
from fastapi.testclient import TestClient
from omega.config import Settings
from omega.serving.app import create_app
from omega.vision.model import StubPredictor

client = TestClient(create_app(predictor=StubPredictor(), settings=Settings(log_level="CRITICAL")))
assert client.get("/health").status_code == 200
assert client.get("/ready").status_code == 200
image = [[0] * 28 for _ in range(28)]
body = client.post("/predict", json={"images": [image], "top_k": 2}).json()
assert len(body["predictions"][0]) == 2, body
assert client.post("/predict", json={"images": [[1, 2, 3]]}).status_code == 422
assert client.post("/predict", json={"images": [image] * 40}).status_code == 422
assert "omega_predictions_total" in client.get("/metrics").text
assert client.get("/health").headers.get("x-request-id")
print("api contract ok")
PY
python3 -c "import json,subprocess,sys; out=subprocess.run([sys.executable,'-m','omega.cli','info'],capture_output=True,text=True); d=json.loads(out.stdout); assert d['predictor']['source'] in ('stub','torch'); print('cli info ok, source', d['predictor']['source'])" || fail "cli info"

echo "== check 5/5: the wheel works in a clean venv WITHOUT torch"
wh=$(mktemp -d)
python3 -m build --wheel --outdir "$wh" -q >/tmp/omega-build.log 2>&1 || { echo "wheel build failed:"; tail -5 /tmp/omega-build.log; exit 1; }
venv=$(mktemp -d)/venv
python3 -m venv "$venv" || fail "venv"
"$venv"/bin/pip install -q "$wh"/*.whl || { echo "wheel install failed"; exit 1; }
# The gate needs its own HTTP client in the clean venv: the ambient env hiding this
# is exactly the class of bug this check exists to catch.
"$venv"/bin/pip install -q httpx || { echo "could not install the gate's HTTP client"; exit 1; }
"$venv"/bin/python -c "import torch" 2>/dev/null && fail "clean venv unexpectedly has torch; the torch-free check is void"
"$venv"/bin/python - <<'PY' || fail "torch-free wheel failed"
from fastapi.testclient import TestClient
from omega.serving.app import create_app
client = TestClient(create_app())
info = client.get("/model").json()
assert info["source"] == "stub", info
assert client.get("/ready").status_code == 200
assert client.post("/predict", json={"images": [[[0.0] * 28] * 28]}).status_code == 200
print("torch-free wheel serves, source:", info["source"])
PY

echo "ALL CHECKS PASSED"
