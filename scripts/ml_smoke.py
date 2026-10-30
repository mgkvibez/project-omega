"""The receipt: train the model, then prove the served path uses the real weights."""

from __future__ import annotations

import json
import sys

from fastapi.testclient import TestClient

from omega.config import Settings
from omega.serving.app import create_app


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    checkpoint = argv[0] if argv else "models/tinycnn.pt"
    min_accuracy = float(argv[1]) if len(argv) > 1 else 0.9
    meta_path = argv[2] if len(argv) > 2 else "models/tinycnn.meta.json"

    with open(meta_path) as fh:
        meta = json.load(fh)
    if meta["val_accuracy"] < min_accuracy:
        print("FAILED: val_accuracy {} is below {}".format(meta["val_accuracy"], min_accuracy))
        return 1

    client = TestClient(create_app(settings=Settings(checkpoint=checkpoint, log_level="CRITICAL")))
    info = client.get("/model").json()
    if info["source"] != "torch":
        print("FAILED: serving fell back to {!r} instead of the checkpoint".format(info["source"]))
        return 1
    image = [[float((x + y) % 2) for x in range(28)] for y in range(28)]
    response = client.post("/predict", json={"images": [image], "top_k": 3})
    if response.status_code != 200:
        print("FAILED: predict returned", response.status_code, response.text[:200])
        return 1
    body = response.json()
    print("ml smoke ok | source {} | val_accuracy {} | latency {} ms | top {} {}".format(
        info["source"], meta["val_accuracy"], body["latency_ms"],
        body["predictions"][0][0]["label"], body["predictions"][0][0]["score"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
