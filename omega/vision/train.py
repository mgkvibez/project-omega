"""Train the tiny CNN on synthetic shapes, so the receipt is reproducible.

Synthetic and tiny on purpose: this proves the pipeline (data -> model -> served
predictions) rather than chasing an accuracy number on a dataset it did not earn.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Tuple

import numpy as np

from .model import SHAPE_CLASSES


def _canvas(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype="float32")


def draw_shape(kind: int, size: int, rng: np.random.Generator) -> np.ndarray:
    img = _canvas(size)
    margin = size // 6
    cx = int(rng.integers(margin, size - margin))
    cy = int(rng.integers(margin, size - margin))
    half = int(rng.integers(size // 6, size // 3))
    yy, xx = np.ogrid[:size, :size]
    if kind == 0:                                     # circle
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= half ** 2
    elif kind == 1:                                   # square
        mask = (np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= half)
    elif kind == 2:                                   # horizontal bar
        mask = (np.abs(xx - cx) <= half) & (np.abs(yy - cy) <= max(1, half // 4))
    else:                                             # vertical bar
        mask = (np.abs(xx - cx) <= max(1, half // 4)) & (np.abs(yy - cy) <= half)
    img[mask] = 1.0
    img += rng.normal(0, 0.04, size=(size, size)).astype("float32")
    return np.clip(img, 0.0, 1.0)


def make_dataset(n: int, size: int = 28, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    kinds = np.arange(n) % len(SHAPE_CLASSES)
    rng.shuffle(kinds)
    images = np.stack([draw_shape(int(k), size, rng) for k in kinds])
    return images[:, None, :, :], kinds.astype("int64")


def normalise(images: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    return ((images - mean) / std).astype("float32")


def train(out: str, epochs: int = 8, samples: int = 1200, batch_size: int = 32,
          size: int = 28, seed: int = 0, lr: float = 3e-3) -> dict:
    import torch
    import torch.nn.functional as F
    from .torch_model import TinyCNN

    torch.manual_seed(seed)
    x, y = make_dataset(samples, size=size, seed=seed)
    split = int(samples * 0.8)
    x_train, y_train = torch.from_numpy(normalise(x[:split])), torch.from_numpy(y[:split])
    x_val, y_val = torch.from_numpy(normalise(x[split:])), torch.from_numpy(y[split:])

    model = TinyCNN(len(SHAPE_CLASSES))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        losses = []
        for i in range(0, len(x_train), batch_size):
            idx = perm[i:i + batch_size]
            optimiser.zero_grad()
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            accuracy = float((model(x_val).argmax(1) == y_val).float().mean())
        print("epoch {} loss {:.4f} val_accuracy {:.3f}".format(epoch + 1, np.mean(losses), accuracy))

    model.eval()
    with torch.no_grad():
        final_accuracy = float((model(x_val).argmax(1) == y_val).float().mean())
    meta = {"classes": list(SHAPE_CLASSES), "version": "tiny-cnn-0.1",
            "val_accuracy": round(final_accuracy, 4), "size": size,
            "samples": samples, "epochs": epochs, "seed": seed,
            "train_seconds": round(time.perf_counter() - started, 2)}
    torch.save({"state_dict": model.state_dict(), "meta": meta}, out)
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="omega-train", description="Train the tiny CNN on synthetic shapes.")
    p.add_argument("--out", default="models/tinycnn.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--samples", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--meta", help="also write the metadata JSON here")
    args = p.parse_args(argv)
    if args.meta:
        import os
        os.makedirs(os.path.dirname(args.meta) or ".", exist_ok=True)
    meta = train(args.out, epochs=args.epochs, samples=args.samples, seed=args.seed)
    print(json.dumps(meta, indent=2))
    if args.meta:
        with open(args.meta, "w") as fh:
            json.dump(meta, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
