"""The actual vision model: a small CNN, deliberately small enough to be honest.

Two convolutions and a linear layer. It classifies four synthetic shapes. It is
not an ImageNet model and the README does not claim it is.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:            # pragma: no cover - exercised by the ml CI job
    raise ImportError("torch is required for the torch model path; install the 'ml' extra") from exc


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(self.dropout(x))


class TorchPredictor:
    source = "torch"

    def __init__(self, checkpoint: str):
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        meta = blob.get("meta", {})
        self.classes: Tuple[str, ...] = tuple(meta.get("classes", ("circle", "square", "bar-horizontal", "bar-vertical")))
        self.version: str = str(meta.get("version", "tiny-cnn-0.1"))
        self.val_accuracy = meta.get("val_accuracy")
        self.model = TinyCNN(len(self.classes))
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

    def predict(self, batch: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(batch, dtype="float32"))
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()
