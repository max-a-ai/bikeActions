"""Model construction, checkpoint loading, and single-sample inference."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from bikeact.labels import NUM_CLASSES
from bikeact.model import SkateFormer_
from bikeact.preprocess import preprocess_eval

# Architecture hyperparameters matching the bundled checkpoints. These MUST NOT
# change or the checkpoints will fail to load.
MODEL_ARGS: dict[str, Any] = {
    "num_classes": NUM_CLASSES,
    "num_people": 1,
    "num_points": 20,
    "kernel_size": 7,
    "num_heads": 32,
    "attn_drop": 0.5,
    "head_drop": 0.0,
    "rel": True,
    "drop_path": 0.2,
    "type_1_size": [8, 4],
    "type_2_size": [8, 5],
    "type_3_size": [8, 4],
    "type_4_size": [8, 5],
    "mlp_ratio": 1.0,
    "index_t": True,
}


def build_model() -> torch.nn.Module:
    """Instantiate an untrained SkateFormer with the checkpoint architecture."""
    model: torch.nn.Module = SkateFormer_(**MODEL_ARGS)  # type: ignore[no-untyped-call]
    return model


def load_model(checkpoint: str | Path, device: str = "cpu") -> torch.nn.Module:
    """Build the model and load a checkpoint's weights onto ``device``."""
    model = build_model()
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and "head.weight" not in state:
        state = state["model"]
    # Strip any DataParallel "module." prefix.
    clean = OrderedDict((k.split("module.")[-1], v) for k, v in state.items())
    model.load_state_dict(clean)
    model.to(device)
    model.eval()
    return model


def predict(
    model: torch.nn.Module,
    skeletons: NDArray[np.float64],
    data_type: str = "j",
    device: str = "cpu",
) -> tuple[int, NDArray[np.float32]]:
    """Classify one ``(T, 20, 3)`` skeleton sequence.

    Returns the predicted class index and the softmax probability vector.
    """
    data, index_t = preprocess_eval(skeletons, data_type=data_type)
    x = torch.from_numpy(data).unsqueeze(0).to(device)  # (1, C, T, V, M)
    t = torch.from_numpy(index_t).unsqueeze(0).to(device)  # (1, T)
    with torch.no_grad():
        logits = model(x, t)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
    return int(probs.argmax()), probs
