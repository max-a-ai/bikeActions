"""Reproducibility guards for the preprocessing pipeline and bundled checkpoint."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest
import torch

from bikeact.dataset import load_sample
from bikeact.infer import load_model, predict
from bikeact.labels import NUM_CLASSES
from bikeact.preprocess import TIME_STEPS, preprocess_eval

DATA_DIR = Path("data/bikeScenes_test")
BONE_CKPT = Path("checkpoints/skateformer_bone.pt")


def test_preprocess_eval_shapes() -> None:
    sk = np.random.randn(31, 20, 3).astype(np.float64)
    data, index_t = preprocess_eval(sk, data_type="j")
    assert data.shape == (3, TIME_STEPS, 20, 1)
    assert index_t.shape == (TIME_STEPS,)
    assert data.dtype == np.float32


def test_preprocess_eval_deterministic() -> None:
    sk = np.random.randn(20, 20, 3).astype(np.float64)
    a, _ = preprocess_eval(sk, data_type="b")
    b, _ = preprocess_eval(sk, data_type="b")
    assert np.array_equal(a, b)


@pytest.mark.skipif(not BONE_CKPT.exists(), reason="bone checkpoint not present")  # type: ignore[untyped-decorator]
def test_bone_checkpoint_accuracy() -> None:
    """The bundled bone checkpoint + our preprocessing must reproduce high accuracy.

    Guards against silent drift in the preprocessing or the label mapping.
    """
    model = load_model(BONE_CKPT, device="cpu")
    files = sorted(glob.glob(str(DATA_DIR / "*.json")))[:200]
    correct = total = 0
    for f in files:
        loaded = load_sample(Path(f))
        if loaded is None:
            continue
        skeletons, gt = loaded
        pred, _ = predict(model, skeletons, data_type="b", device="cpu")
        total += 1
        correct += int(pred == gt)
    assert total > 0
    assert correct / total > 0.85, f"bone accuracy dropped to {correct / total:.3f}"


def test_num_classes() -> None:
    assert NUM_CLASSES == 5
    model = load_model(BONE_CKPT, device="cpu") if BONE_CKPT.exists() else None
    if model is not None:
        with torch.no_grad():
            data, index_t = preprocess_eval(np.random.randn(10, 20, 3).astype(np.float64), data_type="b")
            out = model(torch.from_numpy(data).unsqueeze(0), torch.from_numpy(index_t).unsqueeze(0))
        assert out.shape == (1, NUM_CLASSES)
