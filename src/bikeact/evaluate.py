"""Evaluation metrics for the skeleton classifier."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from bikeact.labels import CLASS_NAMES, NUM_CLASSES


@dataclass
class EvalResult:
    """Aggregate evaluation metrics for one pass over a dataset."""

    accuracy: float
    per_class_acc: dict[str, float]
    confusion: NDArray[np.int64] = field(repr=False)
    n_samples: int

    def format_report(self) -> str:
        lines = [f"accuracy: {self.accuracy:.4f}  (n={self.n_samples})", "per-class recall:"]
        lines += [f"  {name:>10}: {acc:.4f}" for name, acc in self.per_class_acc.items()]
        return "\n".join(lines)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> EvalResult:
    """Run ``model`` over ``loader`` and return accuracy + per-class recall."""
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for data, index_t, labels in loader:
            data = data.to(device)
            index_t = index_t.to(device)
            preds = model(data, index_t).argmax(dim=1).cpu().numpy()
            for gt, pred in zip(labels.numpy(), preds, strict=True):
                confusion[int(gt), int(pred)] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    per_class: dict[str, float] = {}
    for c in range(NUM_CLASSES):
        support = int(confusion[c].sum())
        per_class[CLASS_NAMES[c]] = float(confusion[c, c] / support) if support else float("nan")
    return EvalResult(
        accuracy=correct / total if total else 0.0,
        per_class_acc=per_class,
        confusion=confusion,
        n_samples=total,
    )
