"""Dataset over the bundled skeleton JSON files, driven by split lists."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from bikeact.labels import SOURCE_TO_TRAIN_CLASS
from bikeact.preprocess import preprocess_eval, preprocess_train

SplitName = str  # "train" | "validation" | "test"


def _read_split(split_file: Path) -> list[str]:
    return [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]


def load_sample(path: Path) -> tuple[NDArray[np.float64], int] | None:
    """Load one JSON sample. Returns ``(skeletons, train_label)`` or ``None`` if
    the sample is malformed or its class is not one of the five used classes."""
    data = json.loads(path.read_text())
    source_label = int(data["label"])
    if source_label not in SOURCE_TO_TRAIN_CLASS:
        return None
    skeletons = np.asarray(data["skeletons"], dtype=np.float64)
    if skeletons.ndim != 3 or skeletons.shape[1:] != (20, 3) or skeletons.shape[0] < 2:
        return None
    return skeletons, SOURCE_TO_TRAIN_CLASS[source_label]


class SkeletonDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):  # type: ignore[misc]
    """Skeleton sequences for one split.

    Args:
        data_dir: folder of ``*.json`` sample files.
        split_file: text file listing sample basenames (one per line).
        data_type: ``"j"`` joint or ``"b"`` bone (letters combine).
        train: apply train-time augmentation when True, deterministic when False.
        repeat: dataset-length multiplier (augmentation resampling; train only).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path,
        data_type: str = "j",
        train: bool = False,
        repeat: int = 1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_type = data_type
        self.train = train
        self.repeat = max(1, repeat) if train else 1

        self.samples: list[tuple[NDArray[np.float64], int]] = []
        for name in _read_split(Path(split_file)):
            path = self.data_dir / name
            if not path.exists():
                continue
            loaded = load_sample(path)
            if loaded is not None:
                self.samples.append(loaded)
        if not self.samples:
            raise RuntimeError(f"No usable samples for split {split_file} in {data_dir}")

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        skeletons, label = self.samples[index % len(self.samples)]
        if self.train:
            data, index_t = preprocess_train(skeletons, data_type=self.data_type)
        else:
            data, index_t = preprocess_eval(skeletons, data_type=self.data_type)
        return torch.from_numpy(data), torch.from_numpy(index_t), label
