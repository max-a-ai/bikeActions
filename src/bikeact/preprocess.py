"""Skeleton preprocessing for SkateFormer input.

Faithful reimplementation of the reference ``feeder_bike`` pipeline so the
bundled checkpoints reproduce exactly:

* eval (deterministic): center on the spine joint, per-axis min-max to [-1, 1],
  linspace-resample to 64 frames, optional bone transform, reorder joints into
  body-part groups, emit ``(C, T, V, M)`` plus the ``index_t`` positional vector.
* train: the same, but with random view rotation/scale, random temporal
  sampling, and axis/joint dropout for augmentation.

Joint layout (20 joints): 0 pelvis, 1 spine (centering joint), 2 shoulder-centre,
3 head, 4-7 arm A, 8-11 arm B, 12-15 leg A, 16-19 leg B. Coordinates are (x, y, z)
with y vertical.
"""

from __future__ import annotations

import math
import random

import numpy as np
from numpy.typing import NDArray

TIME_STEPS = 64
NUM_JOINTS = 20
CENTER_JOINT = 1

# Bone parents (1-indexed pairs from the reference feeder): child -> parent.
_BONE_PAIRS: list[tuple[int, int]] = [
    (1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), (9, 3), (10, 9),
    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
]

# Reorder joints into body-part groups: right-arm, left-arm, right-leg, left-leg, torso.
_PARTITION_INDEX: NDArray[np.int64] = np.concatenate(
    [
        np.array([5, 6, 7, 8]) - 1,
        np.array([9, 10, 11, 12]) - 1,
        np.array([13, 14, 15, 16]) - 1,
        np.array([17, 18, 19, 20]) - 1,
        np.array([2, 3, 1, 4]) - 1,
    ]
).astype(np.int64)


def minmax_normalize(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-axis min-max of all points to [-1, 1] (the baseline normalization)."""
    flat = coords.reshape(-1, 3)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1.0
    scaled = (flat - mins) / rng * 2.0 - 1.0
    return scaled.reshape(coords.shape)


def _rand_view_transform(
    coords: NDArray[np.float64], agx: float, agy: float, s: float
) -> NDArray[np.float64]:
    """Rotate about x then y and scale (row-vector convention: X @ Ry @ Rx @ Ss)."""
    ax, ay = math.radians(agx), math.radians(agy)
    rx = np.asarray([[1, 0, 0], [0, math.cos(ax), math.sin(ax)], [0, -math.sin(ax), math.cos(ax)]])
    ry = np.asarray([[math.cos(ay), 0, -math.sin(ay)], [0, 1, 0], [math.sin(ay), 0, math.cos(ay)]])
    ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
    flat = np.dot(coords.reshape(-1, 3), np.dot(ry, np.dot(rx, ss)))
    return flat.reshape(coords.shape)


def _to_bone(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert joint coordinates to bone vectors (child - parent)."""
    bone = np.zeros_like(data)
    for child, parent in _BONE_PAIRS:
        bone[:, child - 1, :] = data[:, child - 1, :] - data[:, parent - 1, :]
    return bone


def _finalize(
    data: NDArray[np.float64], index_t: NDArray[np.float64], data_type: str, partition: bool
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Apply bone/motion, transpose to (C, T, V, M), reorder joints."""
    if "b" in data_type:
        data = _to_bone(data)
    if "m" in data_type:
        motion = np.zeros_like(data)
        motion[:-1] = data[1:] - data[:-1]
        data = motion

    data = np.transpose(data, (2, 0, 1))  # (C, T, V)
    c, t, v = data.shape
    data = data.reshape(c, t, v, 1)  # (C, T, V, M=1)
    if partition:
        data = data[:, :, _PARTITION_INDEX]
    return data.astype(np.float32), index_t.astype(np.float32)


def preprocess_eval(
    skeletons: NDArray[np.float64], data_type: str = "j", partition: bool = True
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Deterministic preprocessing for inference/evaluation.

    Args:
        skeletons: ``(T, 20, 3)`` raw sequence.
        data_type: ``"j"`` joint, ``"b"`` bone, ``"m"`` motion (letters combine).
        partition: reorder joints into body-part groups (required by the model).

    Returns:
        ``data`` ``(C, T=64, V=20, M=1)`` and ``index_t`` ``(T=64,)``, both float32.
    """
    value = np.asarray(skeletons, dtype=np.float64)
    value = value - value[0, CENTER_JOINT, :]
    value = minmax_normalize(value).reshape(-1, NUM_JOINTS, 3)

    length = value.shape[0]
    idx = np.linspace(0, length - 1, TIME_STEPS).astype(int)
    data = value[idx, :, :]
    index_t = 2 * idx.astype(np.float64) / length - 1
    return _finalize(data, index_t, data_type, partition)


def preprocess_train(
    skeletons: NDArray[np.float64],
    data_type: str = "j",
    partition: bool = True,
    drop_prob: float = 0.5,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Augmented preprocessing for training (random view, sampling, dropout).

    Uses the global ``random``/``numpy`` RNG state, matching the reference feeder.
    """
    value = np.asarray(skeletons, dtype=np.float64)
    agx = random.randint(-60, 60)
    agy = random.randint(-60, 60)
    s = random.uniform(0.5, 1.5)

    value = value - value[0, CENTER_JOINT, :]
    value = _rand_view_transform(value, agx, agy, s)
    value = minmax_normalize(value).reshape(-1, NUM_JOINTS, 3)

    length = value.shape[0]
    random_idx = random.sample(list(np.arange(length)) * 100, TIME_STEPS)
    random_idx.sort()
    data = value[random_idx, :, :].copy()
    index_t = 2 * np.array(random_idx).astype(np.float64) / length - 1

    # drop a whole axis
    if random.random() < drop_prob:
        data[:, :, random.randint(0, 2)] = 0.0
    # drop a random joint/time block
    if random.random() < drop_prob:
        t_dim, v_dim, _ = data.shape
        joints = sorted(random.sample(range(v_dim), random.randint(4, 12)))
        frames = sorted(random.sample(range(t_dim), random.randint(16, 32)))
        block = data[np.ix_(frames, joints)]
        data[np.ix_(frames, joints)] = np.zeros_like(block)

    return _finalize(data, index_t, data_type, partition)
