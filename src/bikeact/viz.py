"""Minimal skeleton-sequence visualization: render an animated GIF."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

# 0-indexed bone connectivity for the 20-joint skeleton (child -> parent).
SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
    (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
]


def _render_frame(
    fig: Figure,
    coords: NDArray[np.float64],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
) -> NDArray[np.uint8]:
    """Draw one (20, 3) frame (x horizontal, y vertical) and return an RGB array."""
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    for a, b in SKELETON_EDGES:
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]], "-", color="#1f77b4", lw=2)
    ax.plot(coords[:, 0], coords[:, 1], "o", color="#d62728", ms=4)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def render_gif(
    skeletons: NDArray[np.float64],
    out_path: str | Path,
    title: str = "",
    fps: int = 10,
) -> Path:
    """Render a ``(T, 20, 3)`` skeleton sequence to an animated GIF.

    Coordinates are plotted in the x (lateral) / y (vertical) plane. Axis limits
    are fixed over the whole sequence so the motion is stable.
    """
    coords = np.asarray(skeletons, dtype=np.float64)
    # Center each frame on the pelvis so the body stays put and limb motion is
    # visible (world translation would otherwise shrink the skeleton to a dot).
    coords = coords - coords[:, 0:1, :]
    # Data y increases downward (camera convention); flip so the head is up.
    coords[:, :, 1] *= -1.0
    half = 1.15 * float(max(np.abs(coords[:, :, 0]).max(), np.abs(coords[:, :, 1]).max(), 1e-3))
    xlim = (-half, half)
    ylim = (-half, half)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4, 4), dpi=100)
    try:
        frames = [_render_frame(fig, coords[t], xlim, ylim, title) for t in range(coords.shape[0])]
    finally:
        plt.close(fig)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    return out
