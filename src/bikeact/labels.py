"""Class scheme for the bikeActions 5-class skeleton action task.

The bundled checkpoints were trained on a 5-class scheme. Each source JSON stores
an integer ``label`` in a 9-way source taxonomy; only five of those map to a
training class, the rest are excluded from the dataset.

    source  1 walk   -> 0 Walking
    source  2 stand  -> 1 Standing
    source  4 bike   -> 2 Riding
    source  5 left   -> 3 Turning-L
    source  6 right  -> 4 Turning-R
    source  3 run, 7 wave, 8 stop, 9 wait -> excluded

This mapping is asserted against the bundled checkpoints in ``tests`` / the
``--check`` path of the demo (predictions on labelled data must agree with it).
"""

from __future__ import annotations

CLASS_NAMES: list[str] = ["Walking", "Standing", "Riding", "Turning-L", "Turning-R"]
NUM_CLASSES: int = len(CLASS_NAMES)

# Source-taxonomy integer label -> contiguous 0-indexed training class.
SOURCE_TO_TRAIN_CLASS: dict[int, int] = {
    1: 0,  # walk  -> Walking
    2: 1,  # stand -> Standing
    4: 2,  # bike  -> Riding
    5: 3,  # left  -> Turning-L
    6: 4,  # right -> Turning-R
}


def source_label_is_used(source_label: int) -> bool:
    """True if a source-taxonomy label maps to one of the five training classes."""
    return source_label in SOURCE_TO_TRAIN_CLASS


def class_name(train_class: int) -> str:
    """Human-readable name for a 0-indexed training class."""
    return CLASS_NAMES[train_class]
