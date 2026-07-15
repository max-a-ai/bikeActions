"""Demo: classify a single skeleton sample with a trained checkpoint.

Picks a random sample from a split (or a specific file), runs inference, prints
the predicted class with confidence, and renders an animated skeleton GIF.

Examples:
    uv run python demo.py                                  # random test sample, bone model
    uv run python demo.py --sample data/bikeScenes_test/a01_ID0_0.json
    uv run python demo.py --checkpoint checkpoints/skateformer_joint.pt --modality j
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from bikeact.dataset import load_sample
from bikeact.infer import load_model, predict
from bikeact.labels import CLASS_NAMES
from bikeact.viz import render_gif


def _default_modality(checkpoint: str) -> str:
    """Infer joint/bone modality from the checkpoint filename."""
    name = Path(checkpoint).stem.lower()
    if "bone" in name:
        return "b"
    if "joint" in name:
        return "j"
    return "b"


def _pick_sample(data_dir: Path, split_file: Path, rng: random.Random) -> Path:
    """Choose a random usable sample path from a split."""
    names = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
    rng.shuffle(names)
    for name in names:
        path = data_dir / name
        if path.exists() and load_sample(path) is not None:
            return path
    raise SystemExit(f"No usable samples found in {split_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/skateformer_bone.pt", help="model checkpoint (.pt)")
    p.add_argument("--modality", choices=["j", "b"], default=None, help="joint/bone; inferred from name if unset")
    p.add_argument("--sample", default=None, help="specific sample JSON (overrides --split)")
    p.add_argument("--data-dir", default="data/bikeScenes_test", help="folder of sample JSON files")
    p.add_argument("--split", default="test", help="split to sample from (train|validation|test)")
    p.add_argument("--split-dir", default="data/splits", help="folder of split .txt files")
    p.add_argument("--out", default="demo_output.gif", help="output GIF path")
    p.add_argument("--seed", type=int, default=0, help="random seed for sample choice")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-gif", action="store_true", help="skip GIF rendering")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    modality = args.modality or _default_modality(args.checkpoint)

    if args.sample is not None:
        sample_path = Path(args.sample)
    else:
        split_path = Path(args.split_dir) / f"{args.split}.txt"
        sample_path = _pick_sample(Path(args.data_dir), split_path, random.Random(args.seed))

    loaded = load_sample(sample_path)
    if loaded is None:
        raise SystemExit(f"Sample is not a usable 5-class sample: {sample_path}")
    skeletons, true_label = loaded

    model = load_model(args.checkpoint, device=args.device)
    pred, probs = predict(model, skeletons, data_type=modality, device=args.device)

    print(f"sample     : {sample_path.name}")
    print(f"checkpoint : {args.checkpoint}  (modality={modality}, device={args.device})")
    print(f"prediction : {CLASS_NAMES[pred]}  (confidence {probs[pred]:.3f})")
    print(f"true label : {CLASS_NAMES[true_label]}  [{'CORRECT' if pred == true_label else 'WRONG'}]")
    print("probabilities:")
    order = sorted(range(len(CLASS_NAMES)), key=lambda i: -probs[i])
    for i in order:
        print(f"  {CLASS_NAMES[i]:>10}: {probs[i]:.3f}")

    if not args.no_gif:
        title = f"pred: {CLASS_NAMES[pred]} ({probs[pred]:.2f}) | true: {CLASS_NAMES[true_label]}"
        out = render_gif(skeletons, args.out, title=title)
        print(f"saved GIF  : {out}")


if __name__ == "__main__":
    main()
