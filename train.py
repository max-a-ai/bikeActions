"""Train a SkateFormer skeleton-action classifier on the bundled data.

Uses the train/validation/test splits under ``data/splits``. Logs to Weights &
Biases when ``--wandb-project`` is given, otherwise prints to the console and
writes ``metrics.json`` in the output directory.

Examples:
    uv run python train.py --modality b --epochs 100
    uv run python train.py --modality b --epochs 100 \
        --wandb-project bikeactions --wandb-name bone-run1
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from bikeact.dataset import SkeletonDataset
from bikeact.evaluate import evaluate
from bikeact.infer import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/bikeScenes_test")
    p.add_argument("--split-dir", default="data/splits")
    p.add_argument("--modality", choices=["j", "b"], default="b", help="joint or bone input")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--repeat", type=int, default=10, help="train-set augmentation multiplier")
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-lr", type=float, default=1e-7)
    p.add_argument("--warmup-epochs", type=int, default=25)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-max", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="runs/exp")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-name", default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at_epoch(epoch: int, args: argparse.Namespace) -> float:
    """Linear warmup then cosine decay, computed per epoch."""
    if epoch < args.warmup_epochs:
        frac = (epoch + 1) / max(1, args.warmup_epochs)
        return float(args.warmup_lr + (args.base_lr - args.warmup_lr) * frac)
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return float(args.min_lr + (args.base_lr - args.min_lr) * cosine)


def build_optimizer(model: torch.nn.Module, weight_decay: float, lr: float) -> torch.optim.Optimizer:
    """AdamW with weight decay disabled for the model's no-decay parameters."""
    no_decay: set[str] = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay_params: list[torch.nn.Parameter] = []
    nodecay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (nodecay_params if name in no_decay else decay_params).append(param)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
    grad_max: float,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for data, index_t, labels in loader:
        data = data.to(device)
        index_t = index_t.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(data, index_t)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max)
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        n += int(labels.size(0))
    return total_loss / max(1, n)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_dir = Path(args.split_dir)
    train_ds = SkeletonDataset(args.data_dir, split_dir / "train.txt", args.modality, train=True, repeat=args.repeat)
    val_ds = SkeletonDataset(args.data_dir, split_dir / "validation.txt", args.modality)
    test_ds = SkeletonDataset(args.data_dir, split_dir / "test.txt", args.modality)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=args.num_workers)
    print(f"train={len(train_ds)} (x{args.repeat})  val={len(val_ds)}  test={len(test_ds)}")

    model = build_model().to(args.device)
    optimizer = build_optimizer(model, args.weight_decay, args.base_lr)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    use_wandb = args.wandb_project is not None
    wandb_run: Any = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    history: list[dict[str, float]] = []
    best_val, best_epoch = 0.0, -1
    best_path = out_dir / "best.pt"
    for epoch in range(args.epochs):
        lr = lr_at_epoch(epoch, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device, args.grad_max)
        val = evaluate(model, val_loader, args.device)
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_acc": val.accuracy})
        print(f"epoch {epoch:3d}  lr={lr:.2e}  train_loss={train_loss:.4f}  val_acc={val.accuracy:.4f}")
        if use_wandb:
            wandb_run.log({"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_acc": val.accuracy})
        if val.accuracy >= best_val:
            best_val, best_epoch = val.accuracy, epoch
            torch.save(model.state_dict(), best_path)

    # Final test with the best checkpoint.
    model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True))
    test = evaluate(model, test_loader, args.device)
    print(f"\nbest val_acc={best_val:.4f} @ epoch {best_epoch}")
    print("TEST:\n" + test.format_report())

    metrics = {
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "test_acc": test.accuracy,
        "test_per_class": test.per_class_acc,
        "modality": args.modality,
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir / 'metrics.json'}  and  {best_path}")
    if use_wandb:
        wandb_run.summary["test_acc"] = test.accuracy
        wandb_run.finish()


if __name__ == "__main__":
    main()
