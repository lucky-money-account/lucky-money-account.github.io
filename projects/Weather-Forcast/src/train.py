"""
train.py — Training entry point
=================================
Reads config.yaml, loads data, trains model with early stopping,
saves best checkpoint + TensorBoard logs.

Usage:
    python train.py                    (use config.yaml model.name)
    python train.py --model lstm2      (override model name)
    python train.py --epochs 30        (override epochs)
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import WeatherDataset
from models import build_model, build_loss


def load_config():
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, loss_fn, optimizer, device, clip_norm):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total += loss_fn(model(X), y).item() * X.size(0)
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.lr:
        cfg["train"]["learning_rate"] = args.lr

    proj_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(proj_root, "data")
    log_dir = os.path.join(proj_root, "runs")
    ckpt_dir = os.path.join(proj_root, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device} (GPU training is {'' if device.type == 'cuda' else 'NOT '}available)")

    seed = cfg["train"]["seed"]
    torch.manual_seed(seed)

    train_ds = WeatherDataset(os.path.join(data_dir, "train.npz"))
    val_ds = WeatherDataset(os.path.join(data_dir, "val.npz"))
    batch_size = cfg["train"]["batch_size"]
    nw = cfg["train"].get("num_workers", 0)
    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_ldr = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg["train"]["loss"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler_patience = cfg["train"]["patience"]
    early_stop = cfg["train"].get("early_stop_patience", scheduler_patience * 2)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["train"]["factor"], patience=scheduler_patience,
    )
    clip_norm = cfg["train"]["grad_clip_norm"]
    epochs = cfg["train"]["epochs"]
    model_name = cfg["model"]["name"]

    run_tag = f"{model_name}_{cfg['train']['loss']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_tag))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  Model: {model_name} ({param_count:,} params)")
    print(f"  Loss: {cfg['train']['loss']} | Device: {device}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {cfg['train']['learning_rate']}")
    print(f"  Patience: {scheduler_patience} (scheduler) / {early_stop} (early stop)")
    print(f"{'='*60}")

    best_val, best_ep = float("inf"), 0
    stale = 0

    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_ldr, loss_fn, optimizer, device, clip_norm)
        val_loss = eval_epoch(model, val_ldr, loss_fn, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        writer.add_scalar("Loss/train", tr_loss, ep)
        writer.add_scalar("Loss/val", val_loss, ep)
        writer.add_scalar("LR", lr, ep)

        improved = val_loss < best_val
        if improved:
            best_val, best_ep = val_loss, ep
            stale = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{model_name}_best.pt"))
        else:
            stale += 1

        flag = " *" if improved else ""
        if ep == 1 or ep % 3 == 0 or ep == epochs or improved:
            print(f"  Epoch {ep:3d}/{epochs} | "
                  f"Train: {tr_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}{flag}")

        if stale >= early_stop:
            print(f"\n  Early stopping: no improvement for {stale} epochs.")
            break

    writer.close()
    print(f"\n  Best val loss: {best_val:.6f} @ epoch {best_ep}")
    print(f"  Checkpoint:     {ckpt_dir}/{model_name}_best.pt")
    print(f"  TensorBoard:    tensorboard --logdir {log_dir}/{run_tag}")


if __name__ == "__main__":
    main()
