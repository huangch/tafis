"""
tafis.train

Training utilities for the GPU-capable ANFISRegressor.

Responsibilities:
  - Split data into train/val (optionally test handled in CLI).
  - Create PyTorch DataLoaders.
  - Train ANFISRegressor on GPU (if available).
  - Track best validation loss and save checkpoint:
        - best.pt   (model state_dict + config)
        - last.pt   (last epoch)
  - Save preprocessing metadata (features_used.json) and helpful lists.

Design choices:
  - Pure gradient-based optimization (Adam). Works well on GPU.
  - MSE loss for regression.
  - Optional L2 weight decay for regularization.
  - Optional "firing sparsity" penalty: encourages the model to use fewer rules per sample,
    which can improve interpretability (rules become more distinct).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import FeatureMetadata
from .model import ANFISRegressor


@dataclass
class TrainConfig:
    """
    Training configuration for ANFISRegressor.
    """
    num_rules: int = 32
    batch_size: int = 4096
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    firing_sparsity_lambda: float = 0.0  # 0 disables this penalty
    val_size: float = 0.1
    seed: int = 1337
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 0  # set >0 if your environment supports it
    pin_memory: bool = True


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Create train/val DataLoaders.

    Returns:
      train_loader, val_loader, idx_train, idx_val
    """
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]

    n = X.shape[0]
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = int(round(n * cfg.val_size))
    n_val = max(1, min(n_val, n - 1))
    idx_val = indices[:n_val]
    idx_train = indices[n_val:]

    X_train = torch.from_numpy(X[idx_train]).float()
    y_train = torch.from_numpy(y[idx_train]).float()
    X_val = torch.from_numpy(X[idx_val]).float()
    y_val = torch.from_numpy(y[idx_val]).float()

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, idx_train, idx_val


def firing_sparsity_penalty(firing_norm: torch.Tensor) -> torch.Tensor:
    """
    Encourage sparse rule usage per sample.

    firing_norm: (N, R), each row sums to 1.

    A simple sparsity proxy:
      penalty = mean( 1 - max_i firing_i )

    If one rule dominates (max close to 1), penalty is small.
    If firing is spread out, penalty is larger.

    This is optional and controlled by cfg.firing_sparsity_lambda.
    """
    max_f = torch.max(firing_norm, dim=1).values
    return torch.mean(1.0 - max_f)


def save_checkpoint(
    out_dir: Path,
    name: str,
    model: ANFISRegressor,
    cfg: TrainConfig,
    meta: FeatureMetadata,
    extra: Optional[Dict] = None,
) -> Path:
    """
    Save a checkpoint with everything needed for inference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / name
    payload = {
        "model_state_dict": model.state_dict(),
        "num_features": model.num_features,
        "num_rules": model.num_rules,
        "train_config": cfg.__dict__,
        "feature_metadata": meta.__dict__,
        "extra": extra or {},
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def train_anfis_regressor(
    X: np.ndarray,
    y: np.ndarray,
    meta: FeatureMetadata,
    out_dir: str | Path,
    cfg: TrainConfig,
) -> Dict[str, float]:
    """
    Train ANFISRegressor and save artifacts.

    Artifacts written to out_dir:
      - features_used.json (preprocessing metadata)
      - feature_cols_used.txt, id_cols.txt, dropped_cols.json, rename_map.json
      - best.pt, last.pt
      - train_log.json

    Returns:
      summary dict containing best_val_loss, last_val_loss, etc.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _set_seed(cfg.seed)
    device = _to_device(cfg.device)

    # Save metadata for inference reproducibility
    meta_path = out / "features_used.json"
    meta.to_json(meta_path)

    # Build loaders
    train_loader, val_loader, idx_train, idx_val = make_dataloaders(X, y, cfg)

    # Model
    model = ANFISRegressor(num_features=X.shape[1], num_rules=cfg.num_rules).to(device)

    # Optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    last_val = float("inf")

    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            y_pred, firing = model(xb)
            loss = loss_fn(y_pred, yb)

            if cfg.firing_sparsity_lambda > 0.0:
                sp = firing_sparsity_penalty(firing)
                loss = loss + cfg.firing_sparsity_lambda * sp

            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                y_pred, firing = model(xb)
                loss = loss_fn(y_pred, yb)
                if cfg.firing_sparsity_lambda > 0.0:
                    loss = loss + cfg.firing_sparsity_lambda * firing_sparsity_penalty(firing)
                val_losses.append(loss.detach().item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        last_val = val_loss

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(out, "best.pt", model, cfg, meta, extra={"best_epoch": best_epoch})

        # Always save last
        if epoch == cfg.epochs:
            save_checkpoint(out, "last.pt", model, cfg, meta, extra={"last_epoch": epoch})

    # Write train log
    log = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "last_val_loss": last_val,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_rules": int(cfg.num_rules),
        "val_size": float(cfg.val_size),
        "history": history,
        "train_indices_path": "train_indices.npy",
        "val_indices_path": "val_indices.npy",
    }

    # Save indices to allow exact reproduction of splits
    np.save(out / "train_indices.npy", idx_train)
    np.save(out / "val_indices.npy", idx_val)

    (out / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    return {
        "best_val_loss": best_val,
        "best_epoch": float(best_epoch),
        "last_val_loss": last_val,
    }
