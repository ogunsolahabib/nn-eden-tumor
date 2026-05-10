"""
Training loop with:
  - k-means RBF initialisation (centers frozen for warm-up epochs)
  - class-weighted BCE loss for the 1:4 imbalance
  - per-epoch train/val loss and macro-F1 recording
  - early stopping on val macro-F1 with best-weight restore
  - learning curve saved to aml-final/fig2_learning_curves.png
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from rbf_layer import init_centers_from_kmeans
from network import TumorClassifier


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, preds_all, labels_all = 0.0, [], []

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(yb)
            preds_all.append((logits.detach() > 0).cpu().numpy())
            labels_all.append(yb.cpu().numpy())

    preds  = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    avg_loss = total_loss / len(labels)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return avg_loss, f1


# ── main training function ────────────────────────────────────────────────────

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    *,
    n_centers:    int   = 20,
    warmup_epochs: int  = 20,   # centers frozen; rest of network adapts first
    max_epochs:   int   = 300,
    batch_size:   int   = 64,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    dropout:      float = 0.3,
    patience:     int   = 30,   # early-stopping patience (val macro-F1)
    seed:         int   = 42,
    output_dir:   str   = "aml-final",
) -> tuple[TumorClassifier, dict]:

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── model ────────────────────────────────────────────────────────────────
    model = TumorClassifier(n_centers=n_centers, dropout=dropout).to(device)

    # k-means init (unsupervised — only X, not y)
    print(f"Running k-means (k={n_centers}) on training features …")
    km = init_centers_from_kmeans(model.rbf, X_train, seed=seed)

    # ── loss with class balancing ─────────────────────────────────────────────
    n_neg = (y_train == 0).sum()   # C1, minority
    n_pos = (y_train == 1).sum()   # C2, majority
    # pos_weight < 1  →  down-weight C2 so each class contributes equally
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── data loaders ─────────────────────────────────────────────────────────
    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   batch_size, shuffle=False)

    # ── optimizer (centers frozen during warm-up) ─────────────────────────────
    model.freeze_centers()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=15, factor=0.5, min_lr=1e-5,
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1":   [], "val_f1":   [],
    }

    best_val_f1   = -np.inf
    best_weights  = None
    epochs_no_imp = 0

    print(f"Training for up to {max_epochs} epochs "
          f"(warm-up {warmup_epochs}, patience {patience}) …\n"
          f"{'Epoch':>6}  {'TrLoss':>8}  {'TrF1':>7}  "
          f"{'VaLoss':>8}  {'VaF1':>7}  {'Note'}")
    print("-" * 58)

    for epoch in range(1, max_epochs + 1):

        # Unfreeze centers after warm-up
        if epoch == warmup_epochs + 1:
            model.unfreeze_centers()
            for g in optimizer.param_groups:
                g["params"] = list(
                    filter(lambda p: p.requires_grad, model.parameters())
                )

        tr_loss, tr_f1 = _epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_f1 = _epoch(model, val_loader,   criterion, optimizer, device, train=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        scheduler.step(va_f1)

        note = ""
        if va_f1 > best_val_f1:
            best_val_f1  = va_f1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_imp = 0
            note = "✓ best"
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                note = "→ early stop"
                print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_f1:>7.4f}  "
                      f"{va_loss:>8.4f}  {va_f1:>7.4f}  {note}")
                break

        if epoch % 10 == 0 or note:
            print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_f1:>7.4f}  "
                  f"{va_loss:>8.4f}  {va_f1:>7.4f}  {note}")

    # restore best checkpoint
    model.load_state_dict(best_weights)
    print(f"\nBest val macro-F1: {best_val_f1:.4f}")

    # ── learning curves ───────────────────────────────────────────────────────
    _plot_learning_curves(history, warmup_epochs, output_dir)

    return model, history, km


def _plot_learning_curves(history: dict, warmup_epochs: int, output_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, tr_key, va_key, ylabel in zip(
        axes,
        ["train_loss", "train_f1"],
        ["val_loss",   "val_f1"],
        ["BCE Loss",   "Macro F1"],
    ):
        ax.plot(epochs, history[tr_key], label="Train", linewidth=1.8)
        ax.plot(epochs, history[va_key], label="Val",   linewidth=1.8, linestyle="--")
        ax.axvline(warmup_epochs, color="grey", linestyle=":", linewidth=1,
                   label=f"Centers unfrozen (ep {warmup_epochs})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    # Overfitting gap annotation on the loss panel
    tr_loss = np.array(history["train_loss"])
    va_loss = np.array(history["val_loss"])
    gap = va_loss - tr_loss
    axes[0].fill_between(epochs, tr_loss, va_loss,
                         where=(gap > 0), alpha=0.15, color="red",
                         label="Overfit gap")
    axes[0].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_learning_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Learning curves saved → {path}")
