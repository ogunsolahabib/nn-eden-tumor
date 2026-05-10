"""
Evaluation suite:
  1. Classification report  (precision / recall / F1 per class)
  2. ROC-AUC and PR-AUC    (imbalance-robust aggregate metrics)
  3. Confusion matrix
  4. Decision boundary plot (train + test overlay)
  5. Overfitting summary    (train vs test gap table + verdict)
  6. Complexity sweep        (train vs val F1 vs k-centers)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    f1_score,
    roc_curve,
    precision_recall_curve,
)

from network import TumorClassifier


# ── helpers ──────────────────────────────────────────────────────────────────

def _predict(model: TumorClassifier, X: np.ndarray, device) -> tuple:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device)).squeeze(1)
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return preds, probs


# ── public API ────────────────────────────────────────────────────────────────

def full_evaluation(
    model:      TumorClassifier,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_test:     np.ndarray,
    y_test:     np.ndarray,
    km=None,                   # fitted KMeans (for center overlay)
    output_dir: str = "aml-final",
):
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    tr_preds, tr_probs = _predict(model, X_train, device)
    te_preds, te_probs = _predict(model, X_test,  device)

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    _classification_report(te_preds, y_test)
    _roc_pr_auc(tr_probs, y_train, te_probs, y_test)
    _confusion_matrix(te_preds, y_test, output_dir)
    _decision_boundary(model, X_train, y_train, X_test, y_test, km, device, output_dir)
    _overfitting_summary(tr_preds, tr_probs, y_train, te_preds, te_probs, y_test)
    _roc_pr_curves(tr_probs, y_train, te_probs, y_test, output_dir)


def _classification_report(preds, labels):
    print("\n── Per-class metrics (test set) ──")
    print(classification_report(
        labels, preds,
        target_names=["C1 tumour rim (minority)", "C2 immune ring (majority)"],
        digits=4,
    ))


def _roc_pr_auc(tr_probs, y_train, te_probs, y_test):
    tr_roc = roc_auc_score(y_train, tr_probs)
    te_roc = roc_auc_score(y_test,  te_probs)
    tr_pr  = average_precision_score(y_train, tr_probs)
    te_pr  = average_precision_score(y_test,  te_probs)

    print("── AUC metrics ──")
    print(f"  {'Metric':<18} {'Train':>8}  {'Test':>8}  {'Gap':>8}")
    print(f"  {'ROC-AUC':<18} {tr_roc:>8.4f}  {te_roc:>8.4f}  {tr_roc-te_roc:>+8.4f}")
    print(f"  {'PR-AUC':<18} {tr_pr:>8.4f}  {te_pr:>8.4f}  {tr_pr-te_pr:>+8.4f}")


def _confusion_matrix(preds, labels, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay.from_predictions(
        labels, preds,
        display_labels=["C1 rim", "C2 immune"],
        ax=ax, colorbar=False,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved → {path}")


def _decision_boundary(model, X_train, y_train, X_test, y_test, km, device, output_dir):
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(grid, dtype=torch.float32).to(device)).squeeze(1)
    Z = torch.sigmoid(logits).cpu().numpy().reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, X_pts, y_pts, title in zip(
        axes,
        [X_train, X_test],
        [y_train, y_test],
        ["Train set", "Test set"],
    ):
        ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.6)
        ax.contour( xx, yy, Z, levels=[0.5], colors="k", linewidths=1.5)

        mask0 = y_pts == 0
        ax.scatter(X_pts[mask0, 0], X_pts[mask0, 1],
                   c="#c0392b", s=20, alpha=0.85,
                   edgecolors="white", linewidths=0.3, label="C1 (tumour rim)")
        ax.scatter(X_pts[~mask0, 0], X_pts[~mask0, 1],
                   c="#2980b9", s=8, alpha=0.45,
                   edgecolors="none", label="C2 (immune ring)")

        if km is not None:
            ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                       marker="x", s=60, c="white", linewidths=1.5,
                       label="k-means centers")

        ax.legend(fontsize=8, framealpha=0.7)
        ax.set_xlabel("x₁ (scaled)")
        ax.set_ylabel("x₂ (scaled)")

    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_decision_boundary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Decision boundary saved → {path}")


def _overfitting_summary(tr_preds, tr_probs, y_train, te_preds, te_probs, y_test):
    """
    Compares train vs test on three metrics and prints a plain-language verdict.
    A gap > 0.05 on any metric is flagged as a potential overfitting signal.
    """
    metrics = {}
    for name, tr_val, te_val in [
        ("Macro F1",
         f1_score(y_train, tr_preds, average="macro", zero_division=0),
         f1_score(y_test,  te_preds, average="macro", zero_division=0)),
        ("ROC-AUC",
         roc_auc_score(y_train, tr_probs),
         roc_auc_score(y_test,  te_probs)),
        ("PR-AUC",
         average_precision_score(y_train, tr_probs),
         average_precision_score(y_test,  te_probs)),
    ]:
        metrics[name] = (tr_val, te_val, tr_val - te_val)

    print("\n── Overfitting analysis ──")
    print(f"  {'Metric':<12} {'Train':>8}  {'Test':>8}  {'Gap':>8}  {'Flag'}")
    print(f"  {'-'*52}")
    any_flag = False
    for name, (tr, te, gap) in metrics.items():
        flag = "⚠ overfit?" if gap > 0.05 else "ok"
        if gap > 0.05:
            any_flag = True
        print(f"  {name:<12} {tr:>8.4f}  {te:>8.4f}  {gap:>+8.4f}  {flag}")

    print()
    if any_flag:
        print("  Verdict: train–test gap exceeds 0.05 on at least one metric.")
        print("  Consider: increasing dropout, reducing n_centers, or adding L2.")
    else:
        print("  Verdict: no significant overfitting detected (all gaps ≤ 0.05).")


def _roc_pr_curves(tr_probs, y_train, te_probs, y_test, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for probs, labels, tag, ls, lw in [
        (tr_probs, y_train, "Train", "-",  2.0),
        (te_probs, y_test,  "Test",  "--", 2.0),
    ]:
        fpr, tpr, _  = roc_curve(labels, probs)
        prec, rec, _ = precision_recall_curve(labels, probs)
        auc_roc = roc_auc_score(labels, probs)
        auc_pr  = average_precision_score(labels, probs)

        axes[0].plot(fpr, tpr, linestyle=ls, linewidth=lw,
                     label=f"{tag} (AUC = {auc_roc:.4f})")
        axes[1].plot(rec, prec, linestyle=ls, linewidth=lw,
                     label=f"{tag} (AP = {auc_pr:.4f})")

    axes[0].plot([0, 1], [0, 1], "k:", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    baseline_pr = (y_test == 1).mean()
    axes[1].axhline(baseline_pr, color="k", linestyle=":", linewidth=1,
                    label=f"Baseline (prevalence = {baseline_pr:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig6_roc_pr.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC / PR curves saved → {path}")


# ── complexity sweep (called separately from run.py) ─────────────────────────

def complexity_sweep(
    X_train, y_train, X_val, y_val,
    center_grid: list[int],
    train_fn,          # callable matching train.train signature
    output_dir: str = "aml-final",
):
    """
    Train with different numbers of RBF centers and plot train vs val F1.
    Reveals the bias-variance trade-off and confirms the chosen k is not
    over-parameterised.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_f1s, va_f1s = [], []

    for k in center_grid:
        print(f"\n── Sweep k={k} ──")
        model, history, _ = train_fn(
            X_train, y_train, X_val, y_val,
            n_centers=k, output_dir=output_dir,
        )
        preds_tr, _ = _predict(model, X_train, device)
        preds_va, _ = _predict(model, X_val,   device)
        tr_f1s.append(f1_score(y_train, preds_tr, average="macro", zero_division=0))
        va_f1s.append(f1_score(y_val,   preds_va, average="macro", zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(center_grid, tr_f1s, "o-", label="Train macro-F1")
    ax.plot(center_grid, va_f1s, "s--", label="Val macro-F1")
    ax.set_xlabel("Number of RBF centers (k)")
    ax.set_ylabel("Macro F1")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig5_complexity_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nComplexity sweep saved → {path}")

    return center_grid, tr_f1s, va_f1s
