"""
End-to-end entry point.

Usage:
    python run.py [--sweep]

    --sweep   also run the complexity sweep over different k values
              (adds ~5 minutes of training time)
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# resolve sibling imports when run directly
sys.path.insert(0, os.path.dirname(__file__))

from train import train
from evaluate import full_evaluation, complexity_sweep, architecture_sweep

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "aml-final")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data-generation", "tumor_dataset.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[["x1", "x2"]].values
    y = df["label"].values
    return X, y


def main(run_sweep: bool = False, run_arch_sweep: bool = False):
    print("=" * 60)
    print("Tumor Microenvironment — RBF Network Classifier")
    print("=" * 60)

    # ── data ─────────────────────────────────────────────────────────────────
    X, y = load_data()
    print(f"\nLoaded dataset: {len(X)} samples, "
          f"C1={( y==0).sum()}, C2={(y==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=42,
    )
    # splits: 70 / 10 / 20

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"Split — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ── train ─────────────────────────────────────────────────────────────────
    model, history, km = train(
        X_train, y_train, X_val, y_val,
        n_centers=20,
        warmup_epochs=20,
        max_epochs=300,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        patience=30,
        seed=42,
        output_dir=OUTPUT_DIR,
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    full_evaluation(
        model,
        X_train, y_train,
        X_test,  y_test,
        km=km,
        output_dir=OUTPUT_DIR,
    )

    # ── optional complexity sweep ─────────────────────────────────────────────
    if run_sweep:
        print("\n" + "=" * 60)
        print("Complexity sweep — varying number of RBF centers")
        print("=" * 60)

        def train_for_sweep(Xtr, ytr, Xv, yv, n_centers, output_dir):
            return train(
                Xtr, ytr, Xv, yv,
                n_centers=n_centers,
                warmup_epochs=20,
                max_epochs=200,
                batch_size=64,
                lr=1e-3,
                weight_decay=1e-4,
                dropout=0.3,
                patience=20,
                seed=42,
                output_dir=output_dir,
            )

        complexity_sweep(
            X_train, y_train, X_val, y_val,
            center_grid=[5, 10, 15, 20, 30, 40],
            train_fn=train_for_sweep,
            output_dir=OUTPUT_DIR,
        )

    # ── optional architecture ablation ───────────────────────────────────────
    if run_arch_sweep:
        print("\n" + "=" * 60)
        print("Architecture ablation — varying hidden-layer depth / width")
        print("=" * 60)

        def train_for_arch(Xtr, ytr, Xv, yv, hidden_dims, output_dir):
            return train(
                Xtr, ytr, Xv, yv,
                n_centers=20,
                hidden_dims=hidden_dims,
                warmup_epochs=20,
                max_epochs=200,
                batch_size=64,
                lr=1e-3,
                weight_decay=1e-4,
                dropout=0.3,
                patience=20,
                seed=42,
                output_dir=output_dir,
            )

        architecture_sweep(
            X_train, y_train, X_val, y_val,
            train_fn=train_for_arch,
            output_dir=OUTPUT_DIR,
        )

    print("\nDone. Outputs written to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                        help="Run complexity sweep over RBF center counts")
    parser.add_argument("--arch-sweep", action="store_true",
                        help="Run architecture ablation over hidden-layer configs")
    args = parser.parse_args()
    main(run_sweep=args.sweep, run_arch_sweep=args.arch_sweep)
