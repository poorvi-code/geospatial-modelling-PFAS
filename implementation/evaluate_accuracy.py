#!/usr/bin/env python3
"""
implementation/evaluate_accuracy.py
=====================================
Accuracy, Calibration & PR Curve — uses the production trained model directly.
Fast: one GroupShuffleSplit hold-out, no re-training.

Outputs → outputs/figures/eval_accuracy_bars.png
          outputs/figures/eval_calibration_curve.png
          outputs/figures/eval_precision_recall.png
"""

import json
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, matthews_corrcoef,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
FIG_DIR    = ROOT / "outputs" / "figures"
GOLDEN     = ROOT / "dataset" / "pfas_golden.parquet"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = json.loads((MODELS_DIR / "feature_schema.json").read_text())
TARGET    = "above_100_ng_l"
GROUP_COL = "spatial_block_id"

PALETTE = {
    "bg": "#0f172a", "panel": "#1e293b", "border": "#334155",
    "text": "#e2e8f0", "muted": "#94a3b8",
    "green": "#10b981", "red": "#ef4444", "yellow": "#f59e0b",
    "blue": "#6366f1", "cyan": "#06b6d4", "purple": "#a855f7",
}
METRIC_COLORS = [
    PALETTE["green"], PALETTE["blue"], PALETTE["yellow"],
    PALETTE["red"], PALETTE["cyan"], PALETTE["purple"], PALETTE["muted"],
]


def _style(fig):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in fig.axes:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"])
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE["border"])


def load_eval_data():
    print("Loading dataset …")
    df = pd.read_parquet(GOLDEN).dropna(subset=[TARGET])
    X      = df[FEATURE_COLS].fillna(-1)
    y      = df[TARGET].values.astype(int)
    groups = df[GROUP_COL].values if GROUP_COL in df.columns else np.arange(len(df))

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, test_idx = next(splitter.split(X, y, groups))

    with open(MODELS_DIR / "lgbm_calibrated.pkl", "rb") as f:
        model = pickle.load(f)

    X_test = X.iloc[test_idx]
    y_test = y[test_idx]
    probs  = model.predict_proba(X_test)[:, 1]
    preds  = (probs >= 0.5).astype(int)

    metrics = {
        "ROC-AUC":   roc_auc_score(y_test, probs),
        "PR-AUC":    average_precision_score(y_test, probs),
        "F1":        f1_score(y_test, preds, zero_division=0),
        "Accuracy":  accuracy_score(y_test, preds),
        "Bal. Acc.": balanced_accuracy_score(y_test, preds),
        "Brier ↓":   brier_score_loss(y_test, probs),
        "MCC":       matthews_corrcoef(y_test, preds),
    }
    print("  Metrics computed.")
    for k, v in metrics.items():
        print(f"    {k:12s} = {v:.4f}")

    return y_test, probs, preds, metrics


def plot_accuracy_bars(metrics):
    names  = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(13, 5))
    _style(fig)

    x      = np.arange(len(names))
    width  = 0.55
    bars   = ax.bar(x, values, width, color=METRIC_COLORS[:len(names)],
                    edgecolor=PALETTE["border"], linewidth=0.8)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{v:.4f}", ha="center", fontsize=10,
                color=PALETTE["text"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim([0, max(values) * 1.18])
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Metrics — Spatial Hold-Out Evaluation (20% test)",
                 fontsize=13, color=PALETTE["text"], pad=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.15, color=PALETTE["border"])
    ax.axhline(0.5, color=PALETTE["muted"], lw=0.8, linestyle=":",
               label="0.5 (random baseline for AUC)")
    ax.legend(fontsize=8, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text"])

    plt.tight_layout()
    out = FIG_DIR / "eval_accuracy_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"✓ Saved → {out}")


def plot_calibration(y_true, y_prob):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=12, strategy="quantile")
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 7))
    _style(fig)

    ax.plot([0, 1], [0, 1], ":", color=PALETTE["muted"], lw=1.5, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=PALETTE["green"], lw=2.2,
            markersize=8, markerfacecolor=PALETTE["bg"], markeredgewidth=2.2,
            label="PFAS classifier (sigmoid-calibrated)")
    for mp, fp in zip(mean_pred, frac_pos):
        ax.plot([mp, mp], [mp, fp], color=PALETTE["yellow"], lw=1, alpha=0.5, linestyle="--")

    ax.text(0.03, 0.95,
            f"Brier score = {brier:.4f}\n(0 = perfect, 0.25 = worst)",
            transform=ax.transAxes, fontsize=9, va="top", color=PALETTE["muted"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["panel"],
                      edgecolor=PALETTE["border"], alpha=0.85))

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives (Actual)", fontsize=11)
    ax.set_title("Reliability (Calibration) Diagram", fontsize=13,
                 color=PALETTE["text"], pad=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left",
              facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"])
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.08])
    ax.grid(True, alpha=0.15, color=PALETTE["border"])

    plt.tight_layout()
    out = FIG_DIR / "eval_calibration_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"✓ Saved → {out}")


def plot_precision_recall(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    f1_vals = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
        0.0,
    )
    best_idx    = np.argmax(f1_vals)
    best_thresh = thresholds[best_idx]
    best_f1     = f1_vals[best_idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    _style(fig)

    pos_rate = y_true.mean()
    ax.axhline(pos_rate, color=PALETTE["muted"], linestyle=":", lw=1.2,
               label=f"Random baseline (precision = {pos_rate:.3f})")
    ax.plot(recall, precision, color=PALETTE["blue"], lw=2.2,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.10, color=PALETTE["blue"])
    ax.scatter(recall[best_idx], precision[best_idx], s=120,
               color=PALETTE["yellow"], zorder=10, edgecolors=PALETTE["bg"],
               label=f"Best F1 = {best_f1:.4f}  @  threshold = {best_thresh:.2f}")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (PPV)", fontsize=11)
    ax.set_title("Precision-Recall Curve — PFAS Exceedance Classifier",
                 fontsize=13, color=PALETTE["text"], pad=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right",
              facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"])
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.06])
    ax.grid(True, alpha=0.15, color=PALETTE["border"])

    plt.tight_layout()
    out = FIG_DIR / "eval_precision_recall.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"✓ Saved → {out}")


if __name__ == "__main__":
    y_true, y_prob, preds, metrics = load_eval_data()
    plot_accuracy_bars(metrics)
    plot_calibration(y_true, y_prob)
    plot_precision_recall(y_true, y_prob)
    print("\nDone — all accuracy figures saved to outputs/figures/")
