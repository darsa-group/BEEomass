#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def confusion_counts(gt: pd.Series, pred: pd.Series, labels: list[str]) -> np.ndarray:
    """Return confusion matrix with rows=GT, cols=Pred."""
    gt = pd.Categorical(gt, categories=labels, ordered=True)
    pred = pd.Categorical(pred, categories=labels, ordered=True)
    tab = pd.crosstab(gt, pred, dropna=False)  # includes all categories
    return tab.to_numpy()


def row_normalize(cm: np.ndarray) -> np.ndarray:
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return cm / rs



import matplotlib.colors as mcolors

def plot_cm(ax, cm: np.ndarray, labels: list[str], title: str, mode: str = "count"):
    """
    mode:
      - "count": show counts
      - "rowprop": show row-normalized proportions
    """
    if mode == "count":
        data = cm.astype(float)
        text = cm.astype(int).astype(str)
        vmin, vmax = 0.0, float(cm.max()) if cm.size else 1.0
    elif mode == "rowprop":
        data = row_normalize(cm)
        text = np.vectorize(lambda x: f"{x*100:.0f}%")(data)
        vmin, vmax = 0.0, 1.0
    else:
        raise ValueError("mode must be 'count' or 'rowprop'")

    cmap = plt.get_cmap("YlGnBu")  # very readable, colorblind-safe
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(data, cmap=cmap, norm=norm)

    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("GT")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # annotate with adaptive text color
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            # luminance-based text color
            color = "white" if norm(val) > 0.55 else "black"
            ax.text(j, i, text[i, j], ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect("equal")



def main():
    in_csv = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path("checkpoints_droso_effnet/test_predictions_best.csv")
    out_pdf = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("confusion_matrices.pdf")

    df = pd.read_csv(in_csv)
    required = {"filename", "sex_GT", "species_GT", "sex_Pred", "species_Pred"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    # define label order from GT, and ensure preds can include unseen labels too
    sex_labels = sorted(set(df["sex_GT"].astype(str)) | set(df["sex_Pred"].astype(str)))
    species_labels = sorted(set(df["species_GT"].astype(str)) | set(df["species_Pred"].astype(str)))

    # combined 10 classes: all species x all sex (fixed order)
    combo_labels = [f"{sp}_{sx}" for sp in species_labels for sx in sex_labels]

    df["combo_GT"] = df["species_GT"].astype(str) + "_" + df["sex_GT"].astype(str)
    df["combo_Pred"] = df["species_Pred"].astype(str) + "_" + df["sex_Pred"].astype(str)

    sex_cm = confusion_counts(df["sex_GT"].astype(str), df["sex_Pred"].astype(str), sex_labels)
    sp_cm = confusion_counts(df["species_GT"].astype(str), df["species_Pred"].astype(str), species_labels)
    combo_cm = confusion_counts(df["combo_GT"].astype(str), df["combo_Pred"].astype(str), combo_labels)

    with PdfPages(out_pdf) as pdf:
        # Sex: counts + row prop
        fig, ax = plt.subplots(figsize=(8.5, 7))
        plot_cm(ax, sex_cm, sex_labels, "Sex confusion matrix (counts)", mode="count")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 7))
        plot_cm(ax, sex_cm, sex_labels, "Sex confusion matrix (row proportions)", mode="rowprop")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Species: counts + row prop
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_cm(ax, sp_cm, species_labels, "Species confusion matrix (counts)", mode="count")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_cm(ax, sp_cm, species_labels, "Species confusion matrix (row proportions)", mode="rowprop")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Combined (10): counts + row prop
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_cm(ax, combo_cm, combo_labels, "Species × Sex confusion matrix (counts)", mode="count")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 10))
        plot_cm(ax, combo_cm, combo_labels, "Species × Sex confusion matrix (row proportions)", mode="rowprop")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()

