#!/usr/bin/env python3
"""Load best.pt, run on test set, produce confusion_matrices.pdf."""
import multiprocessing
multiprocessing.set_start_method("fork")

import sys
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent))
from efficientNet import (
    EfficientNetMultiTask, FlyCsvDataset, LabelEncoders, pad_to_square_white
)

# ── config ────────────────────────────────────────────────────────────────────
CKPT   = "checkpoints_temp_prediction/best.pt"
CSV    = "droso_classif.csv"
OUT    = "confusion_matrices.pdf"
IMG_SZ = 456
VAL_BS = 32
WORKERS = 4


# ── helpers ───────────────────────────────────────────────────────────────────

def conf_matrix(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def stats(cm):
    total = cm.sum()
    acc = cm.diagonal().sum() / max(total, 1)
    f1s = []
    rows = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        sup  = cm[i, :].sum()
        f1s.append(f1)
        rows.append((prec, rec, f1, sup))
    macro_f1 = float(np.mean(f1s))
    return acc, macro_f1, rows


def plot_cm_page(pdf, cm, labels, task_name, acc, macro_f1, per_class_rows):
    """One PDF page: confusion matrix + per-class table."""
    n = len(labels)
    fig = plt.figure(figsize=(max(8, n * 0.9 + 3), max(7, n * 0.8 + 3)))

    title = (f"{task_name}   |   Accuracy: {acc*100:.1f}%   |   "
             f"Macro F1: {macro_f1:.3f}   |   N={cm.sum()}")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.45,
                           top=0.92, bottom=0.06, left=0.12, right=0.95)

    # ── confusion matrix ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.maximum(row_sums, 1)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Recall (row-normalised)")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", labelpad=4)
    ax.set_ylabel("True", labelpad=4)

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val == 0:
                continue
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=max(6, 10 - n // 3), color=color)

    # ── per-class table ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    col_labels = ["Class", "Precision", "Recall", "F1", "Support"]
    table_data = [
        [lbl,
         f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}", str(int(sup))]
        for lbl, (prec, rec, f1, sup) in zip(labels, per_class_rows)
    ]
    tbl = ax2.table(cellText=table_data, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D9E2F3")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── load checkpoint ───────────────────────────────────────────────────────────

print(f"Loading {CKPT} …")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
enc_dict = ckpt["encoders"]
encoders = LabelEncoders(
    sex_to_id     = enc_dict["sex_to_id"],
    species_to_id = enc_dict["species_to_id"],
    temp_to_id    = enc_dict["temp_to_id"],
)
print(f"  epoch {ckpt['epoch']}  best_val_loss={ckpt['extra'].get('best_val_loss', '?'):.4f}")
print(f"  sex:     {encoders.sex_to_id}")
print(f"  species: {encoders.species_to_id}")
print(f"  temp:    {encoders.temp_to_id}")

# ── build transform + dataset ─────────────────────────────────────────────────

backbone = "efficientnet_b5"
import torchvision
weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
mean = weights.transforms().mean
std  = weights.transforms().std
tf = T.Compose([
    T.Lambda(pad_to_square_white),
    T.Resize((IMG_SZ, IMG_SZ)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

test_ds = FlyCsvDataset(CSV, split="test", transform=tf, encoders=encoders)
test_loader = DataLoader(test_ds, batch_size=VAL_BS, shuffle=False,
                         num_workers=WORKERS, pin_memory=True)
print(f"  test size: {len(test_ds)}")

# ── model ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetMultiTask(
    backbone=backbone,
    pretrained=False,
    num_sex=len(encoders.sex_to_id),
    num_species=len(encoders.species_to_id),
    num_temp=len(encoders.temp_to_id),
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── inference ─────────────────────────────────────────────────────────────────

id2sex  = {v: k for k, v in encoders.sex_to_id.items()}
id2sp   = {v: k for k, v in encoders.species_to_id.items()}
id2temp = {v: k for k, v in encoders.temp_to_id.items()}

sex_gt, sex_pr   = [], []
sp_gt,  sp_pr    = [], []
temp_gt, temp_pr = [], []

print("Running inference …")
with torch.no_grad():
    for imgs, sex, species, temp in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        out  = model(imgs)
        sex_gt   += [id2sex[i]  for i in sex.tolist()]
        sp_gt    += [id2sp[i]   for i in species.tolist()]
        temp_gt  += [id2temp[i] for i in temp.tolist()]
        sex_pr   += [id2sex[i]  for i in out.sex_logits.argmax(1).cpu().tolist()]
        sp_pr    += [id2sp[i]   for i in out.species_logits.argmax(1).cpu().tolist()]
        temp_pr  += [id2temp[i] for i in out.temp_logits.argmax(1).cpu().tolist()]

print(f"  done ({len(sex_gt)} samples)")

# ── PDF ───────────────────────────────────────────────────────────────────────

sex_labels  = sorted(encoders.sex_to_id.keys())
sp_labels   = sorted(encoders.species_to_id.keys())
temp_labels = sorted(encoders.temp_to_id.keys())

# combined label: "species_sex_temp"
combo_gt = [f"{sp}_{s}_{t}" for sp, s, t in zip(sp_gt, sex_gt, temp_gt)]
combo_pr = [f"{sp}_{s}_{t}" for sp, s, t in zip(sp_pr, sex_pr, temp_pr)]
combo_labels = sorted(set(combo_gt) | set(combo_pr))

tasks = [
    ("Temperature",        temp_gt, temp_pr, temp_labels),
    ("Species",            sp_gt,   sp_pr,   sp_labels),
    ("Sex",                sex_gt,  sex_pr,  sex_labels),
    ("Overall (all 3 correct)", combo_gt, combo_pr, combo_labels),
]

def plot_temp_acc_by_species_sex(pdf, sp_gt, sex_gt, temp_gt, temp_pr,
                                  sp_labels, sex_labels, temp_labels):
    """Facet by species, colour by sex: temp-prediction accuracy per (temp, sex) cell."""
    import pandas as pd

    df = pd.DataFrame({
        "species": sp_gt, "sex": sex_gt,
        "temp_gt": temp_gt, "temp_pr": temp_pr,
    })
    df["correct"] = df["temp_gt"] == df["temp_pr"]

    grp = (df.groupby(["species", "sex", "temp_gt"])
             .agg(acc=("correct", "mean"), n=("correct", "count"))
             .reset_index())

    overall_acc = df["correct"].mean()
    n_sp = len(sp_labels)
    sex_colors = {"F": "#E07B7B", "M": "#5B9BD5"}
    bar_w = 0.35
    x = np.arange(len(temp_labels))

    fig, axes = plt.subplots(1, n_sp, figsize=(3.2 * n_sp, 4.8), sharey=True)
    fig.suptitle(
        f"Temperature prediction accuracy by species & sex   |   "
        f"Overall temp acc: {overall_acc*100:.1f}%   |   N={len(df)}",
        fontsize=11, fontweight="bold", y=1.02,
    )

    for ax, sp in zip(axes, sp_labels):
        for j, sex in enumerate(sex_labels):
            sub = grp[(grp.species == sp) & (grp.sex == sex)]
            accs, ns = [], []
            for t in temp_labels:
                row = sub[sub.temp_gt == t]
                accs.append(float(row["acc"].iloc[0]) if len(row) else np.nan)
                ns.append(int(row["n"].iloc[0])  if len(row) else 0)

            offset = (j - (len(sex_labels) - 1) / 2) * bar_w
            bars = ax.bar(x + offset, accs, bar_w,
                          label=sex, color=sex_colors.get(sex, "grey"),
                          edgecolor="white", linewidth=0.5)

            for bar, acc_val, n_val in zip(bars, accs, ns):
                if np.isnan(acc_val):
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2,
                        acc_val + 0.012, f"{acc_val*100:.0f}%\n(n={n_val})",
                        ha="center", va="bottom", fontsize=7, color="black")

        ax.set_title(sp, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}°C" for t in temp_labels], fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_xlabel("Rearing temperature (GT)", fontsize=8)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Accuracy", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=sex_colors.get(s, "grey"))
               for s in sex_labels]
    fig.legend(handles, sex_labels, title="Sex", loc="lower right",
               bbox_to_anchor=(1.0, 0.0), fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


print(f"Writing {OUT} …")
with PdfPages(OUT) as pdf:
    for task_name, gt, pr, labels in tasks:
        cm = conf_matrix(gt, pr, labels)
        acc, macro_f1, per_class_rows = stats(cm)
        print(f"  {task_name}: acc={acc*100:.1f}%  macro_f1={macro_f1:.3f}")
        plot_cm_page(pdf, cm, labels, task_name, acc, macro_f1, per_class_rows)

    plot_temp_acc_by_species_sex(
        pdf, sp_gt, sex_gt, temp_gt, temp_pr,
        sp_labels, sex_labels, temp_labels,
    )
    print("  Temp-acc by species/sex: done")

    info = pdf.infodict()
    info["Title"]   = "Drosophila temp-prediction — test set confusion matrices"
    info["Author"]  = "eval_confusion.py"

print(f"Saved → {OUT}")
