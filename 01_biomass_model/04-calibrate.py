"""
Post-hoc calibration of a Gaussian log-BF run.

1. Select a checkpoint by the smoothed validation MAE (accuracy), rather than the
   validation NLL, which favours early, under-fitted epochs whose sigma is still honest.
2. Predict the val and test splits with it.
3. Fit one factor k on validation so that sigma' = k * sigma. The NLL-optimal value has a
   closed form: d/dk mean(log(k sigma) + z^2 / (2 k^2)) = 0  =>  k = sqrt(mean(z^2)),
   with z = (y - mu) / sigma. mu is unchanged, so accuracy is unchanged.
4. Report test accuracy and calibration before and after, and optionally a point-model
   checkpoint on the same test images.

Writes `calibration.json` (read by 03-predict.py) and `calibrated_model.pt` into the run
directory, plus `val_test_predictions.csv`.

Usage:
> python 04-calibrate.py --run 01_runs/gaussian_logbf_effnetv2_s/<run> \
      --baseline 01_runs/regression_effnetv2_s/<point run>/selected_model.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models import build_efficientnet, load_weights_to_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Must match 02-train.py.
LOG_SIGMA_MIN = -7.0
LOG_SIGMA_MAX = 2.0
Z95 = 1.959964
# Two-sided z for the central intervals reported.
INTERVALS = {0.50: 0.674490, 0.80: 1.281552, 0.95: Z95}



class ImageDataset(Dataset):
    """Images from IMAGE_FILENAME, with the same ToTensor-only transform as val/test in 02-train.py."""
    def __init__(self, df: pd.DataFrame):
        self.paths = df["IMAGE_FILENAME"].tolist()
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def select_by_val_mae(run_dir: Path, window: int):
    """Nearest saved checkpoint to the argmin of the centred rolling median of val MAE."""
    res = pd.read_csv(run_dir / "epoch_results.csv")
    res = res[res["epoch"] != "test"].astype({"epoch": int})
    smooth = res["val_mae"].rolling(window, center=True, min_periods=1).median()
    target = int(res["epoch"].iloc[int(np.argmin(smooth.to_numpy()))])
    saved = sorted(int(p.stem.replace("checkpoint_epoch", "")) for p in run_dir.glob("checkpoint_epoch*.pt"))
    if not saved:
        raise FileNotFoundError(f"No checkpoint_epoch*.pt in {run_dir}")
    nearest = min(saved, key=lambda e: abs(e - target))
    return target, nearest, run_dir / f"checkpoint_epoch{nearest}.pt"


@torch.no_grad()
def run_model(model, df, batch_size, num_workers):
    loader = DataLoader(ImageDataset(df), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = []
    for images in loader:
        out.append(model(images.to(DEVICE)).cpu().numpy())
    return np.concatenate(out, axis=0)


def gaussian_metrics(y, mu, sigma):
    """y, mu, sigma on log BF. Accuracy on the BF scale, as in 02-train.py."""
    bf, bf_pred = np.exp(y), np.exp(mu)
    z = (y - mu) / sigma
    m = {
        "r2": float(1 - np.sum((bf - bf_pred) ** 2) / np.sum((bf - bf.mean()) ** 2)),
        "mae": float(np.mean(np.abs(bf - bf_pred))),
        "nll": float(np.mean(np.log(sigma) + 0.5 * z ** 2)),  # without 0.5*log(2*pi), as in training
        "mean_sigma": float(np.mean(sigma)),
    }
    for p, zc in INTERVALS.items():
        m[f"cov{int(p * 100)}"] = float(np.mean(np.abs(z) <= zc))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="Gaussian run directory")
    ap.add_argument("--csv", type=Path, default=Path("metadata_enriched.csv"))
    ap.add_argument("--variant", default="v2_s")
    ap.add_argument("--selection-window", type=int, default=11)
    ap.add_argument("--baseline", type=Path, default=None,
                    help="optional point-model checkpoint (1-output head) to score on the same test images")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    target, epoch, ckpt = select_by_val_mae(args.run, args.selection_window)
    print(f"Smoothed val-MAE minimum at epoch {target}; using saved checkpoint epoch {epoch} ({ckpt.name})")

    df = pd.read_csv(args.csv)
    df = df[df["IS_VALID"] == True]
    parts = {s: df[df["SPLIT"] == s].reset_index(drop=True) for s in ("val", "test")}

    model = build_efficientnet(variant=args.variant, pretrained=False, n_outputs=2)
    model = load_weights_to_model(model, ckpt, DEVICE).eval()

    preds, report = {}, {}
    for split, d in parts.items():
        out = run_model(model, d, args.batch_size, args.num_workers)
        preds[split] = (np.log(d["BF_cbrMG_MM"].to_numpy(dtype=float)), out[:, 0],
                        np.exp(np.clip(out[:, 1], LOG_SIGMA_MIN, LOG_SIGMA_MAX)))

    y, mu, sigma = preds["val"]
    k = float(np.sqrt(np.mean(((y - mu) / sigma) ** 2)))
    k95 = float(np.quantile(np.abs((y - mu) / sigma), 0.95) / Z95)
    print(f"Calibration factor k = {k:.3f} (NLL-optimal on val); k for exact 95% val coverage = {k95:.3f}")

    for split in ("val", "test"):
        y, mu, sigma = preds[split]
        report[split] = {"raw": gaussian_metrics(y, mu, sigma), "calibrated": gaussian_metrics(y, mu, k * sigma)}

    if args.baseline is not None:
        base = build_efficientnet(variant=args.variant, pretrained=False, n_outputs=1)
        base = load_weights_to_model(base, args.baseline, DEVICE).eval()
        bf = parts["test"]["BF_cbrMG_MM"].to_numpy(dtype=float)
        bf_pred = run_model(base, parts["test"], args.batch_size, args.num_workers).reshape(-1)
        report["test"]["baseline"] = {
            "r2": float(1 - np.sum((bf - bf_pred) ** 2) / np.sum((bf - bf.mean()) ** 2)),
            "mae": float(np.mean(np.abs(bf - bf_pred))),
        }

    rows = []
    for split, r in report.items():
        for kind, m in r.items():
            rows.append({"split": split, "model": kind, **m})
    table = pd.DataFrame(rows).set_index(["split", "model"])
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        print(table)

    calib = {
        "checkpoint": ckpt.name, "epoch": epoch, "smoothed_val_mae_argmin": target,
        "selection": f"rolling median of val_mae, window {args.selection_window}",
        "sigma_scale": k, "sigma_scale_cov95": k95,
        "baseline": str(args.baseline) if args.baseline else None,
        "metrics": report,
    }
    (args.run / "calibration.json").write_text(json.dumps(calib, indent=2))
    data = torch.load(ckpt, map_location="cpu")
    state = data["model_state_dict"] if isinstance(data, dict) and "model_state_dict" in data else data
    torch.save(state, args.run / "calibrated_model.pt")

    out = []
    for split, (y, mu, sigma) in preds.items():
        d = parts[split][["IMAGE_FILENAME", "INSECT_ID", "DATASET", "SPLIT", "BF_cbrMG_MM", "ROI_SIZE_MM", "DRYMASS_MG"]].copy()
        d["LOGBF_MU"], d["LOGBF_SIGMA_RAW"], d["LOGBF_SIGMA"] = mu, sigma, k * sigma
        out.append(d)
    pd.concat(out).to_csv(args.run / "val_test_predictions.csv", index=False)
    print(f"Wrote calibration.json, calibrated_model.pt and val_test_predictions.csv to {args.run}")


if __name__ == "__main__":
    main()
