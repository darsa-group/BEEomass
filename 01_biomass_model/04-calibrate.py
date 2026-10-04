"""
Post-hoc calibration of a Gaussian log-BF run.

1. Select a checkpoint by validation accuracy in % terms (mean |log ratio| on mass), rather
   than the validation NLL, which favours early, under-fitted epochs whose sigma is still
   honest. Runs that log `val_male` per epoch use its smoothed curve; older runs have every
   saved checkpoint scored on validation directly.
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

from models import build_efficientnet, load_weights_to_model, load_gaussian_model, call_gaussian_model

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
        # [log L, log projected area], used only by scale-input models
        self.scales = np.log(df[["ROI_SIZE_MM", "AREA_MM2"]].to_numpy(dtype=np.float32))
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB")), torch.from_numpy(self.scales[idx])


def log_mass_error(y, mu, target):
    """Per-image log(M_hat / M); for "logbf" it is 3x the log-BF error, as L is measured."""
    return (3.0 if target == "logbf" else 1.0) * (mu - y)


def select_checkpoint(run_dir: Path, window: int, target: str, val_df: pd.DataFrame, args):
    """Saved checkpoint with the lowest validation mean |log ratio| on mass.

    Returns (description, epoch, path). With a per-epoch `val_male` column, the nearest saved
    checkpoint to the argmin of its centred rolling median; otherwise every saved checkpoint
    is scored on the validation split.
    """
    saved = sorted(int(p.stem.replace("checkpoint_epoch", "")) for p in run_dir.glob("checkpoint_epoch*.pt"))
    if not saved:
        raise FileNotFoundError(f"No checkpoint_epoch*.pt in {run_dir}")
    res = pd.read_csv(run_dir / "epoch_results.csv")
    res = res[res["epoch"] != "test"].astype({"epoch": int})
    if "val_male" in res.columns:
        smooth = res["val_male"].rolling(window, center=True, min_periods=1).median()
        best = int(res["epoch"].iloc[int(np.argmin(smooth.to_numpy()))])
        nearest = min(saved, key=lambda e: abs(e - best))
        how = f"rolling median of val |log ratio| (window {window}), argmin epoch {best}"
        return how, nearest, run_dir / f"checkpoint_epoch{nearest}.pt"
    col = "BF_cbrMG_MM" if target == "logbf" else "DRYMASS_MG"
    y = np.log(val_df[col].to_numpy(dtype=float))
    scores = {}
    for e in saved:
        model = load_gaussian_model(run_dir / f"checkpoint_epoch{e}.pt", DEVICE, args.variant)
        mu = run_model(model, val_df, args.batch_size, args.num_workers)[:, 0]
        scores[e] = float(np.mean(np.abs(log_mass_error(y, mu, target))))
        print(f"  checkpoint epoch {e:4d}: val mean |log ratio| {scores[e]:.4f}", flush=True)
    best = min(scores, key=scores.get)
    return "val mean |log ratio| of each saved checkpoint", best, run_dir / f"checkpoint_epoch{best}.pt"


@torch.no_grad()
def run_model(model, df, batch_size, num_workers):
    loader = DataLoader(ImageDataset(df), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = []
    for images, scales in loader:
        out.append(call_gaussian_model(model, images.to(DEVICE), scales.to(DEVICE)).cpu().numpy())
    return np.concatenate(out, axis=0)


def accuracy_on_mass(log_err):
    """% accuracy of dry mass from per-image log(M_hat / M)."""
    ape = np.abs(np.expm1(log_err)) * 100
    return {"mdape": float(np.median(ape)), "male": float(np.mean(np.abs(log_err))),
            "within10": float(np.mean(ape <= 10)), "within25": float(np.mean(ape <= 25)),
            "within50": float(np.mean(ape <= 50))}


def gaussian_metrics(y, mu, sigma, target="logbf"):
    """y, mu, sigma on the log target. mdape (median % error), male (mean |log ratio|) and
    withinX (share within X%) are on dry mass for either target. r2 and mae are on the BF
    scale for "logbf" and on log mass for "logmass", as in 02-train.py, for continuity."""
    bf, bf_pred = (np.exp(y), np.exp(mu)) if target == "logbf" else (y, mu)
    z = (y - mu) / sigma
    m = {
        **accuracy_on_mass(log_mass_error(y, mu, target)),
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
    ap.add_argument("--target", choices=["logbf", "logmass"], default="logbf",
                    help="what the run predicts (the --target it was trained with)")
    ap.add_argument("--selection-window", type=int, default=11)
    ap.add_argument("--baseline", type=Path, default=None,
                    help="optional point-model checkpoint (1-output head) to score on the same test images")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["IS_VALID"] == True]
    parts = {s: df[df["SPLIT"] == s].reset_index(drop=True) for s in ("val", "test")}

    how, epoch, ckpt = select_checkpoint(args.run, args.selection_window, args.target, parts["val"], args)
    print(f"Selected checkpoint epoch {epoch} ({ckpt.name}) by {how}")

    model = load_gaussian_model(ckpt, DEVICE, args.variant)

    preds, report = {}, {}
    for split, d in parts.items():
        out = run_model(model, d, args.batch_size, args.num_workers)
        col = "BF_cbrMG_MM" if args.target == "logbf" else "DRYMASS_MG"
        preds[split] = (np.log(d[col].to_numpy(dtype=float)), out[:, 0],
                        np.exp(np.clip(out[:, 1], LOG_SIGMA_MIN, LOG_SIGMA_MAX)))

    y, mu, sigma = preds["val"]
    k = float(np.sqrt(np.mean(((y - mu) / sigma) ** 2)))
    k95 = float(np.quantile(np.abs((y - mu) / sigma), 0.95) / Z95)
    print(f"Calibration factor k = {k:.3f} (NLL-optimal on val); k for exact 95% val coverage = {k95:.3f}")

    for split in ("val", "test"):
        y, mu, sigma = preds[split]
        report[split] = {"raw": gaussian_metrics(y, mu, sigma, args.target),
                         "calibrated": gaussian_metrics(y, mu, k * sigma, args.target)}

    if args.baseline is not None and args.target == "logbf":
        base = build_efficientnet(variant=args.variant, pretrained=False, n_outputs=1)
        base = load_weights_to_model(base, args.baseline, DEVICE).eval()
        bf = parts["test"]["BF_cbrMG_MM"].to_numpy(dtype=float)
        bf_pred = run_model(base, parts["test"], args.batch_size, args.num_workers).reshape(-1)
        report["test"]["baseline"] = {
            **accuracy_on_mass(3.0 * (np.log(np.clip(bf_pred, 1e-6, None)) - np.log(bf))),
            "r2": float(1 - np.sum((bf - bf_pred) ** 2) / np.sum((bf - bf.mean()) ** 2)),
            "mae": float(np.mean(np.abs(bf - bf_pred))),
        }

    rows = []
    for split, r in report.items():
        for kind, m in r.items():
            rows.append({"split": split, "model": kind, **m})
    table = pd.DataFrame(rows).set_index(["split", "model"])
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 220, "display.max_columns", None):
        print(table)

    calib = {
        "target": args.target, "checkpoint": ckpt.name, "epoch": epoch, "selection": how,
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
        name = "LOGBF" if args.target == "logbf" else "LOGMASS"
        d[f"{name}_MU"], d[f"{name}_SIGMA_RAW"], d[f"{name}_SIGMA"] = mu, sigma, k * sigma
        out.append(d)
    pd.concat(out).to_csv(args.run / "val_test_predictions.csv", index=False)
    print(f"Wrote calibration.json, calibrated_model.pt and val_test_predictions.csv to {args.run}")


if __name__ == "__main__":
    main()
