"""Degrade held-out images at increasing strength and record how mu, sigma and the real error move.

Usage (from 01_biomass_model/, which holds models.py):
  python ../tools/degrade-study.py <val_test_predictions.csv> <checkpoint.pt> <sigma_scale> <out_dir> [logbf|logmass] [kinds]

Errors are reported on log BF for "logbf" (as before) and also on log mass for both targets,
so a BF model and a direct-mass model can be compared on the same footing.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path.cwd()))
from models import load_gaussian_model, call_gaussian_model

LOG_SIGMA_MIN, LOG_SIGMA_MAX, Z95 = -7.0, 2.0, 1.959964

# name -> list of levels; level 0 of each is the untouched image.
# Training augmentation already covered blur up to sigma 4 (kernel 3 or 9) and brightness +-20%.
DEGRADATIONS = {
    "blur":       [0, 1, 2, 4, 8],            # Gaussian blur radius, px (image is 224 px)
    "resolution": [1, 2, 4, 8, 16],           # downsample factor, then upsample back to 224 px
    "noise":      [0, 0.05, 0.1, 0.2, 0.4],   # additive Gaussian noise SD, on a 0-1 scale
    "jpeg":       [0, 30, 15, 8, 3],          # JPEG quality (0 = none)
    "darken":     [1, 0.7, 0.5, 0.3, 0.15],   # brightness factor
    "occlusion":  [0, 0.1, 0.2, 0.35, 0.5],   # grey square covering this share of the specimen's bounding box
}


def degrade(img: Image.Image, kind: str, level, seed: int) -> Image.Image:
    if kind == "blur" and level:
        return img.filter(ImageFilter.GaussianBlur(radius=level))
    if kind == "resolution" and level > 1:
        w, h = img.size
        small = img.resize((max(1, w // level), max(1, h // level)), Image.BILINEAR)
        return small.resize((w, h), Image.BICUBIC)
    if kind == "noise" and level:
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = a + np.random.default_rng(seed).normal(0, level, a.shape)
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    if kind == "jpeg" and level:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(level))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind == "darken" and level != 1:
        return ImageEnhance.Brightness(img).enhance(level)
    if kind == "occlusion" and level:
        a = np.asarray(img).copy()
        fg = (a.astype(int).sum(axis=2) < 3 * 240)          # non-white pixels = specimen
        ys, xs = np.nonzero(fg)
        if len(ys) == 0:
            return img
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        side = int(round(np.sqrt(level * (y1 - y0 + 1) * (x1 - x0 + 1))))
        rng = np.random.default_rng(seed)
        oy = rng.integers(y0, max(y0 + 1, y1 - side + 2))
        ox = rng.integers(x0, max(x0 + 1, x1 - side + 2))
        a[oy:oy + side, ox:ox + side] = 128
        return Image.fromarray(a)
    return img


class DegradedDataset(Dataset):
    def __init__(self, paths, scales, kind, level):
        self.paths, self.scales, self.kind, self.level = paths, scales, kind, level
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        # degradation leaves the physical size unchanged: same [log L, log area]
        return self.to_tensor(degrade(img, self.kind, self.level, seed=i)), torch.from_numpy(self.scales[i])


def main():
    pred_csv, ckpt, k, out = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
    target = sys.argv[5] if len(sys.argv) > 5 else "logbf"
    kinds = sys.argv[6].split(",") if len(sys.argv) > 6 else list(DEGRADATIONS)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = pd.read_csv(pred_csv)
    paths = d.IMAGE_FILENAME.tolist()
    area = pd.read_csv("metadata_enriched.csv").set_index("IMAGE_FILENAME").loc[d.IMAGE_FILENAME, "AREA_MM2"]
    scales = np.log(np.column_stack([d.ROI_SIZE_MM.to_numpy(float), area.to_numpy(float)])).astype(np.float32)
    y = np.log(d.BF_cbrMG_MM.to_numpy(float))
    log_m = np.log(d.DRYMASS_MG.to_numpy(float))
    log_l = np.log(d.ROI_SIZE_MM.to_numpy(float))

    model = load_gaussian_model(ckpt, device)

    rows, summary = [], []
    for kind in kinds:
        levels = DEGRADATIONS[kind]
        for li, level in enumerate(levels):
            loader = DataLoader(DegradedDataset(paths, scales, kind, level), batch_size=64, num_workers=8)
            outs = []
            with torch.no_grad():
                for x, sc in loader:
                    outs.append(call_gaussian_model(model, x.to(device), sc.to(device)).cpu().numpy())
            o = np.concatenate(outs)
            mu = o[:, 0]
            sigma = k * np.exp(np.clip(o[:, 1], LOG_SIGMA_MIN, LOG_SIGMA_MAX))
            # log-mass prediction and SD for either target
            lm_hat, lm_sd = (3 * (mu + log_l), 3 * sigma) if target == "logbf" else (mu, sigma)
            err_m = lm_hat - log_m
            err = (y - mu) if target == "logbf" else -err_m
            rows.append(pd.DataFrame({"kind": kind, "level_idx": li, "level": level, "INSECT_ID": d.INSECT_ID,
                                      "IMAGE_FILENAME": d.IMAGE_FILENAME, "y": y, "mu": mu, "sigma": sigma}))
            s = {"kind": kind, "level_idx": li, "level": level,
                 "mean_sigma": sigma.mean(), "rms_error": np.sqrt((err ** 2).mean()),
                 "bias": err.mean(), "cov95": np.mean(np.abs(err) <= Z95 * sigma),
                 "nll": np.mean(np.log(sigma) + 0.5 * (err / sigma) ** 2),
                 "mass_bias": err_m.mean(), "mass_rms": np.sqrt((err_m ** 2).mean()),
                 "mass_sd": lm_sd.mean(), "mass_cov95": np.mean(np.abs(err_m) <= Z95 * lm_sd)}
            summary.append(s)
            print(f"{target:7s} {kind:10s} {str(level):>5s}  log-mass bias {s['mass_bias']:+.3f}  rms {s['mass_rms']:.3f}  "
                  f"sd {s['mass_sd']:.3f}  cov95 {s['mass_cov95']:.3f}", flush=True)
    pd.concat(rows).to_csv(out / "degrade_per_image.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "degrade_summary.csv", index=False)


if __name__ == "__main__":
    main()
