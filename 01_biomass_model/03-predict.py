"""
Inference script: load a .pt model (EfficientNet with a Gaussian head on log BF), run inference over
all images listed in a metadata CSV, and save the predictions as `predictions.csv`.

Columns added:
- `LOGBF_MU`, `LOGBF_SIGMA`: the predicted distribution, log BF ~ N(mu, sigma^2). sigma is
  multiplied by the calibration factor k from `calibration.json` next to the weights
  (written by 04-calibrate.py), unless --sigma-scale is given.
- `BF_PRED`: exp(mu), the predicted median BF
- if the CSV has `ROI_SIZE_MM` (L): log M = 3 (log BF + log L) ~ N(3 (mu + log L), (3 sigma)^2), so
  mass is log-normal. `MASS_PRED_MG` is its median, `MASS_MEAN_MG` its mean
  (median * exp(9 sigma^2 / 2)), and `MASS_LO95_MG` / `MASS_HI95_MG` the central 95% interval.

Constants at top control paths. Uses a simple DataLoader for batched inference.

Assumptions:
- metadata CSV has a column `IMAGE_FILENAME` (relative to ROOT_IMG_DIR or absolute path).
- The saved weights file can be either a state_dict for the model or a full model (torch.save(model)).

Usage example:
> python predict_from_weights.py --csv .old-metadata-full.csv --root data --weights runs/regression_resnet50/best_model.pt --out runs/predictions

"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from models import build_efficientnet, load_weights_to_model
# -------------------- CONSTANTS --------------------
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Must match the clamp applied in 02-train.py.
LOG_SIGMA_MIN = -7.0
LOG_SIGMA_MAX = 2.0
Z95 = 1.959964

# -------------------- DATASET --------------------
class InferenceDataset(Dataset):
    """Loads images from IMAGE_FILENAME column of a dataframe. Returns (image_tensor, index).
    The returned index is the original dataframe index to allow writing predictions back in place.
    """
    def __init__(self, df: pd.DataFrame, root_dir: Optional[Path] = None, transform=None):
        self.df = df.reset_index(drop=False)  # keep original index in column 'index'
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.transform = transform
        if "IMAGE_FILENAME" not in self.df.columns:
            raise ValueError("DataFrame must contain 'IMAGE_FILENAME' column")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["IMAGE_FILENAME"])
        if not img_path.is_absolute() and self.root_dir is not None:
            img_path = self.root_dir / img_path

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        orig_index = int(row["index"])  # original dataframe index
        return img, orig_index

# -------------------- TRANSFORMS --------------------

def get_inference_transform():
    # match training/test transforms used in your training script (ToTensor only in your current script)
    return T.Compose([
        # If you used resizing/center-crop during training, enable similar behavior here.
        T.ToTensor(),
        # Uncomment normalization if your model expects normalized inputs
        # T.Normalize(mean=MEAN, std=STD),
    ])

# -------------------- INFERENCE --------------------

def tta_8views(images: torch.Tensor) -> torch.Tensor:
    """
    images: (B, C, H, W)
    returns: (8*B, C, H, W) in order:
      rot0, rot90, rot180, rot270, and the same after horizontal flip
    """
    rots = [torch.rot90(images, k=k, dims=(2, 3)) for k in (0, 1, 2, 3)]
    flips = [torch.flip(r, dims=(3,)) for r in rots]  # horizontal mirror (flip W)
    return torch.cat(rots + flips, dim=0)

def median_even(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    True median for even counts: average of the two middle values.
    x: tensor, e.g. (B, 8)
    returns: (B,)
    """
    xs, _ = torch.sort(x, dim=dim)
    n = xs.size(dim)
    # For n=8 -> middle indices 3 and 4
    lo = xs.select(dim, n//2 - 1)
    hi = xs.select(dim, n//2)
    return 0.5 * (lo + hi)

def run_inference(csv_path: Path, weights_path: Path, out_dir: Path, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, root_img_dir: Path = ".", tta: bool = False, sigma_scale: float = 1.0):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded metadata CSV with {len(df)} rows")

    transform = get_inference_transform()
    ds = InferenceDataset(df=df, root_dir=root_img_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # build model and load weights
    # model = build_resnet18(pretrained=False)
    model = build_efficientnet(variant=EFFNET_VARIANT, pretrained=False, n_outputs=2)
    model = load_weights_to_model(model, weights_path, DEVICE)
    model.eval()
    preds = np.full((len(df), 2), np.nan, dtype=float)  # mu, log sigma


    # Optional test-time augmentation: median over 4 rotations x 2 flips, taken separately
    # for mu and log sigma.
    # Off by default - the manuscript states augmentation is applied during
    # training only, so the reported pipeline should not augment at inference.
    print(f"TTA: {'8-view median' if tta else 'disabled (single forward pass)'}")
    with torch.no_grad():
        for images, orig_indices in loader:
            images = images.to(DEVICE)  # (B,C,H,W)
            B = images.size(0)

            if tta:
                aug = tta_8views(images)                    # (8B,C,H,W)
                out = model(aug).detach()                   # (8B,2)
                out = out.view(8, B, 2).transpose(0, 1)     # (B,8,2)
                vals = median_even(out, dim=1)              # (B,2)
            else:
                vals = model(images).detach()               # (B,2)

            vals_np = vals.cpu().numpy()
            for i, orig_idx in enumerate(orig_indices):
                preds[int(orig_idx)] = vals_np[i]

    mu = preds[:, 0]
    sigma = sigma_scale * np.exp(np.clip(preds[:, 1], LOG_SIGMA_MIN, LOG_SIGMA_MAX))

    df_out = df.copy()
    df_out["LOGBF_MU"] = mu
    df_out["LOGBF_SIGMA"] = sigma
    df_out["BF_PRED"] = np.exp(mu)
    added = ["LOGBF_MU", "LOGBF_SIGMA", "BF_PRED"]

    if "ROI_SIZE_MM" in df_out.columns:
        log_m_mu = 3.0 * (mu + np.log(df_out["ROI_SIZE_MM"].to_numpy(dtype=float)))
        log_m_sigma = 3.0 * sigma
        df_out["MASS_PRED_MG"] = np.exp(log_m_mu)
        df_out["MASS_MEAN_MG"] = np.exp(log_m_mu + 0.5 * log_m_sigma ** 2)
        df_out["MASS_LO95_MG"] = np.exp(log_m_mu - Z95 * log_m_sigma)
        df_out["MASS_HI95_MG"] = np.exp(log_m_mu + Z95 * log_m_sigma)
        added += ["MASS_PRED_MG", "MASS_MEAN_MG", "MASS_LO95_MG", "MASS_HI95_MG"]
    else:
        print("No ROI_SIZE_MM column: mass columns not written")

    out_csv = out_dir / "predictions.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv} (added columns {', '.join(added)})")
    return df_out

# -------------------- CLI --------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("metadata_enriched.csv"),
                        help="metadata CSV to predict over")
    # No default: the point-regression checkpoints have a 1-output head and do not load here.
    parser.add_argument("--weights", type=Path, required=True,
                        help="checkpoint of a Gaussian log-BF model (2-output head)")
    parser.add_argument("--out", type=Path, default=Path("."), help="directory to write predictions.csv into")
    parser.add_argument("--variant", default="v2_s", help="EfficientNet variant the checkpoint was trained with")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--tta", action="store_true",
                        help="enable 8-view test-time augmentation (off by default)")
    parser.add_argument("--sigma-scale", type=float, default=None,
                        help="factor applied to sigma (default: k from calibration.json beside the weights, else 1)")
    args = parser.parse_args()

    # run_inference reads this at call time, so it must be set before the call.
    EFFNET_VARIANT = args.variant

    print(f"weights: {args.weights}")
    sigma_scale = args.sigma_scale
    calib_path = args.weights.with_name("calibration.json")
    if sigma_scale is None and calib_path.exists():
        sigma_scale = json.loads(calib_path.read_text())["sigma_scale"]
        print(f"sigma scale: {sigma_scale:.3f} (from {calib_path})")
    elif sigma_scale is None:
        sigma_scale = 1.0
        print("sigma scale: 1.0 (no calibration.json beside the weights: intervals are uncalibrated)")
    else:
        print(f"sigma scale: {sigma_scale:.3f} (from --sigma-scale)")

    run_inference(csv_path=args.csv, weights_path=args.weights,
                  out_dir=args.out, batch_size=args.batch_size, num_workers=args.num_workers,
                  tta=args.tta, sigma_scale=sigma_scale)
