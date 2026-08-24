"""
FIXME: this should be a link to same file in droso experiment
"""

import argparse
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


import torch

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

def run_inference(csv_path: Path, weights_path: Path, out_dir: Path, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, root_img_dir: Path = ".", tta: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded metadata CSV with {len(df)} rows")

    transform = get_inference_transform()
    ds = InferenceDataset(df=df, root_dir=root_img_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # build model and load weights
    # model = build_resnet18(pretrained=False)
    model = build_efficientnet(variant=EFFNET_VARIANT, pretrained=False)
    model = load_weights_to_model(model, weights_path, DEVICE)
    model.eval()
    preds = np.full((len(df),), np.nan, dtype=float)

    with torch.no_grad():
        for images, orig_indices in loader:
            images = images.to(DEVICE)  # (B,C,H,W)

            B = images.size(0)

            if tta:
                aug = tta_8views(images)                    # (8B,C,H,W)
                out = model(aug).view(-1).detach()          # (8B,)
                out = out.view(8, B).transpose(0, 1)        # (B,8)
                vals = median_even(out, dim=1)              # (B,)
            else:
                vals = model(images).view(-1).detach()      # (B,)

            vals_np = vals.cpu().numpy()
            for i, orig_idx in enumerate(orig_indices):
                preds[int(orig_idx)] = float(vals_np[i])
    # append BF_PRED column to dataframe and save
    df_out = df.copy()
    df_out["BF_PRED"] = preds

    out_csv = out_dir / "predictions.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv} (added column 'BF_PRED')")
    return df_out

# -------------------- CLI --------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("metadata_enriched.csv"))
    parser.add_argument("--weights", type=Path,
                        default=Path("../../01_biomass_model/01_runs/regression_effnetv2_s/"
                                     "MSaligned_2026-08-24_16-54-27/best_model.pt"))
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--variant", default="v2_s")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--tta", action="store_true",
                        help="enable 8-view test-time augmentation (off by default)")
    args = parser.parse_args()

    # run_inference reads this global at call time, so set it before calling.
    EFFNET_VARIANT = args.variant

    print(f"weights: {args.weights}")
    run_inference(csv_path=args.csv, weights_path=args.weights,
                  out_dir=args.out, batch_size=args.batch_size, num_workers=args.num_workers,
                  tta=args.tta)
