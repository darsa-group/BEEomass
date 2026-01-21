"""
Inference script: load a .pt model (ResNet50 single-output), run inference over all images listed
in a metadata CSV, write predictions back into a new CSV column `BF_PRED` and save as `predictions.csv`.

Constants at top control paths. Uses a simple DataLoader for batched inference.

Assumptions:
- metadata CSV has a column `IMAGE_FILENAME` (relative to ROOT_IMG_DIR or absolute path).
- The saved weights file can be either a state_dict for the model or a full model (torch.save(model)).

Usage example:
> python predict_from_weights.py --csv .old-metadata-full.csv --root data --weights runs/regression_resnet50/best_model.pt --out runs/predictions

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
from models import build_resnet, load_weights_to_model
# -------------------- CONSTANTS --------------------
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def run_inference(csv_path: Path, weights_path: Path, out_dir: Path, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, root_img_dir: Path = "."):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded metadata CSV with {len(df)} rows")

    transform = get_inference_transform()
    ds = InferenceDataset(df=df, root_dir=root_img_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # build model and load weights
    # model = build_resnet18(pretrained=False)
    model = build_resnet(architecture=ARCH, pretrained=False)
    model = load_weights_to_model(model, weights_path, DEVICE)
    model.eval()
    preds = np.full((len(df),), np.nan, dtype=float)

    with torch.no_grad():
        for images, orig_indices in loader:
            images = images.to(DEVICE)
            outputs = model(images).detach().cpu().numpy().reshape(-1)
            for i, orig_idx in enumerate(orig_indices):
                preds[orig_idx] = float(outputs[i])

    # append BF_PRED column to dataframe and save
    df_out = df.copy()
    df_out["BF_PRED"] = preds

    out_csv = out_dir / "predictions.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv} (added column 'BF_PRED')")
    return df_out

# -------------------- CLI --------------------

if __name__ == "__main__":

    # ROOT_IMG_DIR = Path("data")  # <- change me
    METADATA_CSV = Path("metadata_enriched.csv")  # <- change me
    # WEIGHTS = Path("runs/regression_resnet101-bak/best_model.pt")
    # WEIGHTS = Path("01_runs/regression_resnet50/2025-12-03_18-09-27/best_model.pt")
    WEIGHTS = Path("../../01_biomass_model/01_runs/regression_resnet50/2025-12-04_08-23-47/best_model.pt")
    OUT_DIR = Path(".")
    NUM_WORKERS = 16
    BATCH_SIZE = 16
    ARCH="50"


    run_inference(csv_path=METADATA_CSV, weights_path=WEIGHTS,
                  out_dir=Path("./"), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
