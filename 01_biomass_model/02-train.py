# <keep your existing header/docstring>

import os
import random
import argparse
from pathlib import Path
from platform import architecture
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
from models import build_resnet
from datetime import datetime



# -------------------- CONSTANTS / HYPERPARAMS --------------------
# Paths
ROOT_IMG_DIR = Path("00_data/02_resized")            # <- change me
METADATA_CSV = Path("metadata_enriched.csv")      # <- change me
RESNET_ARCH = "50"

OUT_DIR = Path(f"01_runs/regression_resnet{RESNET_ARCH}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

# Training hyperparams
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
STEP_LR_STEP = 100
STEP_LR_GAMMA = 0.5

# Image size and normalization (ResNet default expected normalization)
IMG_SIZE = 224

# Label smoothing multiplicative range for training (apply per-sample)
LABEL_SMOOTH_MIN = 0.8
LABEL_SMOOTH_MAX = 1.2
#how much the images can be downscaled during augmentation. this is a special
# augmentation that also modifies the target
DOWNSCALING_MIN = 0.5
# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Misc
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# -------------------- UTILITIES --------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# helper metric functions (work on numpy arrays)
def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # R^2 = 1 - SS_res / SS_tot
    # handle degenerate case (all y_true equal) -> return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

# -------------------- DATASET --------------------


class RandomDownscale:
    """
    Downscale image by factor in [min_factor, max_factor] (≤ 1),
    paste on white background of original size,
    return both image and scale factor.
    """

    def __init__(self, min_scale=DOWNSCALING_MIN, max_scale=1.0):
        assert 0 < min_scale <= max_scale <= 1.0
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: Image.Image):
        W, H = img.size

        # sample scale ∈ [min_scale, max_scale]
        scale = random.uniform(self.min_scale, self.max_scale)

        new_w = int(W * scale)
        new_h = int(H * scale)

        # downscale
        img_small = img.resize((new_w, new_h), Image.BICUBIC)

        # create white canvas
        canvas = Image.new("RGB", (W, H), (255, 255, 255))

        # center the scaled image
        offset = ((W - new_w) // 2, (H - new_h) // 2)
        canvas.paste(img_small, offset)

        return canvas, scale

class ImageRegDataset(Dataset):
    """Dataset that accepts a pandas DataFrame with at least these columns:
    - `IMAGE_FILENAME`: image filename (can be relative)
    - `DATASET`: subdirectory name under `root_dir` where the image lives
    - `BF_cbrMG_MM`: regression target (float)
    - `SPLIT`: 'train' | 'val' | 'test'
    Images are resolved as: root_dir / DATASET / IMAGE_FILENAME (unless IMAGE_FILENAME is absolute).
    """

    def __init__(self, df: pd.DataFrame, split: str, root_dir: Optional[Path] = None, transform=None):
        assert split in ("train", "val", "test")
        self.split = split
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.transform = transform

        # Filter by split
        self.df = df[df["SPLIT"] == split][df["IS_VALID"] == True].reset_index(drop=True).copy()

        # Column checks
        required_cols = {"IMAGE_FILENAME", "DATASET", "BF_cbrMG_MM"}

        missing = required_cols - set(self.df.columns)

        if self.split == "train":
            self.rescale_aug = RandomDownscale()
        else:
            self.rescale_aug = None

        if missing:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(row["IMAGE_FILENAME"])
        # print(row.DATASET)
        # If path is not absolute, resolve as <root>/<DATASET>/<IMAGE_FILENAME>

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.rescale_aug:
            img, scale = self.rescale_aug(img)
        else:
            img, scale = img, 1.0

        if self.transform is not None:
            img = self.transform(img)

        target = float(row["BF_cbrMG_MM"]) * scale ** 3
        # print(float(row["BF_cbrMG_MM"]), scale, target)
        return img, torch.tensor(target, dtype=torch.float32)

# -------------------- TRANSFORMS --------------------

def random_blur_or_sharpness():
    # Randomly apply GaussianBlur or Sharpness adjustment
    aug = random.choice([
        T.GaussianBlur(kernel_size=random.choice([3, 9]), sigma=(0.1, 4.0)),
        T.RandomAdjustSharpness(sharpness_factor=random.uniform(0.5, 2.0), p=1.0)
    ])
    return aug

def random_quadrant_rotation(img):
    """Randomly rotate image by 0, 90, 180, or 270 degrees."""
    angle = random.choice([0, 90, 180, 270])
    return F.rotate(img, angle, fill=(255, 255, 255))  # white background

def get_transforms(split: str):
    if split == "train":
        train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda img: random_quadrant_rotation(img)),
            # Color jitter (brightness, contrast, saturation, hue)
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # Gaussian blur (kernel size chosen relative to image size)
            T.Lambda(lambda img: random_blur_or_sharpness()(img)),
            T.ToTensor(),
        ])
        return train_transforms
    else:
        # val/test: deterministic scaling / center crop
        return T.Compose([
            T.ToTensor(),
        ])



# -------------------- TRAIN / VAL LOOP --------------------

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, label_smooth_min, label_smooth_max):
    model.train()
    running_loss = 0.0
    n_samples = 0

    # accumulators for metrics
    preds_all = []
    targets_all = []

    for images, targets in dataloader:
        tile_images_cv2(images, cols=8, pad=2, to_bgr=True, resize_to=(128, 128), window_name="train batch")
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)  # shape (B,1)

        # Apply multiplicative label smoothing / jitter per-sample (only for loss)
        with torch.no_grad():
            multipliers = torch.empty((targets.size(0), 1), device=targets.device).uniform_(label_smooth_min, label_smooth_max)
            smooth_targets = targets * multipliers

        preds = model(images)  # shape (B,1)

        loss = criterion(preds, smooth_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # detach preds and targets for metrics (use true targets, not smoothed)
        preds_cpu = preds.detach().cpu().numpy().reshape(-1)
        targets_cpu = targets.detach().cpu().numpy().reshape(-1)
        preds_all.append(preds_cpu)
        targets_all.append(targets_cpu)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / max(1, n_samples)

    # compute R^2 and MAE over collected arrays
    preds_all = np.concatenate(preds_all, axis=0) if len(preds_all) > 0 else np.array([])
    targets_all = np.concatenate(targets_all, axis=0) if len(targets_all) > 0 else np.array([])

    if preds_all.size > 0:
        train_r2 = compute_r2(targets_all, preds_all)
        train_mae = compute_mae(targets_all, preds_all)
    else:
        train_r2 = 0.0
        train_mae = 0.0

    print(f"Epoch {epoch} train loss: {epoch_loss:.6f}  R2: {train_r2:.4f}  MAE: {train_mae:.6f}")
    return epoch_loss, train_r2, train_mae

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    preds_all = []
    targets_all = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        preds = model(images)
        loss = criterion(preds, targets)

        preds_cpu = preds.detach().cpu().numpy().reshape(-1)
        targets_cpu = targets.detach().cpu().numpy().reshape(-1)
        preds_all.append(preds_cpu)
        targets_all.append(targets_cpu)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / max(1, n_samples)

    preds_all = np.concatenate(preds_all, axis=0) if len(preds_all) > 0 else np.array([])
    targets_all = np.concatenate(targets_all, axis=0) if len(targets_all) > 0 else np.array([])

    if preds_all.size > 0:
        val_r2 = compute_r2(targets_all, preds_all)
        val_mae = compute_mae(targets_all, preds_all)
    else:
        val_r2 = 0.0
        val_mae = 0.0

    print(f"Epoch {epoch} val loss: {epoch_loss:.6f}  R2: {val_r2:.4f}  MAE: {val_mae:.6f}")
    return epoch_loss, val_r2, val_mae

# -------------------- MAIN TRAIN FUNCTION --------------------

import cv2
import numpy as np
import torch
from typing import Optional, Sequence, Union

def _to_numpy_uint8(img: Union[torch.Tensor, np.ndarray],
                    denorm_mean: Optional[Sequence[float]] = None,
                    denorm_std: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Convert one image to HxWxC uint8 in RGB.
    Accepts:
      - torch.Tensor [C,H,W] float in [0,1] (or any float needing clamp)
      - np.ndarray HxWxC (float or uint8)
    """
    if isinstance(img, torch.Tensor):
        if img.ndim == 3 and img.size(0) in (1, 3):  # [C,H,W] -> [H,W,C]
            img = img.detach().cpu().permute(1, 2, 0).numpy()
        else:
            raise ValueError(f"Unexpected tensor shape {img.shape}; expected [C,H,W].")
    elif isinstance(img, np.ndarray):
        if img.ndim != 3:
            raise ValueError(f"Unexpected ndarray shape {img.shape}; expected [H,W,C].")
        img = img.copy()
    else:
        # Try PIL
        try:
            from PIL import Image
            if isinstance(img, Image.Image):
                img = np.array(img)  # PIL gives RGB uint8
            else:
                raise
        except Exception:
            raise ValueError("Unsupported image type; provide torch.Tensor, numpy.ndarray, or PIL.Image.")

    # Ensure 3 channels
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # If float, optionally de-normalize then clamp to [0,1]
    if img.dtype != np.uint8:
        if denorm_mean is not None and denorm_std is not None:
            mean = np.array(denorm_mean).reshape(1, 1, -1)
            std  = np.array(denorm_std).reshape(1, 1, -1)
            img = img * std + mean
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).round().astype(np.uint8)

    return img  # RGB uint8


def tile_images_cv2(
    batch: Union[torch.Tensor, Sequence[Union[torch.Tensor, np.ndarray]]],
    cols: int = 8,
    pad: int = 2,
    to_bgr: bool = True,
    resize_to: Optional[tuple[int, int]] = None,
    denorm_mean: Optional[Sequence[float]] = None,
    denorm_std: Optional[Sequence[float]] = None,
    window_name: str = "batch",
    show: bool = True,
) -> np.ndarray:
    """
    Make a tiled canvas from a batch of images and (optionally) show it with OpenCV.

    Args
    ----
    batch: Tensor [B,3,H,W] or list/tuple of images (Tensor [3,H,W] / ndarray HxWxC / PIL).
    cols:  tiles per row.
    pad:   pixel padding between tiles and around the border.
    to_bgr: convert RGB->BGR for OpenCV display.
    resize_to: (h, w) to resize each tile; None keeps original size.
    denorm_mean/std: pass if your tensors are normalized; values in [0,1], e.g. mean=[0.485,0.456,0.406].
    window_name: OpenCV window name.
    show: whether to call cv2.imshow + waitKey(1).

    Returns
    -------
    canvas: numpy uint8 image (BGR if to_bgr=True else RGB).
    """
    # Normalize input to a list of images
    imgs = []
    if isinstance(batch, torch.Tensor):
        assert batch.ndim == 4 and batch.size(1) in (1, 3), f"Expected [B,C,H,W], got {batch.shape}"
        for i in range(batch.size(0)):
            imgs.append(batch[i])
    else:
        imgs = list(batch)

    # Convert and (optionally) resize each to RGB uint8
    tiles = []
    for im in imgs:
        rgb = _to_numpy_uint8(im, denorm_mean, denorm_std)  # RGB
        if resize_to is not None:
            rh, rw = resize_to
            rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
        tiles.append(rgb)

    if len(tiles) == 0:
        raise ValueError("Empty batch.")

    H, W = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    canvas_h = rows * H + (rows + 1) * pad
    canvas_w = cols * W + (cols + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 0, dtype=np.uint8)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(tiles):
                break
            y0 = r * H + (r + 1) * pad
            x0 = c * W + (c + 1) * pad
            canvas[y0:y0+H, x0:x0+W] = tiles[idx]
            idx += 1

    # Convert to BGR for OpenCV display if requested
    if to_bgr:
        canvas_disp = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    else:
        canvas_disp = canvas

    if show:
        cv2.imshow(window_name, canvas_disp)
        cv2.waitKey(1)

    return canvas_disp


def run_training(
    csv_path: Path,
    root_img_dir: Path,
    out_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    num_workers: int = NUM_WORKERS,
    device: torch.device = DEVICE,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    df = pd.read_csv(csv_path)
    print(f"Total samples in csv: {len(df)}")

    # Datasets
    train_ds = ImageRegDataset(df=df, split="train", root_dir=root_img_dir, transform=get_transforms("train"))
    val_ds = ImageRegDataset(df=df, split="val", root_dir=root_img_dir, transform=get_transforms("val"))
    test_ds = ImageRegDataset(df=df, split="test", root_dir=root_img_dir, transform=get_transforms("test"))

    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)

    # Model, optimizer, loss
    model = build_resnet(architecture = RESNET_ARCH, pretrained=True).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP, gamma=STEP_LR_GAMMA)

    best_val_loss = float('inf')

    # prepare epoch results CSV
    results_csv_path = out_dir / "epoch_results.csv"
    # write header
    results_df = pd.DataFrame(columns=["epoch",
                                       "train_loss", "train_r2", "train_mae",
                                       "val_loss", "val_r2", "val_mae"])
    results_df.to_csv(results_csv_path, index=False)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):

        # images: [B,3,H,W] float


        train_loss, train_r2, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, LABEL_SMOOTH_MIN, LABEL_SMOOTH_MAX)
        val_loss, val_r2, val_mae = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # checkpoint
        ckpt_path = out_dir / f"checkpoint_epoch{epoch}.pt"
        if epoch % 50 ==0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, ckpt_path)

        # keep best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"Saved new best model at epoch {epoch} with val_loss={val_loss:.6f}")

        # append epoch results to CSV
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_r2": train_r2,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_r2": val_r2,
            "val_mae": val_mae,
        }
        row_df = pd.DataFrame([row])
        row_df.to_csv(results_csv_path, mode="a", header=False, index=False)

    # final test evaluation
    test_loss, test_r2, test_mae = validate(model, test_loader, criterion, device, epoch="test")
    print(f"Final test loss: {test_loss:.6f}  R2: {test_r2:.4f}  MAE: {test_mae:.6f}")

    # save history (optional)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False)

    # optionally append final test row into epoch_results.csv as epoch='test'
    final_row = {
        "epoch": "test",
        "train_loss": np.nan,
        "train_r2": np.nan,
        "train_mae": np.nan,
        "val_loss": test_loss,
        "val_r2": test_r2,
        "val_mae": test_mae,
    }
    pd.DataFrame([final_row]).to_csv(results_csv_path, mode="a", header=False, index=False)

    return history

# -------------------- SIMPLE CLI --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a regression ResNet50 from CSV metadata")
    parser.add_argument("--csv", type=str, default=str(METADATA_CSV), help="path to metadata csv")
    parser.add_argument("--root", type=str, default=str(ROOT_IMG_DIR), help="root dir to resolve image paths")
    parser.add_argument("--out", type=str, default=str(OUT_DIR), help="output directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    run_training(csv_path=Path(args.csv), root_img_dir=Path(args.root), out_dir=Path(args.out), batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr)
