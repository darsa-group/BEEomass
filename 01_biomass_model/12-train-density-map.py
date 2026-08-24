"""
Dense-map regression training script with OpenCV debug windows.

- Model outputs per-pixel continuous map (B,1,H,W)
- Scalar prediction = mean(map)
- Easy map retrieval (return_map=True or get_last_map())
- Debug: show tiled input images + tiled predicted maps + optional overlay

Author: <keep your existing header/docstring>
"""

import random
import argparse
from pathlib import Path
from typing import Optional, Sequence, Union, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large


# -------------------- CONSTANTS / HYPERPARAMS --------------------
ROOT_IMG_DIR = Path("00_data/02_resized")          # <- change me
METADATA_CSV = Path("metadata_enriched.csv")       # <- change me

BACKBONE = "resnet50"  # "resnet50" | "mobilenetv3"
OUT_DIR = Path(f"01_runs/dense_regression_deeplab_{BACKBONE}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-5
STEP_LR_STEP = 100
STEP_LR_GAMMA = 0.5

LABEL_SMOOTH_MIN = 0.8
LABEL_SMOOTH_MAX = 1.2

DOWNSCALING_MIN = 0.5

# Explainable-map regularization (smoothness)
USE_TV_REG = True
TV_LAMBDA = 1e-4  # try 1e-5 .. 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# -------------------- DEBUG WINDOWS --------------------
SHOW_DEBUG_WINDOWS = True
SHOW_EVERY_N_BATCHES = 1
DEBUG_TILE_COLS = 8
DEBUG_TILE_PAD = 2
DEBUG_TILE_RESIZE_TO = (128, 128)  # (h, w)

SHOW_OVERLAY_WINDOW = True
OVERLAY_ALPHA = 0.45  # heatmap strength


# -------------------- UTILITIES --------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def tv_loss(m: torch.Tensor) -> torch.Tensor:
    dx = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().mean()
    dy = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().mean()
    return dx + dy


# -------------------- DATASET --------------------

class RandomDownscale:
    def __init__(self, min_scale=DOWNSCALING_MIN, max_scale=1.0):
        assert 0 < min_scale <= max_scale <= 1.0
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, float]:
        W, H = img.size
        scale = random.uniform(self.min_scale, self.max_scale)

        new_w = int(W * scale)
        new_h = int(H * scale)

        img_small = img.resize((new_w, new_h), Image.BICUBIC)

        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        offset = ((W - new_w) // 2, (H - new_h) // 2)
        canvas.paste(img_small, offset)

        return canvas, scale

class ImageRegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, root_dir: Optional[Path] = None, transform=None):
        assert split in ("train", "val", "test")
        self.split = split
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.transform = transform

        self.df = df[df["SPLIT"] == split][df["IS_VALID"] == True].reset_index(drop=True).copy()

        required_cols = {"IMAGE_FILENAME", "DATASET", "BF_cbrMG_MM"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

        self.rescale_aug = RandomDownscale() if self.split == "train" else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["IMAGE_FILENAME"])

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.rescale_aug:
            img, scale = self.rescale_aug(img)
        else:
            scale = 1.0

        if self.transform is not None:
            img = self.transform(img)

        # BF = M^(1/3)/L, so a simulated shrink by `scale` gives (scale^3 M)^(1/3)/L = scale * BF.
        # NOT scale**3: BF is already a cube root, so cubing applies the exponent twice.
        # This deviates from Eq (6) of the manuscript, which is incorrect.
        target = float(row["BF_cbrMG_MM"]) * scale
        return img, torch.tensor(target, dtype=torch.float32)


# -------------------- TRANSFORMS --------------------

def random_blur_or_sharpness():
    aug = random.choice([
        T.GaussianBlur(kernel_size=random.choice([3, 9]), sigma=(0.1, 4.0)),
        T.RandomAdjustSharpness(sharpness_factor=random.uniform(0.5, 2.0), p=1.0)
    ])
    return aug

def random_quadrant_rotation(img):
    angle = random.choice([0, 90, 180, 270])
    return TF.rotate(img, angle, fill=(255, 255, 255))

def get_transforms(split: str):
    if split == "train":
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(random_quadrant_rotation),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.Lambda(lambda img: random_blur_or_sharpness()(img)),
            T.ToTensor(),
        ])
    else:
        return T.Compose([T.ToTensor()])


# -------------------- DEBUG VIS (TILING + HEATMAPS) --------------------

def _tensor_to_rgb_uint8(img_chw: torch.Tensor) -> np.ndarray:
    """
    img_chw: [C,H,W] float ~[0,1]
    returns: RGB uint8 [H,W,3]
    """
    if img_chw.ndim != 3 or img_chw.size(0) not in (1, 3):
        raise ValueError(f"Expected [C,H,W] with C=1 or 3, got {tuple(img_chw.shape)}")

    x = img_chw.detach().cpu()
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    x = x.permute(1, 2, 0).numpy()  # HWC
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)
    return x

def _normalize_map_to_uint8(m_hw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize map per-image to [0,255] for visualization.
    """
    mn = float(m_hw.min())
    mx = float(m_hw.max())
    if mx - mn < eps:
        return np.zeros_like(m_hw, dtype=np.uint8)
    x = (m_hw - mn) / (mx - mn + eps)
    return (x * 255.0).round().astype(np.uint8)

def _apply_colormap(gray_u8: np.ndarray) -> np.ndarray:
    """
    gray_u8: [H,W] uint8
    returns: BGR uint8 (OpenCV colormap output)
    """
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)

def tile_rgb_images_cv2(
    batch_bchw: torch.Tensor,
    cols: int,
    pad: int,
    resize_to: Optional[Tuple[int, int]],
) -> np.ndarray:
    """
    batch_bchw: [B,3,H,W] float
    returns: BGR canvas uint8
    """
    tiles = []
    for i in range(batch_bchw.size(0)):
        rgb = _tensor_to_rgb_uint8(batch_bchw[i])  # RGB
        if resize_to is not None:
            rh, rw = resize_to
            rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tiles.append(bgr)

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
    return canvas

def tile_heatmaps_cv2(
    map_b1hw: torch.Tensor,
    cols: int,
    pad: int,
    resize_to: Optional[Tuple[int, int]],
) -> np.ndarray:
    """
    map_b1hw: [B,1,H,W] float (any range)
    returns: BGR canvas uint8 (colored)
    """
    tiles = []
    m = map_b1hw.detach().cpu().numpy()  # B,1,H,W
    for i in range(m.shape[0]):
        m_hw = m[i, 0]
        gray = _normalize_map_to_uint8(m_hw)        # H,W uint8
        heat = _apply_colormap(gray)                # H,W,3 BGR
        if resize_to is not None:
            rh, rw = resize_to
            heat = cv2.resize(heat, (rw, rh), interpolation=cv2.INTER_AREA)
        tiles.append(heat)

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
    return canvas

def tile_overlay_cv2(
    images_bchw: torch.Tensor,
    map_b1hw: torch.Tensor,
    cols: int,
    pad: int,
    resize_to: Optional[Tuple[int, int]],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Create tiled overlay of heatmap over input images.
    Returns BGR canvas.
    """
    tiles = []
    m = map_b1hw.detach().cpu().numpy()
    for i in range(images_bchw.size(0)):
        rgb = _tensor_to_rgb_uint8(images_bchw[i])  # RGB uint8
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        gray = _normalize_map_to_uint8(m[i, 0])
        heat = _apply_colormap(gray)  # BGR

        # Resize both consistently if needed
        if resize_to is not None:
            rh, rw = resize_to
            bgr = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_AREA)
            heat = cv2.resize(heat, (rw, rh), interpolation=cv2.INTER_AREA)

        overlay = cv2.addWeighted(bgr, 1.0 - alpha, heat, alpha, 0.0)
        tiles.append(overlay)

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
    return canvas


# -------------------- MODEL --------------------

class DenseRegressorDeepLabV3(nn.Module):
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, activation: str = "softplus"):
        super().__init__()

        if backbone == "resnet50":
            self.net = deeplabv3_resnet50(weights="DEFAULT" if pretrained else None)
            in_ch = self.net.classifier[-1].in_channels
        elif backbone == "mobilenetv3":
            self.net = deeplabv3_mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
            in_ch = self.net.classifier[-1].in_channels
        else:
            raise ValueError("backbone must be 'resnet50' or 'mobilenetv3'")

        self.net.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)

        if activation in (None, "none"):
            self.act = nn.Identity()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "softplus":
            self.act = nn.Softplus(beta=1.0)
        else:
            raise ValueError("activation must be 'none'|'relu'|'softplus'")

        self._last_map = None

    def forward(self, x: torch.Tensor, return_map: bool = False):
        out = self.net(x)["out"]  # (B,1,h,w)
        out = self.act(out)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)  # (B,1,H,W)

        self._last_map = out
        pred = out.mean(dim=(2, 3))  # (B,1)

        if return_map:
            return pred, out
        return pred

    def get_last_map(self) -> torch.Tensor:
        if self._last_map is None:
            raise RuntimeError("No cached map yet. Run a forward pass first.")
        return self._last_map


# -------------------- TRAIN / VAL LOOP --------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    label_smooth_min: float,
    label_smooth_max: float,
):
    model.train()
    running_loss = 0.0
    n_samples = 0

    preds_all = []
    targets_all = []

    for bidx, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)  # (B,1)

        # label jitter for loss only
        with torch.no_grad():
            multipliers = torch.empty((targets.size(0), 1), device=targets.device).uniform_(label_smooth_min, label_smooth_max)
            smooth_targets = targets * multipliers

        # Forward with map (for TV regularizer + debug viz)
        preds, dense_map = model(images, return_map=True)  # preds (B,1), dense_map (B,1,H,W)

        loss = criterion(preds, smooth_targets)
        if USE_TV_REG:
            loss = loss + (TV_LAMBDA * tv_loss(dense_map))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Debug windows ---
        if SHOW_DEBUG_WINDOWS and (bidx % SHOW_EVERY_N_BATCHES == 0):
            # Use a no-grad forward for clean viz timing; reuse current dense_map is also fine
            with torch.no_grad():
                _, dense_map_viz = model(images, return_map=True)

            img_canvas = tile_rgb_images_cv2(
                images, cols=DEBUG_TILE_COLS, pad=DEBUG_TILE_PAD, resize_to=DEBUG_TILE_RESIZE_TO
            )
            map_canvas = tile_heatmaps_cv2(
                dense_map_viz, cols=DEBUG_TILE_COLS, pad=DEBUG_TILE_PAD, resize_to=DEBUG_TILE_RESIZE_TO
            )
            cv2.imshow("train batch (images)", img_canvas)
            cv2.imshow("train batch (maps)", map_canvas)

            if SHOW_OVERLAY_WINDOW:
                overlay_canvas = tile_overlay_cv2(
                    images, dense_map_viz,
                    cols=DEBUG_TILE_COLS, pad=DEBUG_TILE_PAD,
                    resize_to=DEBUG_TILE_RESIZE_TO,
                    alpha=OVERLAY_ALPHA
                )
                cv2.imshow("train batch (overlay)", overlay_canvas)

            cv2.waitKey(1)

        preds_all.append(preds.detach().cpu().numpy().reshape(-1))
        targets_all.append(targets.detach().cpu().numpy().reshape(-1))

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / max(1, n_samples)
    preds_all = np.concatenate(preds_all, axis=0) if preds_all else np.array([])
    targets_all = np.concatenate(targets_all, axis=0) if targets_all else np.array([])

    if preds_all.size > 0:
        train_r2 = compute_r2(targets_all, preds_all)
        train_mae = compute_mae(targets_all, preds_all)
    else:
        train_r2, train_mae = 0.0, 0.0

    print(f"Epoch {epoch} train loss: {epoch_loss:.6f}  R2: {train_r2:.4f}  MAE: {train_mae:.6f}")
    return epoch_loss, train_r2, train_mae


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, epoch):
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

        preds_all.append(preds.detach().cpu().numpy().reshape(-1))
        targets_all.append(targets.detach().cpu().numpy().reshape(-1))

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / max(1, n_samples)

    preds_all = np.concatenate(preds_all, axis=0) if preds_all else np.array([])
    targets_all = np.concatenate(targets_all, axis=0) if targets_all else np.array([])

    if preds_all.size > 0:
        val_r2 = compute_r2(targets_all, preds_all)
        val_mae = compute_mae(targets_all, preds_all)
    else:
        val_r2, val_mae = 0.0, 0.0

    print(f"Epoch {epoch} val loss: {epoch_loss:.6f}  R2: {val_r2:.4f}  MAE: {val_mae:.6f}")
    return epoch_loss, val_r2, val_mae


# -------------------- MAIN TRAIN FUNCTION --------------------

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

    train_ds = ImageRegDataset(df=df, split="train", root_dir=root_img_dir, transform=get_transforms("train"))
    val_ds   = ImageRegDataset(df=df, split="val",   root_dir=root_img_dir, transform=get_transforms("val"))
    test_ds  = ImageRegDataset(df=df, split="test",  root_dir=root_img_dir, transform=get_transforms("test"))

    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS
    )

    model = DenseRegressorDeepLabV3(backbone=BACKBONE, pretrained=True, activation="softplus").to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP, gamma=STEP_LR_GAMMA)

    best_val_loss = float("inf")

    results_csv_path = out_dir / "epoch_results.csv"
    pd.DataFrame(columns=[
        "epoch",
        "train_loss", "train_r2", "train_mae",
        "val_loss", "val_r2", "val_mae"
    ]).to_csv(results_csv_path, index=False)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_r2, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, LABEL_SMOOTH_MIN, LABEL_SMOOTH_MAX
        )
        val_loss, val_r2, val_mae = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 50 == 0:
            ckpt_path = out_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"Saved new best model at epoch {epoch} with val_loss={val_loss:.6f}")

        pd.DataFrame([{
            "epoch": epoch,
            "train_loss": train_loss,
            "train_r2": train_r2,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_r2": val_r2,
            "val_mae": val_mae,
        }]).to_csv(results_csv_path, mode="a", header=False, index=False)

    test_loss, test_r2, test_mae = validate(model, test_loader, criterion, device, epoch="test")
    print(f"Final test loss: {test_loss:.6f}  R2: {test_r2:.4f}  MAE: {test_mae:.6f}")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    pd.DataFrame([{
        "epoch": "test",
        "train_loss": np.nan,
        "train_r2": np.nan,
        "train_mae": np.nan,
        "val_loss": test_loss,
        "val_r2": test_r2,
        "val_mae": test_mae,
    }]).to_csv(results_csv_path, mode="a", header=False, index=False)

    # Close OpenCV windows at end (optional)
    if SHOW_DEBUG_WINDOWS:
        cv2.destroyAllWindows()

    return history


# -------------------- CLI --------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dense-map regression model (DeepLabV3) from CSV metadata")
    parser.add_argument("--csv", type=str, default=str(METADATA_CSV), help="path to metadata csv")
    parser.add_argument("--root", type=str, default=str(ROOT_IMG_DIR), help="root dir (optional)")
    parser.add_argument("--out", type=str, default=str(OUT_DIR), help="output directory")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--backbone", type=str, default=BACKBONE, choices=["resnet50", "mobilenetv3"])
    parser.add_argument("--no-tv", action="store_true", help="disable TV smoothness regularizer")
    parser.add_argument("--no-debug", action="store_true", help="disable OpenCV debug windows")
    args = parser.parse_args()

    BACKBONE = args.backbone
    if args.no_tv:
        USE_TV_REG = False
    if args.no_debug:
        SHOW_DEBUG_WINDOWS = False

    run_training(
        csv_path=Path(args.csv),
        root_img_dir=Path(args.root),
        out_dir=Path(args.out),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
    )
