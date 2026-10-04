# <keep your existing header/docstring>

import os
import math
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
from models import build_efficientnet, ScaleInputNet
from datetime import datetime



# -------------------- CONSTANTS / HYPERPARAMS --------------------
# Paths
ROOT_IMG_DIR = Path("00_data/02_resized")            # <- change me
METADATA_CSV = Path("metadata_enriched.csv")      # <- change me
# RESNET_ARCH = "50"
EFFNET_VARIANT = "v2_s"

# What the Gaussian head predicts: "logbf" (log biomass factor, the BEEomass model) or
# "logmass" (log dry mass straight from the image; a reviewer-requested comparison).
# The images are the same 224 px crops either way, so "logmass" gets no physical scale.
TARGET = "logbf"

# Weight of the resolution-consistency penalty (0 = off). A random quarter of each batch is
# also seen at reduced resolution or blurred; the penalty is the squared gap between its mu
# and the clean mu, in units of the clean sigma, with the clean prediction held fixed. The
# insect's physical size is unchanged, so a model that reads size from sharpness pays for it.
RES_CONSISTENCY = 0.0

# Feed physical scale (log L, log projected area) to the head as numbers (ScaleInputNet).
# For the reviewer comparison with --target logmass: a direct model that is told the size.
SCALE_INPUT = False

OUT_DIR = Path(f"01_runs/gaussian_logbf_effnet{EFFNET_VARIANT}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

# Training hyperparams
SEED = 42
# Raised from 32: the GPU was ~92% busy but only 5.8 of 16.4 GB in use, and the
# dataloader has ~2.5x headroom, so larger batches convert idle memory into speed.
BATCH_SIZE = 64
NUM_WORKERS = 8   # dataloader had headroom at 4; more keeps the larger batch fed
NUM_EPOCHS = 500
# Scaled with the batch increase. AdamW is largely invariant to gradient scale, so
# sqrt scaling (not the SGD linear rule) is the usual choice: 1e-4 * sqrt(64/32).
LR = 1.4e-4
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
STEP_LR_STEP = 100
STEP_LR_GAMMA = 0.5

# Image size and normalization (ResNet default expected normalization)
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Distributional head: the model predicts log BF ~ N(mu, sigma^2) as (mu, log sigma).
# No target jitter: under a likelihood loss, sigma would learn the injected noise.
# The first NLL_WARMUP_EPOCHS fit mu alone with MSE, so sigma does not inflate to
# absorb the error of an untrained mean. Validation always reports the NLL.
NLL_WARMUP_EPOCHS = 5
# Bounds on log sigma, to keep exp(-log sigma) finite early in training.
LOG_SIGMA_MIN = -7.0
LOG_SIGMA_MAX = 2.0
# z for the central 95% interval, used for the coverage diagnostic.
Z95 = 1.959964
#how much the images can be downscaled during augmentation. this is a special
# augmentation that also modifies the target
DOWNSCALING_MIN = 0.5

# Checkpoint cadence, and the window of the centred rolling median used to pick one.
CHECKPOINT_EVERY = 50
SELECTION_WINDOW = 11

# Per-batch OpenCV debug display; off by default, see --debug-tiles.
DEBUG_TILES = False
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

def split_output(out: torch.Tensor):
    """(B,2) head output -> mu (B,1), log sigma (B,1), with log sigma clamped."""
    return out[:, :1], out[:, 1:].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)

def gaussian_nll(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood of log BF, without the constant 0.5*log(2*pi)."""
    mu, log_sigma = split_output(out)
    return (log_sigma + 0.5 * ((y - mu) * torch.exp(-log_sigma)) ** 2).mean()

def mu_mse(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Warm-up loss: MSE on the mean only; the log sigma output gets no gradient."""
    return ((out[:, :1] - y) ** 2).mean()

METRICS = ("r2", "mae", "mdape", "male", "cov95", "sigma")


def summarise_predictions(out_all: np.ndarray, y_all: np.ndarray) -> dict:
    """Metrics from (N,2) outputs and (N,) log targets, as a dict keyed by METRICS.

    mdape: median absolute % error on dry mass, |M_hat / M - 1| * 100, with the median of the
        predicted distribution as M_hat. The headline accuracy figure.
    male: mean absolute log ratio on dry mass, mean |log(M_hat / M)|; exp(male) is the typical
        error factor. Symmetric in over- and underestimates, so it is the selection metric.
    Both are on mass for either target: L is measured, so for "logbf" the log-mass error is
    exactly 3x the log-BF error and needs no L here.
    r2, mae: for "logbf" on the BF scale (comparable with the point-regression runs), for
        "logmass" on log mass. Kept for continuity, not for decisions: on BF an MAE weights
        every specimen alike in absolute BF, which hides that the lightest are worst in %.
    cov95: share of targets inside mu +- 1.96 sigma (0.95 if calibrated). sigma: its mean.
    """
    if out_all.size == 0:
        return {k: 0.0 for k in METRICS}
    mu = out_all[:, 0]
    sigma = np.exp(np.clip(out_all[:, 1], LOG_SIGMA_MIN, LOG_SIGMA_MAX))
    point_true, point_pred = (np.exp(y_all), np.exp(mu)) if TARGET == "logbf" else (y_all, mu)
    log_mass_err = (3.0 if TARGET == "logbf" else 1.0) * (mu - y_all)
    return {
        "r2": compute_r2(point_true, point_pred),
        "mae": compute_mae(point_true, point_pred),
        "mdape": float(np.median(np.abs(np.expm1(log_mass_err))) * 100),
        "male": float(np.mean(np.abs(log_mass_err))),
        "cov95": float(np.mean(np.abs(y_all - mu) <= Z95 * sigma)),
        "sigma": float(np.mean(sigma)),
    }


def format_metrics(m: dict) -> str:
    return (f"R2: {m['r2']:.4f}  MAE: {m['mae']:.6f}  %err(median): {m['mdape']:.1f}  "
            f"|log ratio|: {m['male']:.4f}  cov95: {m['cov95']:.3f}  sigma: {m['sigma']:.4f}")


# -------------------- DATASET --------------------


def rot90_rand(img: Image.Image) -> Image.Image:
    """Rotate by 0/90/180/270 degrees randomly, keeping square canvas."""
    k = random.randint(0, 3)
    if k == 0:
        return img
    # PIL transpose is fast and avoids fill artifacts for multiples of 90
    return img.transpose([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][k - 1])

def make_mosaic_2x2_random(imgs, out_size: int = 224) -> Image.Image:
    """
    imgs: list of 4 PIL RGB images
    Creates a 2x2 mosaic in a square canvas out_size x out_size.
    Randomizes which image goes to which quadrant.
    """
    assert len(imgs) == 4
    half = out_size // 2

    # random assignment of images to quadrants
    random.shuffle(imgs)

    # resize each to half size (you can use BICUBIC or LANCZOS)
    tiles = [im.resize((half, half), Image.BICUBIC) for im in imgs]

    canvas = Image.new("RGB", (out_size, out_size), (255, 255, 255))
    positions = [(0, 0), (half, 0), (0, half), (half, half)]
    for tile, pos in zip(tiles, positions):
        canvas.paste(tile, pos)

    return canvas

class RandomDownscale:
    """
    Downscale image by factor in [min_factor, max_factor] (≤ 1),
    paste on white background of original size,
    return both image and scale factor.
    """

    def __init__(self, min_scale=None, max_scale=1.0):
        # Read the global at call time so --downscale-min can override it.
        min_scale = DOWNSCALING_MIN if min_scale is None else min_scale
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
    - `BF_cbrMG_MM`: biomass factor (float, > 0); the target is its log
    - `SPLIT`: 'train' | 'val' | 'test'
    Images are resolved as: root_dir / DATASET / IMAGE_FILENAME (unless IMAGE_FILENAME is absolute).
    """

    def __init__(self, df: pd.DataFrame, split: str, root_dir: Optional[Path] = None, transform=None):
        assert split in ("train", "val", "test")
        self.split = split
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.transform = transform
        # Stored on the instance: dataloader workers re-import this module, so a global
        # overridden from the CLI would read as its default inside them.
        self.target = TARGET

        # Filter by split
        self.df = df[df["SPLIT"] == split][df["IS_VALID"] == True].reset_index(drop=True).copy()

        # Column checks
        required_cols = {"IMAGE_FILENAME", "DATASET", "BF_cbrMG_MM", "DRYMASS_MG", "ROI_SIZE_MM", "AREA_MM2"}

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

        # BF = M^(1/3)/L, so a simulated shrink by `scale` gives (scale^3 M)^(1/3)/L = scale * BF.
        # NOT scale**3: BF is already a cube root, so cubing applies the exponent twice.
        # This deviates from Eq (6) of the manuscript, which is incorrect.
        # On the log scale the shrink is additive: log(scale * BF) = log BF + log scale.
        target = float(np.log(float(row["BF_cbrMG_MM"]) * scale))
        if self.target == "logmass":
            # Shrinking the specimen inside a fixed frame is a specimen s times smaller, i.e.
            # s^3 times lighter under isometry: the same physics as the BF rule above.
            target = float(np.log(float(row["DRYMASS_MG"])) + 3 * np.log(scale))
        # Physical scale as seen by a ScaleInputNet: the frame (L) is unchanged by the
        # downscale augmentation, while the specimen's projected area shrinks by scale^2.
        scale_in = torch.tensor([np.log(float(row["ROI_SIZE_MM"])),
                                 np.log(float(row["AREA_MM2"])) + 2 * np.log(scale)], dtype=torch.float32)
        return img, scale_in, torch.tensor(target, dtype=torch.float32)

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


def apply_random_blur_or_sharpness(img):
    """Module-level wrapper: a local lambda cannot be pickled for dataloader workers."""
    return random_blur_or_sharpness()(img)


def get_transforms(split: str):
    if split == "train":
        train_transforms = T.Compose([

            T.RandomHorizontalFlip(p=0.5),

            T.Lambda(random_quadrant_rotation),
            # Color jitter (brightness, contrast, saturation, hue)
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            # Gaussian blur (kernel size chosen relative to image size)

            T.Lambda(apply_random_blur_or_sharpness),

            T.ToTensor(),
            # T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            # Clamp01(),
            # T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            # Clamp01(),
            # T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return train_transforms
    else:
        # val/test: deterministic scaling / center crop
        return T.Compose([
            T.ToTensor(),
            # T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])



# -------------------- TRAIN / VAL LOOP --------------------

def degrade_resolution(x: torch.Tensor) -> torch.Tensor:
    """Per image, either resample down by 1.5-16x and back up, or Gaussian-blur with sigma 0.5-8 px."""
    H, W = x.shape[-2:]
    out = []
    for img in x:
        if random.random() < 0.5:
            f = math.exp(random.uniform(math.log(1.5), math.log(16)))
            small = nn.functional.interpolate(img[None], size=(max(4, round(H / f)), max(4, round(W / f))), mode="area")
            img = nn.functional.interpolate(small, size=(H, W), mode="bicubic", align_corners=False)[0].clamp(0, 1)
        else:
            sig = random.uniform(0.5, 8.0)
            k = 2 * math.ceil(3 * sig) + 1
            img = F.gaussian_blur(img, [k, k], [sig, sig])
        out.append(img)
    return torch.stack(out)


def call_model(model, images, scales):
    """Scale-input models take (images, scales); the others ignore the scales."""
    return model(images, scales) if isinstance(model, ScaleInputNet) else model(images)


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    n_samples = 0

    # accumulators for metrics
    preds_all = []
    targets_all = []

    for images, scales, targets in dataloader:
        if DEBUG_TILES:
            # Costs ~24 ms/batch (~13% of an epoch at batch 32) - opt in only.
            tile_images_cv2(images, cols=8, pad=2, to_bgr=True, resize_to=(128, 128),
                            window_name="train batch", show=True)
        images = images.to(device, non_blocking=True)
        scales = scales.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)  # shape (B,1), log BF

        if RES_CONSISTENCY > 0:
            n_deg = max(1, images.size(0) // 4)  # a quarter keeps batch 64 within 16 GB
            out = call_model(model, torch.cat([images, degrade_resolution(images[:n_deg])]),
                             torch.cat([scales, scales[:n_deg]]))
            preds, preds_deg = out[:images.size(0)], out[images.size(0):]
            mu_c, log_sigma_c = split_output(preds[:n_deg].detach())
            gap = (preds_deg[:, :1] - mu_c) * torch.exp(-log_sigma_c)
            loss = criterion(preds, targets) + RES_CONSISTENCY * (gap ** 2).mean()
        else:
            preds = call_model(model, images, scales)  # shape (B,2): mu, log sigma
            loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy().reshape(-1))

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / max(1, n_samples)

    preds_all = np.concatenate(preds_all, axis=0) if len(preds_all) > 0 else np.empty((0, 2))
    targets_all = np.concatenate(targets_all, axis=0) if len(targets_all) > 0 else np.array([])
    metrics = summarise_predictions(preds_all, targets_all)
    print(f"Epoch {epoch} train loss: {epoch_loss:.6f}  {format_metrics(metrics)}")
    return epoch_loss, metrics

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    preds_all = []
    targets_all = []

    for images, scales, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        preds = call_model(model, images, scales.to(device, non_blocking=True))
        loss = criterion(preds, targets)

        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy().reshape(-1))

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / max(1, n_samples)

    preds_all = np.concatenate(preds_all, axis=0) if len(preds_all) > 0 else np.empty((0, 2))
    targets_all = np.concatenate(targets_all, axis=0) if len(targets_all) > 0 else np.array([])
    metrics = summarise_predictions(preds_all, targets_all)
    print(f"Epoch {epoch} val loss: {epoch_loss:.6f}  {format_metrics(metrics)}")
    return epoch_loss, metrics

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
    show: bool = False,
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


def select_checkpoint(out_dir: Path, val_losses, window: int = 11, every: int = 50):
    """Pick a checkpoint by a smoothed validation metric (lower is better) rather than its raw minimum.

    Returns (epoch, path) and writes `selected_model.pt`. Because checkpoints are only
    written every `every` epochs, the nearest saved epoch to the smoothed argmin is used;
    the offset is reported so it is never silent.
    """
    import shutil
    s = pd.Series(val_losses)
    if len(s) < 2:
        return None, None
    smooth = s.rolling(window, center=True, min_periods=1).median()
    target = int(smooth.idxmin()) + 1                      # epochs are 1-based
    saved = sorted(int(p.stem.replace("checkpoint_epoch", ""))
                   for p in out_dir.glob("checkpoint_epoch*.pt"))
    if not saved:
        return target, None
    nearest = min(saved, key=lambda e: abs(e - target))
    src = out_dir / f"checkpoint_epoch{nearest}.pt"
    data = torch.load(src, map_location="cpu")
    state = data["model_state_dict"] if isinstance(data, dict) and "model_state_dict" in data else data
    torch.save(state, out_dir / "selected_model.pt")
    if nearest != target:
        print(f"  smoothed argmin at epoch {target}; nearest saved checkpoint is {nearest} "
              f"(offset {nearest - target:+d}). Lower --checkpoint-every to reduce this.", flush=True)
    return nearest, out_dir / "selected_model.pt"


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
    if SCALE_INPUT:
        model = ScaleInputNet(variant=EFFNET_VARIANT, pretrained=True, n_outputs=2)
        sc = np.log(train_ds.df[["ROI_SIZE_MM", "AREA_MM2"]].to_numpy(dtype=float))
        model.scale_mean.copy_(torch.tensor(sc.mean(0), dtype=torch.float32))
        model.scale_std.copy_(torch.tensor(sc.std(0), dtype=torch.float32))
        model = model.to(device)
    else:
        model = build_efficientnet(variant = EFFNET_VARIANT, pretrained=True, n_outputs=2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP, gamma=STEP_LR_GAMMA)

    best_val_loss = float('inf')

    # prepare epoch results CSV
    results_csv_path = out_dir / "epoch_results.csv"
    # write header
    # train_loss is the warm-up MSE for the first NLL_WARMUP_EPOCHS, the NLL after;
    # val_loss is always the NLL, so checkpoint selection compares like with like.
    columns = (["epoch", "train_loss"] + [f"train_{k}" for k in METRICS]
               + ["val_loss"] + [f"val_{k}" for k in METRICS])
    pd.DataFrame(columns=columns).to_csv(results_csv_path, index=False)

    history = {"train_loss": [], "val_loss": [], "val_male": []}

    for epoch in range(1, num_epochs + 1):

        # images: [B,3,H,W] float

        train_criterion = mu_mse if epoch <= NLL_WARMUP_EPOCHS else gaussian_nll
        train_loss, train_m = train_one_epoch(model, train_loader, optimizer, train_criterion, device, epoch)
        val_loss, val_m = validate(model, val_loader, gaussian_nll, device, epoch)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_male"].append(val_m["male"])

        # checkpoint
        ckpt_path = out_dir / f"checkpoint_epoch{epoch}.pt"
        if epoch % CHECKPOINT_EVERY == 0:
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
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
        row = {c: row[c] for c in columns}
        row_df = pd.DataFrame([row])
        row_df.to_csv(results_csv_path, mode="a", header=False, index=False)

    # -------- model selection --------
    # The per-epoch validation loss is noise-dominated once the curve plateaus: its
    # epoch-to-epoch SD is comparable to the entire remaining drift, so argmin over
    # 500 epochs picks whichever epoch happened to land in a noise dip. Selecting on
    # a centred rolling median of the validation loss instead. This cannot be done
    # online (it needs the future), so it runs here, against the periodic checkpoints.
    # For the Gaussian head, select on accuracy, not val NLL: sigma shrinks as the model
    # fits the training set, so the NLL favours early, less accurate epochs. Their
    # overconfidence is corrected afterwards by 04-calibrate.py. Accuracy here is the mean
    # absolute log ratio on mass (a symmetric % error), not MAE on BF, which hides that
    # the lightest specimens are the worst in relative terms.
    sel_epoch, sel_path = select_checkpoint(out_dir, history["val_male"],
                                            window=SELECTION_WINDOW,
                                            every=CHECKPOINT_EVERY)
    if sel_path is not None:
        print(f"Selected checkpoint: epoch {sel_epoch} -> {sel_path.name} "
              f"(smoothed val |log ratio| argmin, window {SELECTION_WINDOW})", flush=True)

    # final test evaluation
    test_loss, test_m = validate(model, test_loader, gaussian_nll, device, epoch="test")
    print(f"Final test loss (last epoch's model): {test_loss:.6f}  {format_metrics(test_m)}")

    # save history (optional)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False)

    # optionally append final test row into epoch_results.csv as epoch='test'
    final_row = {"epoch": "test", "train_loss": np.nan, "val_loss": test_loss,
                 **{f"train_{k}": np.nan for k in METRICS}, **{f"val_{k}": v for k, v in test_m.items()}}
    final_row = {c: final_row[c] for c in columns}
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
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--debug-tiles", action="store_true",
                        help="Show each training batch in an OpenCV window (slow: ~24 ms/batch)")
    parser.add_argument("--downscale-min", type=float, default=DOWNSCALING_MIN,
                        help="Lower bound of the scale-augmentation factor (1.0 disables it)")
    parser.add_argument("--scale-input", action="store_true",
                        help="feed log L and log projected area to the head (ScaleInputNet)")
    parser.add_argument("--target", choices=["logbf", "logmass"], default=TARGET,
                        help="predict log BF (default) or log dry mass directly from the image")
    parser.add_argument("--res-consistency", type=float, default=RES_CONSISTENCY,
                        help="weight of the penalty on mu changing when resolution drops (0 = off)")
    parser.add_argument("--nll-warmup", type=int, default=NLL_WARMUP_EPOCHS,
                        help="Epochs of MSE on the mean before switching to the Gaussian NLL (0 = none)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed; vary it to measure run-to-run variance")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                        help="Epoch interval at which checkpoints are written")
    parser.add_argument("--selection-window", type=int, default=SELECTION_WINDOW,
                        help="Centred rolling-median window used to select a checkpoint (1 = raw argmin)")
    args = parser.parse_args()

    # Override module-level constants so the augmentation is sweepable without edits.
    DEBUG_TILES = args.debug_tiles
    NUM_WORKERS = args.num_workers
    DOWNSCALING_MIN = args.downscale_min
    NLL_WARMUP_EPOCHS = args.nll_warmup
    TARGET = args.target
    SCALE_INPUT = args.scale_input
    RES_CONSISTENCY = args.res_consistency
    SEED = args.seed
    CHECKPOINT_EVERY = args.checkpoint_every
    SELECTION_WINDOW = args.selection_window
    print(f"seed={SEED} augmentation: downscale_min={DOWNSCALING_MIN} "
          f"label_smooth=off gaussian_noise=off elastic=off "
          f"target={TARGET} ~ N(mu, sigma) nll_warmup={NLL_WARMUP_EPOCHS} "
          f"res_consistency={RES_CONSISTENCY} scale_input={SCALE_INPUT}", flush=True)

    run_training(csv_path=Path(args.csv), root_img_dir=Path(args.root), out_dir=Path(args.out),
                 batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr,
                 num_workers=args.num_workers)  # passed explicitly: the default arg is
                 # bound at definition time, so overriding the global would be ignored
