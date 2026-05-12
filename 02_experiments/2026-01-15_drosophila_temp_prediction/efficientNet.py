from __future__ import annotations

import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode


import cv2
import numpy as np

def denormalize_batch(x: torch.Tensor, mean, std) -> torch.Tensor:
    """
    x: [B,3,H,W] normalized tensor
    returns: [B,3,H,W] in [0,1]
    """
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    x = x * std + mean
    return x.clamp(0, 1)

def make_tile(bchw_01: torch.Tensor, ncols: int = 8, pad: int = 2) -> np.ndarray:
    """
    bchw_01: [B,3,H,W] float in [0,1] (CPU or GPU ok)
    returns: uint8 BGR image for OpenCV
    """
    x = bchw_01.detach().cpu()
    b, c, h, w = x.shape
    ncols = min(ncols, b)
    nrows = int(np.ceil(b / ncols))

    # pad each image with white border
    x = (x * 255.0).byte()  # [B,3,H,W]
    x = x.permute(0, 2, 3, 1).numpy()  # [B,H,W,3] RGB

    # Create canvas
    tile_h = nrows * h + (nrows - 1) * pad
    tile_w = ncols * w + (ncols - 1) * pad
    canvas = np.ones((tile_h, tile_w, 3), dtype=np.uint8) * 255

    for i in range(b):
        r = i // ncols
        c_ = i % ncols
        y0 = r * (h + pad)
        x0 = c_ * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = x[i]

    # RGB -> BGR for OpenCV
    canvas = canvas[:, :, ::-1]
    return canvas

def show_batch_cv2(
    images: torch.Tensor,
    mean, std,
    win_name: str = "train_batch",
    every_ms: int = 1,
    ncols: int = 8,
):
    """
    images: [B,3,H,W] normalized tensor
    """
    imgs = denormalize_batch(images, mean, std)
    tile = make_tile(imgs, ncols=ncols)
    print(tile.shape)
    cv2.imshow("test", tile)
    cv2.waitKey(every_ms)  # keep small; 1 is usually fine


# ---------------------------
# Dataset + label encoders
# ---------------------------

@dataclass
class LabelEncoders:
    sex_to_id: Dict[str, int]
    species_to_id: Dict[str, int]
    temp_to_id: Dict[str, int]


class FlyCsvDataset(Dataset):
    """
    CSV columns expected:
      - IMAGE_FILENAME: path to image (absolute or relative to root_dir if provided)
      - species: string
      - sex: string
      - temp: numeric or string (rearing temperature)
      - split: "train" | "val" | "test"

    Returns:
      image: FloatTensor [3,H,W]
      sex: LongTensor scalar
      species: LongTensor scalar
      temp: LongTensor scalar
    """
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        transform=None,
        root_dir: Optional[str | Path] = None,
        encoders: Optional[LabelEncoders] = None,
        strict_paths: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.split = split
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.strict_paths = strict_paths

        df = pd.read_csv(self.csv_path)

        required = {"IMAGE_FILENAME", "species", "sex", "temp", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        df = df[df["split"].astype(str).str.lower() == split.lower()].copy()
        if len(df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {csv_path}")

        df["species"] = df["species"].astype(str).str.strip()
        df["sex"] = df["sex"].astype(str).str.strip()
        df["temp"] = df["temp"].astype(str).str.strip()
        df["IMAGE_FILENAME"] = df["IMAGE_FILENAME"].astype(str).str.strip()

        # IMPORTANT: build encoders ONLY from training split, then reuse for val/test
        if encoders is None:
            sex_vals = sorted(df["sex"].unique().tolist())
            species_vals = sorted(df["species"].unique().tolist())
            temp_vals = sorted(df["temp"].unique().tolist())
            self.encoders = LabelEncoders(
                sex_to_id={k: i for i, k in enumerate(sex_vals)},
                species_to_id={k: i for i, k in enumerate(species_vals)},
                temp_to_id={k: i for i, k in enumerate(temp_vals)},
            )
        else:
            self.encoders = encoders

        def map_or_fail(val: str, mapping: Dict[str, int], field: str) -> int:
            if val not in mapping:
                raise ValueError(
                    f"Found unseen {field}='{val}' in split='{split}'. "
                    f"Known {field} values: {sorted(mapping.keys())}"
                )
            return mapping[val]

        df["sex_id"] = df["sex"].map(lambda v: map_or_fail(v, self.encoders.sex_to_id, "sex"))
        df["species_id"] = df["species"].map(lambda v: map_or_fail(v, self.encoders.species_to_id, "species"))
        df["temp_id"] = df["temp"].map(lambda v: map_or_fail(v, self.encoders.temp_to_id, "temp"))

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute() and self.root_dir is not None:
            path = self.root_dir / path
        return path

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row["IMAGE_FILENAME"])

        if self.strict_paths and not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        with Image.open(img_path) as im:
            # Handle transparency explicitly
            if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
                bg = Image.new("RGB", im.size, (255, 255, 255))  # white background
                bg.paste(im, mask=im.split()[-1])  # alpha channel as mask
                im = bg
            else:
                im = im.convert("RGB")

            im_t = self.transform(im) if self.transform is not None else T.ToTensor()(im)
        sex = torch.tensor(int(row["sex_id"]), dtype=torch.long)
        species = torch.tensor(int(row["species_id"]), dtype=torch.long)
        temp = torch.tensor(int(row["temp_id"]), dtype=torch.long)
        return im_t, sex, species, temp


# ---------------------------
# Augmentations
# ---------------------------

def random_blur_or_sharpness():
    return random.choice([
        T.GaussianBlur(kernel_size=random.choice([3, 15]), sigma=(0.1, 4.0)),
        T.RandomAdjustSharpness(sharpness_factor=random.uniform(0.7, 1.8), p=1.0),
    ])


def random_quadrant_rotation(img):
    angle = random.choice([0, 90, 180, 270])
    return torchvision.transforms.functional.rotate(img, angle, fill=(255, 255, 255))

def pad_to_square_white(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img

    size = max(w, h)
    pad_left   = (size - w) // 2
    pad_right  = size - w - pad_left
    pad_top    = (size - h) // 2
    pad_bottom = size - h - pad_top

    return ImageOps.expand(
        img,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=(255, 255, 255)
    )

# ---------------------------
# Model: EfficientNet multitask
# ---------------------------

@dataclass
class MultiTaskOutput:
    sex_logits: torch.Tensor       # [B, 2]
    species_logits: torch.Tensor   # [B, N_species]
    temp_logits: torch.Tensor      # [B, N_temp]


class EfficientNetMultiTask(nn.Module):
    """
    EfficientNet backbone with three heads:
      - sex: 2 classes
      - species: N classes
      - temp: T classes (rearing temperature)
    """
    def __init__(
        self,
        backbone: str = "efficientnet_b3",  # b0..b7, v2_s/m/l (depending on torchvision)
        pretrained: bool = True,
        num_sex: int = 2,
        num_species: int = 5,
        num_temp: int = 2,
        dropout_p: float = 0.2,
    ):
        super().__init__()

        if backbone == "efficientnet_b0":
            net = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
        elif backbone == "efficientnet_b3":
            net = torchvision.models.efficientnet_b3(
                weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            )
        elif backbone == "efficientnet_b5":
            net = torchvision.models.efficientnet_b5(
                weights=torchvision.models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
            )
        elif backbone == "efficientnet_v2_s":
            net = torchvision.models.efficientnet_v2_s(
                weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            )
        elif backbone == "efficientnet_v2_m":
            net = torchvision.models.efficientnet_v2_m(
                weights=torchvision.models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported EfficientNet backbone: {backbone}")

        # EfficientNet in torchvision: features -> avgpool -> classifier
        self.backbone = net.features
        self.avgpool = net.avgpool  # AdaptiveAvgPool2d(1)
        feat_dim = net.classifier[1].in_features

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.sex_head = nn.Linear(feat_dim, num_sex)
        self.species_head = nn.Linear(feat_dim, num_species)
        self.temp_head = nn.Linear(feat_dim, num_temp)

        for head in (self.sex_head, self.species_head, self.temp_head):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> MultiTaskOutput:
        x = self.backbone(x)         # [B, C, H', W']
        x = self.avgpool(x)          # [B, C, 1, 1]
        x = torch.flatten(x, 1)      # [B, C]
        x = self.dropout(x)
        return MultiTaskOutput(
            sex_logits=self.sex_head(x),
            species_logits=self.species_head(x),
            temp_logits=self.temp_head(x),
        )


# ---------------------------
# Loss + evaluation
# ---------------------------

def multitask_loss(
    out: MultiTaskOutput,
    sex_target: torch.Tensor,
    species_target: torch.Tensor,
    temp_target: torch.Tensor,
    w_sex: float = 1.0,
    w_species: float = 1.0,
    w_temp: float = 1.0,
    sex_class_weights: Optional[torch.Tensor] = None,
    species_class_weights: Optional[torch.Tensor] = None,
    temp_class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_sex = F.cross_entropy(out.sex_logits, sex_target, weight=sex_class_weights)
    loss_species = F.cross_entropy(out.species_logits, species_target, weight=species_class_weights)
    loss_temp = F.cross_entropy(out.temp_logits, temp_target, weight=temp_class_weights)
    loss = w_sex * loss_sex + w_species * loss_species + w_temp * loss_temp
    return loss, {
        "loss": float(loss.detach().cpu()),
        "loss_sex": float(loss_sex.detach().cpu()),
        "loss_species": float(loss_species.detach().cpu()),
        "loss_temp": float(loss_temp.detach().cpu()),
    }


@torch.no_grad()
def run_eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum = loss_sex_sum = loss_species_sum = loss_temp_sum = 0.0
    sex_correct = species_correct = temp_correct = all_correct = 0
    n = 0

    for images, sex, species, temp in loader:
        images = images.to(device, non_blocking=True)
        sex = sex.to(device, non_blocking=True)
        species = species.to(device, non_blocking=True)
        temp = temp.to(device, non_blocking=True)

        out = model(images)
        _, m = multitask_loss(out, sex, species, temp)

        bs = images.size(0)
        n += bs
        loss_sum += m["loss"] * bs
        loss_sex_sum += m["loss_sex"] * bs
        loss_species_sum += m["loss_species"] * bs
        loss_temp_sum += m["loss_temp"] * bs

        ps = out.sex_logits.argmax(dim=1)
        pp = out.species_logits.argmax(dim=1)
        pt = out.temp_logits.argmax(dim=1)
        sex_correct += (ps == sex).sum().item()
        species_correct += (pp == species).sum().item()
        temp_correct += (pt == temp).sum().item()
        all_correct += ((ps == sex) & (pp == species) & (pt == temp)).sum().item()

    return {
        "loss": loss_sum / max(n, 1),
        "loss_sex": loss_sex_sum / max(n, 1),
        "loss_species": loss_species_sum / max(n, 1),
        "loss_temp": loss_temp_sum / max(n, 1),
        "sex_acc": sex_correct / max(n, 1),
        "species_acc": species_correct / max(n, 1),
        "temp_acc": temp_correct / max(n, 1),
        "all_acc": all_correct / max(n, 1),
    }

@torch.no_grad()
def save_predictions_csv(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    encoders: LabelEncoders,
    out_csv: str | Path,
):
    """
    Save per-image GT and predictions to CSV:
      filename, sex_GT, species_GT, temp_GT, sex_Pred, species_Pred, temp_Pred

    Assumes loader.shuffle == False so dataset order matches iteration order.
    """
    model.eval()
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ds = loader.dataset
    if not hasattr(ds, "df"):
        raise ValueError("Dataset must have attribute .df with IMAGE_FILENAME/species/sex/temp columns.")

    id_to_sex = {v: k for k, v in encoders.sex_to_id.items()}
    id_to_species = {v: k for k, v in encoders.species_to_id.items()}
    id_to_temp = {v: k for k, v in encoders.temp_to_id.items()}

    rows = []
    seen = 0

    for images, sex, species, temp in loader:
        bs = images.size(0)

        images = images.to(device, non_blocking=True)
        sex = sex.to(device, non_blocking=True)
        species = species.to(device, non_blocking=True)
        temp = temp.to(device, non_blocking=True)

        out = model(images)
        sex_pred = out.sex_logits.argmax(dim=1).detach().cpu().numpy()
        sp_pred = out.species_logits.argmax(dim=1).detach().cpu().numpy()
        temp_pred = out.temp_logits.argmax(dim=1).detach().cpu().numpy()

        batch_df = ds.df.iloc[seen:seen + bs]
        if len(batch_df) != bs:
            raise RuntimeError("Dataset/loader length mismatch while writing prediction CSV.")

        for i in range(bs):
            rows.append({
                "filename": str(batch_df.iloc[i]["IMAGE_FILENAME"]),
                "sex_GT": str(batch_df.iloc[i]["sex"]),
                "species_GT": str(batch_df.iloc[i]["species"]),
                "temp_GT": str(batch_df.iloc[i]["temp"]),
                "sex_Pred": id_to_sex[int(sex_pred[i])],
                "species_Pred": id_to_species[int(sp_pred[i])],
                "temp_Pred": id_to_temp[int(temp_pred[i])],
            })

        seen += bs

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[pred] wrote {out_csv} ({len(rows)} rows)")


# ---------------------------
# Checkpointing: BEST + LAST only
# ---------------------------

def save_checkpoint(
    out_dir: str | Path,
    kind: str,  # "best" | "last"
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    encoders: Optional[LabelEncoders],
    extra: Optional[Dict] = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "encoders": {
            "sex_to_id": encoders.sex_to_id,
            "species_to_id": encoders.species_to_id,
            "temp_to_id": encoders.temp_to_id,
        } if encoders is not None else None,
        "extra": extra or {},
    }

    path = out_dir / f"{kind}.pt"
    torch.save(payload, path)
    print(f"\n[ckpt] saved {kind}: {path}")



# ---------------------------
# Training loop
# ---------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epochs: int,
    out_dir: str | Path,
    encoders: Optional[LabelEncoders],
    w_sex: float = 1.0,
    w_species: float = 1.0,
    w_temp: float = 1.0,
    grad_clip_norm: Optional[float] = 1.0,
    use_amp: bool = True,
):
    best_val_loss = float("inf")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = loss_sex_sum = loss_species_sum = loss_temp_sum = 0.0
        sex_correct = species_correct = temp_correct = all_correct = 0
        n = 0
        t0 = time.time()

        for step, (images, sex, species, temp) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            sex = sex.to(device, non_blocking=True)
            species = species.to(device, non_blocking=True)
            temp = temp.to(device, non_blocking=True)

            if step == 1 or step % 10 == 0:
                show_batch_cv2(
                    images,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    win_name="train_batch",
                    every_ms=1,
                    ncols=4,
                )
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(images)
                loss, m = multitask_loss(out, sex, species, temp, w_sex=w_sex, w_species=w_species, w_temp=w_temp)

            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            n += bs
            loss_sum += m["loss"] * bs
            loss_sex_sum += m["loss_sex"] * bs
            loss_species_sum += m["loss_species"] * bs
            loss_temp_sum += m["loss_temp"] * bs

            ps = out.sex_logits.argmax(dim=1)
            pp = out.species_logits.argmax(dim=1)
            pt = out.temp_logits.argmax(dim=1)
            sex_correct += (ps == sex).sum().item()
            species_correct += (pp == species).sum().item()
            temp_correct += (pt == temp).sum().item()
            all_correct += ((ps == sex) & (pp == species) & (pt == temp)).sum().item()

            if step == 1 or step % 5 == 0 or step == len(train_loader):
                lr = optimizer.param_groups[0]["lr"]
                it_s = step / max(time.time() - t0, 1e-9)
                print(
                    f"\rEpoch {epoch:03d}/{epochs:03d} "
                    f"[{step:04d}/{len(train_loader):04d}] "
                    f"lr={lr:.3e} "
                    f"loss={loss_sum/max(n,1):.4f} "
                    f"(sex={loss_sex_sum/max(n,1):.4f}, sp={loss_species_sum/max(n,1):.4f}, t={loss_temp_sum/max(n,1):.4f}) "
                    f"acc(sex={sex_correct/max(n,1):.3f}, sp={species_correct/max(n,1):.3f}, "
                    f"t={temp_correct/max(n,1):.3f}, all={all_correct/max(n,1):.3f}) "
                    f"{it_s:.1f} it/s",
                    end="",
                    flush=True,
                )

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        val_metrics = run_eval_epoch(model, val_loader, device)

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["loss"])

        train_metrics = {
            "loss": loss_sum / max(n, 1),
            "loss_sex": loss_sex_sum / max(n, 1),
            "loss_species": loss_species_sum / max(n, 1),
            "loss_temp": loss_temp_sum / max(n, 1),
            "sex_acc": sex_correct / max(n, 1),
            "species_acc": species_correct / max(n, 1),
            "temp_acc": temp_correct / max(n, 1),
            "all_acc": all_correct / max(n, 1),
        }

        print(
            f"\nEpoch {epoch:03d} done | "
            f"train loss={train_metrics['loss']:.4f} "
            f"sex={train_metrics['sex_acc']:.3f} sp={train_metrics['species_acc']:.3f} "
            f"t={train_metrics['temp_acc']:.3f} all={train_metrics['all_acc']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"sex={val_metrics['sex_acc']:.3f} sp={val_metrics['species_acc']:.3f} "
            f"t={val_metrics['temp_acc']:.3f} all={val_metrics['all_acc']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                out_dir=out_dir,
                kind="best",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                encoders=encoders,
                extra={"best_val_loss": best_val_loss},
            )
            save_predictions_csv(
                model=model,
                loader=val_loader,
                device=device,
                encoders=encoders,
                out_csv=Path(out_dir) / f"val_predictions_epoch_{epoch:03d}.csv",
            )

        save_checkpoint(
            out_dir=out_dir,
            kind="last",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            encoders=encoders,
            extra={"best_val_loss": best_val_loss},
        )


# ---------------------------
# Main
# ---------------------------

class RandomDownscaleUpscale:
    def __init__(self, scale_min=0.2, scale_max=1.0, interpolation=InterpolationMode.BILINEAR):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.interpolation = interpolation

    def __call__(self, img):
        # img: PIL Image or torch Tensor (C,H,W)
        s = random.uniform(self.scale_min, self.scale_max)

        if isinstance(img, torch.Tensor):
            _, h, w = img.shape
        else:
            w, h = img.size

        new_h = max(1, int(h * s))
        new_w = max(1, int(w * s))
        import torchvision.transforms.functional as TVF
        img = TVF.resize(img, (new_h, new_w), interpolation=self.interpolation)
        img = TVF.resize(img, (h, w), interpolation=self.interpolation)

        return img

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork")
    # ---- config ----
    csv_path = "droso_classif.csv"
    root_dir = None                  # set if IMAGE_FILENAME is relative
    out_dir = "checkpoints_temp_prediction"
    epochs = 50

    # Large image training: choose a bigger input size.
    # EfficientNet-B3 default is 300, B5 default is 456, B7 default is 600.
    # Pick something your GPU can handle. Start with 456.
    image_size = 456


    train_bs = 8                     # likely needs to be smaller for large images
    val_bs = 16
    num_workers = 4

    backbone = "efficientnet_b5"     # b3/b5 are good for large inputs

    # Use the pretrained weights' normalization
    if backbone == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    elif backbone == "efficientnet_b3":
        weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
    elif backbone == "efficientnet_b5":
        weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
    elif backbone == "efficientnet_v2_s":
        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    elif backbone == "efficientnet_v2_m":
        weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
    else:
        raise ValueError(f"Unsupported backbone for weights: {backbone}")

    mean = weights.transforms().mean
    std = weights.transforms().std
    normalize = T.Normalize(mean=mean, std=std)

    train_tf = T.Compose([

        T.Lambda(pad_to_square_white),
        # T.RandomErasing(p=0.25, scale=(0.01, 0.08), ratio=(0.3, 3.3), value=1.0),

        T.Resize((image_size, image_size)),
        # RandomDownscaleUpscale(0.2, 1.0),
        T.RandomHorizontalFlip(p=0.5),
        # T.Lambda(random_quadrant_rotation),
        T.RandomRotation(degrees=(0, 360), fill=(255, 255, 255)),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        T.RandomApply([T.Lambda(lambda img: random_blur_or_sharpness()(img))], p=0.4),
        T.ToTensor(),

        T.RandomErasing(p=0.25, scale=(0.01, 0.06), value=1.0),
        normalize,
    ])

    val_tf = T.Compose([

        T.Lambda(pad_to_square_white),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalize,
    ])

    # Datasets (encoders from train)
    train_ds = FlyCsvDataset(csv_path, split="train", transform=train_tf, root_dir=root_dir)
    enc = train_ds.encoders
    val_ds = FlyCsvDataset(csv_path, split="val", transform=val_tf, root_dir=root_dir, encoders=enc)
    test_ds = FlyCsvDataset(csv_path, split="test", transform=val_tf, root_dir=root_dir, encoders=enc)

    print("sex mapping:", enc.sex_to_id)
    print("species mapping:", enc.species_to_id)
    print("temp mapping:", enc.temp_to_id)
    print("train/val/test sizes:", len(train_ds), len(val_ds), len(test_ds))

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=train_bs, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_bs, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=val_bs, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Device + model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetMultiTask(
        backbone=backbone,
        pretrained=True,
        num_sex=len(enc.sex_to_id),
        num_species=len(enc.species_to_id),
        num_temp=len(enc.temp_to_id),
        dropout_p=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    # LR decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        out_dir=out_dir,
        encoders=enc,
        w_sex=1.0,
        w_species=1.0,
        w_temp=1.0,
        grad_clip_norm=1.0,
        use_amp=True,  # important for big images
    )

    # Test evaluation (LAST model in memory)
    test_metrics = run_eval_epoch(model, test_loader, device)
    save_predictions_csv(
        model=model,
        loader=test_loader,
        device=device,
        encoders=enc,
        out_csv=Path(out_dir) / "test_predictions_best.csv",
    )
    print("\nTEST:", test_metrics)
