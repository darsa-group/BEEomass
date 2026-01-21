from __future__ import annotations
import numpy as np
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Utils
# ---------------------------

def pad_to_square_white(img: Image.Image) -> Image.Image:
    """Pad the shorter side to make a square, using white background."""
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(255, 255, 255))


def open_rgb_white_bg(path: Path) -> Image.Image:
    """Open image, composite transparency onto white, and return RGB."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or ("transparency" in im.info):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    return im


# ---------------------------
# Dataset
# ---------------------------

@dataclass
class LabelEncoders:
    sex_to_id: Dict[str, int]
    species_to_id: Dict[str, int]


class FlyCsvDataset(Dataset):
    """
    CSV columns:
      IMAGE_FILENAME, species, sex, split
    Returns:
      image [3,H,W], sex_id, species_id
    """
    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        transform,
        root_dir: Optional[str | Path] = None,
        encoders: Optional[LabelEncoders] = None,
        strict_paths: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.split = split.lower()
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.strict_paths = strict_paths

        df = pd.read_csv(self.csv_path)
        required = {"IMAGE_FILENAME", "species", "sex", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        df = df[df["split"].astype(str).str.lower() == self.split].copy()
        if len(df) == 0:
            raise ValueError(f"No rows for split='{split}' in {csv_path}")

        df["IMAGE_FILENAME"] = df["IMAGE_FILENAME"].astype(str).str.strip()
        df["species"] = df["species"].astype(str).str.strip()
        df["sex"] = df["sex"].astype(str).str.strip()

        # Build encoders from training split only; reuse for val/test
        if encoders is None:
            sex_vals = sorted(df["sex"].unique().tolist())
            species_vals = sorted(df["species"].unique().tolist())
            self.encoders = LabelEncoders(
                sex_to_id={k: i for i, k in enumerate(sex_vals)},
                species_to_id={k: i for i, k in enumerate(species_vals)},
            )
        else:
            self.encoders = encoders

        def map_or_fail(val: str, mapping: Dict[str, int], field: str) -> int:
            if val not in mapping:
                raise ValueError(
                    f"Unseen {field}='{val}' in split='{split}'. Known: {sorted(mapping.keys())}"
                )
            return mapping[val]

        df["sex_id"] = df["sex"].map(lambda v: map_or_fail(v, self.encoders.sex_to_id, "sex"))
        df["species_id"] = df["species"].map(lambda v: map_or_fail(v, self.encoders.species_to_id, "species"))
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute() and self.root_dir is not None:
            path = self.root_dir / path
        return path

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve(row["IMAGE_FILENAME"])
        if self.strict_paths and not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        im = open_rgb_white_bg(img_path)
        im = self.transform(im)

        sex = torch.tensor(int(row["sex_id"]), dtype=torch.long)
        species = torch.tensor(int(row["species_id"]), dtype=torch.long)
        return im, sex, species


# ---------------------------
# Model: ConvNeXt multitask
# ---------------------------

@dataclass
class MultiTaskOutput:
    sex_logits: torch.Tensor
    species_logits: torch.Tensor


class ConvNeXtMultiTask(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        num_sex: int = 2,
        num_species: int = 5,
        dropout_p: float = 0.2,
    ):
        super().__init__()

        if backbone == "convnext_tiny":
            net = torchvision.models.convnext_tiny(
                weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            )
        elif backbone == "convnext_small":
            net = torchvision.models.convnext_small(
                weights=torchvision.models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            )
        elif backbone == "convnext_base":
            net = torchvision.models.convnext_base(
                weights=torchvision.models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ConvNeXt structure: features -> avgpool -> classifier( LayerNorm + Linear )
        self.features = net.features

        # Get feature dim from the original classifier's last Linear
        feat_dim = net.classifier[-1].in_features

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.sex_head = nn.Linear(feat_dim, num_sex)
        self.species_head = nn.Linear(feat_dim, num_species)

    def forward(self, x: torch.Tensor) -> MultiTaskOutput:
        x = self.features(x)              # [B, C, H, W]
        x = x.mean(dim=(2, 3))            # global average pool -> [B, C]
        x = self.dropout(x)
        return MultiTaskOutput(
            sex_logits=self.sex_head(x),       # [B, 2]
            species_logits=self.species_head(x) # [B, num_species]
        )



# ---------------------------
# Train / Eval
# ---------------------------

def multitask_loss(out: MultiTaskOutput, sex: torch.Tensor, species: torch.Tensor,
                   w_sex: float = 1.0, w_species: float = 1.0,
                   label_smoothing: float = 0.0) -> torch.Tensor:
    l1 = F.cross_entropy(out.sex_logits, sex, label_smoothing=label_smoothing)
    l2 = F.cross_entropy(out.species_logits, species, label_smoothing=label_smoothing)
    return w_sex * l1 + w_species * l2


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n = 0
    sex_correct = species_correct = both_correct = 0

    for x, sex, sp in loader:
        x = x.to(device, non_blocking=True)
        sex = sex.to(device, non_blocking=True)
        sp = sp.to(device, non_blocking=True)

        out = model(x)
        loss = multitask_loss(out, sex, sp)

        bs = x.size(0)
        n += bs
        loss_sum += float(loss.detach().cpu()) * bs

        ps = out.sex_logits.argmax(1)
        pp = out.species_logits.argmax(1)
        sex_correct += (ps == sex).sum().item()
        species_correct += (pp == sp).sum().item()
        both_correct += ((ps == sex) & (pp == sp)).sum().item()

    return {
        "loss": loss_sum / max(n, 1),
        "sex_acc": sex_correct / max(n, 1),
        "species_acc": species_correct / max(n, 1),
        "both_acc": both_correct / max(n, 1),
    }


def save_ckpt(path: Path, epoch: int, model: nn.Module,
              optimizer: torch.optim.Optimizer, scheduler, enc: LabelEncoders, extra: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "encoders": {"sex_to_id": enc.sex_to_id, "species_to_id": enc.species_to_id},
        "extra": extra,
    }, path)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    out_dir: str | Path,
    enc: LabelEncoders,
    use_amp: bool = True,
    label_smoothing: float = 0.05,
):
    out_dir = Path(out_dir)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n = 0
        sex_correct = species_correct = both_correct = 0

        for step, (x, sex, sp) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            sex = sex.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(x)
                loss = multitask_loss(out, sex, sp, label_smoothing=label_smoothing)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            n += bs
            loss_sum += float(loss.detach().cpu()) * bs

            ps = out.sex_logits.argmax(1)
            pp = out.species_logits.argmax(1)
            sex_correct += (ps == sex).sum().item()
            species_correct += (pp == sp).sum().item()
            both_correct += ((ps == sex) & (pp == sp)).sum().item()

            if step == 1 or step % 20 == 0 or step == len(train_loader):
                lr = optimizer.param_groups[0]["lr"]
                it_s = step / max(time.time() - t0, 1e-9)
                print(
                    f"\rEp {epoch:03d}/{epochs:03d} "
                    f"[{step:04d}/{len(train_loader):04d}] "
                    f"lr={lr:.2e} "
                    f"loss={loss_sum/max(n,1):.4f} "
                    f"acc(sex={sex_correct/max(n,1):.3f}, sp={species_correct/max(n,1):.3f}, both={both_correct/max(n,1):.3f}) "
                    f"{it_s:.1f} it/s",
                    end="",
                    flush=True,
                )

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        train_metrics = {
            "loss": loss_sum / max(n, 1),
            "sex_acc": sex_correct / max(n, 1),
            "species_acc": species_correct / max(n, 1),
            "both_acc": both_correct / max(n, 1),
        }
        val_metrics = eval_and_save_predictions_csv(
               model, val_loader, device, enc, Path(out_dir) / "val_predictions.csv"
        )

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["loss"])

        print(
            f"\nEpoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} both={train_metrics['both_acc']:.3f} | "
            f"val: {val_metrics}"
        )

        # save last (overwrite)
        save_ckpt(out_dir / "last.pt", epoch, model, optimizer, scheduler, enc, {"best_val_loss": best_val})

        # save best (overwrite only when improved)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_ckpt(out_dir / "best.pt", epoch, model, optimizer, scheduler, enc, {"best_val_loss": best_val})
            print(f"[best] val loss improved -> {best_val:.4f}")

@torch.no_grad()
def eval_and_save_predictions_csv(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    enc: LabelEncoders,
    out_csv: str | Path,
) -> Dict[str, float]:
    """
    Writes a CSV with:
      filename, sex_GT, species_GT, sex_Pred, species_Pred

    Also returns summary metrics.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # id -> string
    id_to_sex = {v: k for k, v in enc.sex_to_id.items()}
    id_to_species = {v: k for k, v in enc.species_to_id.items()}

    model.eval()

    rows = []
    loss_sum = 0.0
    n = 0
    sex_correct = species_correct = both_correct = 0

    # We rely on dataset row order matching the loader iteration order (shuffle=False)
    ds = loader.dataset
    if not hasattr(ds, "df"):
        raise ValueError("Dataset must have a .df with IMAGE_FILENAME/species/sex columns to write GT strings.")

    for batch_idx, (x, sex, sp) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        sex = sex.to(device, non_blocking=True)
        sp = sp.to(device, non_blocking=True)

        out = model(x)
        loss = multitask_loss(out, sex, sp)

        bs = x.size(0)
        start = batch_idx * loader.batch_size
        stop = start + bs

        ps = out.sex_logits.argmax(1).detach().cpu().numpy().tolist()
        pp = out.species_logits.argmax(1).detach().cpu().numpy().tolist()

        # read filenames + GT strings from the dataset df
        batch_df = ds.df.iloc[start:stop]

        for i in range(bs):
            rows.append({
                "filename": str(batch_df.iloc[i]["IMAGE_FILENAME"]),
                "sex_GT": str(batch_df.iloc[i]["sex"]),
                "species_GT": str(batch_df.iloc[i]["species"]),
                "sex_Pred": id_to_sex[int(ps[i])],
                "species_Pred": id_to_species[int(pp[i])],
            })

        # metrics
        loss_sum += float(loss.detach().cpu()) * bs
        n += bs

        sex_cpu = sex.detach().cpu().numpy()
        sp_cpu = sp.detach().cpu().numpy()
        sex_correct += int((sex_cpu == np.array(ps)).sum())
        species_correct += int((sp_cpu == np.array(pp)).sum())
        both_correct += int(((sex_cpu == np.array(ps)) & (sp_cpu == np.array(pp))).sum())

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[csv] wrote: {out_csv} ({len(rows)} rows)")

    return {
        "loss": loss_sum / max(n, 1),
        "sex_acc": sex_correct / max(n, 1),
        "species_acc": species_correct / max(n, 1),
        "both_acc": both_correct / max(n, 1),
    }

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    csv_path = "droso_classif.csv"
    root_dir = None
    out_dir = "checkpoints_droso_convnext"
    epochs = 200
    image_size = 512

    backbone = "convnext_tiny"   # try tiny first; small/base may overfit faster
    train_bs = 8                 # 512x512 is heavy; adjust for your GPU
    val_bs = 16
    num_workers = 4

    # ConvNeXt pretrained weights mean/std
    if backbone == "convnext_tiny":
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
    elif backbone == "convnext_small":
        weights = torchvision.models.ConvNeXt_Small_Weights.DEFAULT
    elif backbone == "convnext_base":
        weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
    else:
        raise ValueError(backbone)

    mean = weights.transforms().mean
    std = weights.transforms().std
    normalize = T.Normalize(mean=mean, std=std)

    train_tf = T.Compose([
        T.Lambda(pad_to_square_white),
        T.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=30, fill=(255, 255, 255)),   # consider not using full 360 unless needed
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.10),
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

    train_ds = FlyCsvDataset(csv_path, "train", train_tf, root_dir=root_dir)
    enc = train_ds.encoders
    val_ds = FlyCsvDataset(csv_path, "val", val_tf, root_dir=root_dir, encoders=enc)
    test_ds = FlyCsvDataset(csv_path, "test", val_tf, root_dir=root_dir, encoders=enc)

    print("sex mapping:", enc.sex_to_id)
    print("species mapping:", enc.species_to_id)
    print("sizes:", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=val_bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtMultiTask(
        backbone=backbone,
        pretrained=True,
        num_sex=len(enc.sex_to_id),
        num_species=len(enc.species_to_id),
        dropout_p=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-2)
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
        enc=enc,
        use_amp=True,
        label_smoothing=0.05,
    )

    test_metrics = eval_and_save_predictions_csv(
            model, test_loader, device, enc, Path(out_dir) / "test_predictions.csv"
        )
    print("\nTEST:", test_metrics)
