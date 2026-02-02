#!/usr/bin/env python3
"""
preprocess.py

For each image listed in the metadata CSV, look up its dataset, find the corresponding
image (jpg or png) in <root_dir>/01_segmented/<dataset>/, measure background area, and
export a resized 224px (max side) padded white JPG in <root_dir>/02_resized/<dataset>/.

Adds columns:
- ORIGINAL_IMAGE_FILENAME
- IMAGE_FILENAME (new file name in 02_resized)
- WHITE_PIXELS (for JPGs)
- TRANSPARENT_PIXELS (for PNGs)

All other columns are preserved and saved in a new CSV.
"""

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed
import multiprocessing

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

def measure_background(im: Image.Image, path: Path, white_thresh=250):
    """
    Return the number of foreground (non-background) pixels.

    For JPG/JPEG:
      - background is defined as near-white pixels (>= white_thresh in all RGB channels)

    For PNG:
      - background is defined as fully transparent pixels (alpha == 0)

    Returns
    -------
    count : int
        Number of non-background pixels
    colname : str
        Column name ("FOREGROUND_PIXELS")
    """
    w, h = im.size

    if path.suffix.lower() in [".jpg", ".jpeg"]:
        arr = np.asarray(im)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            # near-white background (robust to JPEG + resampling)
            bg_mask = np.all(arr[:, :, :3] >= white_thresh, axis=2)
            fg_pixels = (~bg_mask).sum()
            return int(fg_pixels), "FOREGROUND_PIXELS"

    elif path.suffix.lower() == ".png":
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        arr = np.asarray(im)
        alpha = arr[..., 3]
        bg_mask = alpha == 0
        fg_pixels = (~bg_mask).sum()
        return int(fg_pixels), "FOREGROUND_PIXELS"

    # fallback: assume everything is foreground
    return w * h, "FOREGROUND_PIXELS"


def resize_and_pad(im: Image.Image, output_size: int = 224):
    """
    Resize an image so that its longest side equals `output_size`,
    preserving aspect ratio, then center-pad to a square white canvas.

    Returns
    -------
    canvas : PIL.Image.Image
        Square RGB image of shape (output_size, output_size)
    scale : float
        Resize factor applied to the original image
        (new_size = original_size * scale)
    """
    orig_w, orig_h = im.size

    # scale so that the longest side matches output_size
    scale = output_size / max(orig_w, orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = im.resize((new_w, new_h), resample=Image.LANCZOS)

    # white square canvas
    canvas = Image.new("RGB", (output_size, output_size), color=(255, 255, 255))

    # center placement
    x0 = (output_size - new_w) // 2
    y0 = (output_size - new_h) // 2
    canvas.paste(resized, (x0, y0))

    return canvas, scale

def process_one_row(row, root_dir: Path):
    """
    row: pandas namedtuple from itertuples(index=False)
    returns: dict record or None (if skipped)
    """
    record = row._asdict()
    dataset = record["DATASET"]
    orig_name = record["IMAGE_FILENAME"]

    img_name = Path(orig_name).name
    in_dir = root_dir / "01_segmented" / str(dataset)

    candidates = [
        in_dir / img_name,
        in_dir / (Path(img_name).stem + ".jpg"),
        in_dir / (Path(img_name).stem + ".jpeg"),
        in_dir / (Path(img_name).stem + ".png"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None  # skip (you can return record with a flag if you prefer)

    try:
        im = Image.open(path)
    except Exception:
        return None

    # foreground/background measurement on original
    fg_pixels, fg_col = measure_background(im, path)
    record[fg_col] = fg_pixels

    # composite PNG on white
    if path.suffix.lower() == ".png":
        im_rgba = im.convert("RGBA")
        white_bg = Image.new("RGBA", im_rgba.size, (255, 255, 255, 255))
        white_bg.paste(im_rgba, mask=im_rgba.split()[-1])
        im = white_bg.convert("RGB")
    else:
        im = im.convert("RGB")

    # resize + pad
    im_resized, scale = resize_and_pad(im, output_size=224)

    # write output image
    out_dir = root_dir / "02_resized" / str(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(img_name).stem + ".jpg")
    im_resized.save(out_path, format="JPEG", quality=97, subsampling=0, optimize=True)

    # derived fields
    dpi = float(record["DPI"])
    drymass_mg = float(record["DRYMASS_MG"])
    if dpi <= 0:
        roi_size_mm = np.nan
        bf = np.nan
        area_mm2 = np.nan
    else:
        roi_size_mm = (224 / scale) / dpi * 25.4
        bf = (drymass_mg ** (1 / 3)) / roi_size_mm if drymass_mg > 0 else np.nan
        area_mm2 = fg_pixels * (25.4 / dpi) ** 2

    record.update(
        {
            "ORIGINAL_IMAGE_FILENAME": str(path),
            "IMAGE_FILENAME": str(out_path),
            "SCALE": float(scale),
            "ROI_SIZE_MM": float(roi_size_mm) if np.isfinite(roi_size_mm) else np.nan,
            "BF_cbrMG_MM": float(bf) if np.isfinite(bf) else np.nan,
            "AREA_MM2": float(area_mm2) if np.isfinite(area_mm2) else np.nan,
        }
    )

    return record

def main(args):
    csv_path = Path(args.csv)
    root_dir = Path(args.root)
    out_csv = Path(args.out)

    # validate...
    df = pd.read_csv(csv_path)

    required_cols = {"IMAGE_FILENAME", "DATASET", "DPI", "DRYMASS_MG"}
    missing = required_cols - set(df.columns)


    if missing:
        print(f"ERROR: metadata csv is missing required columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(2)


    print(f"Loaded metadata CSV: {csv_path} (rows: {len(df)})")

    if "CONF" in df.columns:
        before = len(df)
        df = df[df["CONF"] > CONF_THRESHOLD].reset_index(drop=True)
        print(f"Filtered by CONF > {CONF_THRESHOLD}: {before} → {len(df)} rows")
    else:
        print("No CONF column found — skipping confidence filtering")
    iterator = list(df.itertuples(index=False))  # joblib needs a materialized iterable

    n_jobs = getattr(args, "n_jobs", None) or DEFAULT_N_JOBS

    records = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_one_row)(row, root_dir) for row in iterator
    )

    # drop skipped
    records = [r for r in records if r is not None]

    out_df = pd.DataFrame.from_records(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Wrote updated metadata CSV: {out_csv}")


if __name__ == "__main__":
    CONF_THRESHOLD=0.9
    DEFAULT_N_JOBS=8
    parser = argparse.ArgumentParser(description="Resize and measure background area in JPG/PNG images per dataset.")
    parser.add_argument("--csv", default="metadata.csv" , help="Path to metadata CSV (must contain IMAGE_FILENAME and dataset columns)")
    parser.add_argument("--root", default="00_data",  help="Root directory containing 01_segmented/ and 02_resized/")
    parser.add_argument("--out", default="metadata_enriched.csv", help="Output CSV path")
    args = parser.parse_args()
    main(args)



