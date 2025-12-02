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
from torch.xpu import device

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False


def measure_background(im: Image.Image, path: Path):
    """Return the number of background pixels depending on format."""
    w, h = im.size
    arr = np.asarray(im)
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        # count pure white pixels
        if arr.ndim == 3 and arr.shape[2] >= 3:
            white_mask = np.all(arr[:, :, :3] == 255, axis=2)
            return int(w * h -white_mask.sum() ), "NON_TRANSPARENT_PIXELS"
    elif path.suffix.lower() == ".png":
        if im.mode != "RGBA":
            im = im.convert("RGBA")
            arr = np.asarray(im)
        alpha = arr[..., 3]
        trans_mask = alpha == 0
        return int(w * h - trans_mask.sum() ), "NON_TRANSPARENT_PIXELS"
    return w * h, "NON_TRANSPARENT_PIXELS"  # default


def resize_and_pad(im: Image.Image, size=224):
    """Resize image to keep aspect ratio, then pad to square with white background."""
    # keep proportions
    w, h = im.size
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    im_resized = im.resize((new_w, new_h), Image.LANCZOS)
    # create white square canvas
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(im_resized, offset)
    return canvas, scale


def main(args):
    csv_path = Path(args.csv)
    root_dir = Path(args.root)
    out_csv = Path(args.out)

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)
    if not root_dir.exists():
        print(f"ERROR: root dir not found: {root_dir}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(csv_path)
    if "IMAGE_FILENAME" not in df.columns or "DATASET" not in df.columns:
        print("ERROR: metadata csv must contain 'IMAGE_FILENAME' and 'DATASET' columns", file=sys.stderr)
        sys.exit(2)

    print(f"Loaded metadata CSV: {csv_path} (rows: {len(df)})")

    records = []
    iterator = df.itertuples(index=False)
    if TQDM:
        iterator = tqdm(list(iterator), desc="processing images")

    for row in iterator:
        dataset = getattr(row, "DATASET")
        orig_name = getattr(row, "IMAGE_FILENAME")
        img_name = Path(orig_name).name
        dataset_dir = root_dir / "01_segmented" / dataset

        # find jpg or png
        jpg_path = dataset_dir / img_name
        png_path = dataset_dir / (Path(img_name).stem + ".png")
        if not jpg_path.exists() and not png_path.exists():
            warnings.warn(f"Image not found for {img_name} in {dataset_dir}")
            continue

        path = jpg_path if jpg_path.exists() else png_path
        try:
            im = Image.open(path)
        except Exception as e:
            warnings.warn(f"Failed to open {path}: {e}")
            continue

        # measure background
        count, colname = measure_background(im, path)

        # if PNG, replace transparent background with white before saving
        if path.suffix.lower() == ".png":
            if im.mode != "RGBA":
                im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg.convert("RGB")

        # resize and pad
        im_resized, scale = resize_and_pad(im, 224)

        # output dir
        out_dir = root_dir / "02_resized" / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        new_name = Path(img_name).stem + ".jpg"
        out_path = out_dir / new_name

        im_resized.save(out_path, format="JPEG", quality=97, subsampling=0, optimize=True)

        record = row._asdict()
        record["ORIGINAL_IMAGE_FILENAME"] = str(path)
        record["IMAGE_FILENAME"] = str(out_path)
        record["SCALE"] = scale
        record["ROI_SIZE_MM"]  = (224/scale) / record["DPI"] * 25.4
        record["BF_cbrMG_MM"] =  record["DRYMASS_MG"] ** (1/3) / record["ROI_SIZE_MM"]

        record[colname] = count

        record["AREA_MM2"] = record["NON_TRANSPARENT_PIXELS"] * (25.4/ record["DPI"]) ** 2

        records.append(record)

    out_df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Wrote updated metadata CSV: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and measure background area in JPG/PNG images per dataset.")
    parser.add_argument("--csv", default="metadata.csv" , help="Path to metadata CSV (must contain IMAGE_FILENAME and dataset columns)")
    parser.add_argument("--root", default="00_data",  help="Root directory containing 01_segmented/ and 02_resized/")
    parser.add_argument("--out", default="metadata_enriched.csv", help="Output CSV path")
    args = parser.parse_args()
    main(args)
