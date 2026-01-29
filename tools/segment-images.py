#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _bbox_to_int_crop(bbox, width, height):
    """
    COCO bbox is [x, y, w, h] in pixels (often floats).
    Convert to an (left, upper, right, lower) crop box, clipped to image bounds.
    """
    x, y, w, h = bbox
    left = int(np.floor(x))
    top = int(np.floor(y))
    right = int(np.ceil(x + w))
    bottom = int(np.ceil(y + h))

    left = max(0, min(left, width))
    top = max(0, min(top, height))
    right = max(0, min(right, width))
    bottom = max(0, min(bottom, height))

    if right <= left or bottom <= top:
        raise ValueError(f"Invalid bbox after clipping: {bbox} -> {(left, top, right, bottom)}")

    return left, top, right, bottom


def _mask_from_polygon(segmentation, img_w, img_h):
    """
    segmentation is a list of polygons (each polygon is a list of x,y coords).
    Returns a PIL 'L' mask (0..255) size (img_w, img_h).
    """
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    # COCO polygon format: [ [x1,y1,x2,y2,...], [x1,y1,...], ... ]
    for poly in segmentation:
        if not poly:
            continue
        if len(poly) < 6 or len(poly) % 2 != 0:
            raise ValueError(f"Malformed polygon segmentation with {len(poly)} coords: {poly[:10]}...")
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(pts, fill=255)

    return mask


def _mask_from_rle(segmentation, img_w, img_h):
    """
    segmentation in RLE can be:
      - dict with 'counts' and 'size'
      - list of such dicts
    We'll try to decode with pycocotools if available.
    """
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception as e:
        raise ImportError(
            "RLE segmentation found but pycocotools is not installed.\n"
            "Install it with: pip install pycocotools\n"
            f"Original import error: {e}"
        )

    rle = segmentation
    if isinstance(segmentation, list):
        # Merge multiple RLEs by OR-ing them
        decoded = None
        for r in segmentation:
            m = mask_utils.decode(r)  # HxWx1 or HxW
            if m.ndim == 3:
                m = m[:, :, 0]
            decoded = m if decoded is None else np.logical_or(decoded, m)
        arr = (decoded.astype(np.uint8) * 255)
    else:
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m[:, :, 0]
        arr = (m.astype(np.uint8) * 255)

    if arr.shape[0] != img_h or arr.shape[1] != img_w:
        raise ValueError(f"Decoded RLE mask shape {arr.shape} != image shape {(img_h, img_w)}")

    return Image.fromarray(arr, mode="L")


def _build_instance_mask(ann, img_w, img_h, crop_box):
    """
    Builds an alpha mask for the instance, then crops it to crop_box.
    If segmentation is missing, uses a filled rectangle equal to bbox.
    """
    seg = ann.get("segmentation", None)

    if seg is None or seg == []:
        # Fallback: bbox rectangle as mask
        full = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(full)
        left, top, right, bottom = _bbox_to_int_crop(ann["bbox"], img_w, img_h)
        draw.rectangle([left, top, right, bottom], fill=255)
        return full.crop(crop_box)

    # Polygon format (most common): list of lists of floats
    if isinstance(seg, list):
        full = _mask_from_polygon(seg, img_w, img_h)
        return full.crop(crop_box)

    # RLE format: dict with counts/size, or list of dicts
    if isinstance(seg, dict) or (isinstance(seg, list) and seg and isinstance(seg[0], dict)):
        full = _mask_from_rle(seg, img_w, img_h)
        return full.crop(crop_box)

    raise ValueError(f"Unknown segmentation format type: {type(seg)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Directory containing JPEG images referenced by COCO file_name")
    ap.add_argument("--coco_json", required=True, help="Path to COCO instances JSON")
    ap.add_argument("--output_dir", required=True, help="Directory to write PNG crops with transparency")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    coco_path = Path(args.coco_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    # Map image_id -> image info
    img_by_id = {}
    for im in images:
        if "id" not in im or "file_name" not in im:
            raise ValueError("Each image entry must have 'id' and 'file_name'")
        img_by_id[im["id"]] = im

    # Group annotations by image_id
    anns_by_img = defaultdict(list)
    for a in anns:
        if "image_id" not in a:
            raise ValueError("Each annotation must have 'image_id'")
        anns_by_img[a["image_id"]].append(a)

    # Process each image in COCO
    for image_id, im in img_by_id.items():
        file_name = im["file_name"]
        src_path = images_dir / file_name

        if not src_path.exists():
            raise FileNotFoundError(f"Image not found: {src_path}")

        image_anns = anns_by_img.get(image_id, [])
        if len(image_anns) != 1:
            raise RuntimeError(
                f"Expected exactly 1 instance for image_id={image_id} ({file_name}), "
                f"but found {len(image_anns)}"
            )

        ann = image_anns[0]
        if "bbox" not in ann:
            raise ValueError(f"Annotation missing bbox for image_id={image_id} ({file_name})")

        # Load image
        img = Image.open(src_path).convert("RGB")
        img_w, img_h = img.size

        # Crop by bbox
        crop_box = _bbox_to_int_crop(ann["bbox"], img_w, img_h)
        crop_rgb = img.crop(crop_box)

        # Build alpha mask (segmentation if possible, else bbox rectangle) then crop
        alpha = _build_instance_mask(ann, img_w, img_h, crop_box)

        # Combine into RGBA
        crop_rgba = crop_rgb.convert("RGBA")
        crop_rgba.putalpha(alpha)

        # Output name: same as parent, but .png
        base = Path(file_name).name
        stem = Path(base).stem  # handles .jpg/.jpeg/.JPG etc.
        out_path = out_dir / f"{stem}.png"

        crop_rgba.save(out_path, format="PNG")

    print(f"Done. Wrote PNGs to: {out_dir}")


if __name__ == "__main__":
    main()
