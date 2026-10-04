"""Two-insect composites ("chimeras") at their true relative scale.

Mass adds, BF does not: for insects A and B with masses M_A, M_B placed together in a crop of
longest side L_C (mm), the composite's target is BF_C = (M_A + M_B)^(1/3) / L_C, so its BF depends
on the layout through L_C while its mass does not.

Composites are built from the full-resolution segmented RGBA crops (00_data/01_segmented), not from
the 224 px training images: those are each resized differently, so pasting them would put the
insects at the wrong relative sizes. Both crops are brought to a common pixel size (the coarser of
the two), rotated/flipped independently, laid out end to end, side by side or diagonally with a
random gap, cropped like the pipeline's crops, and passed through the same resize-to-224, white
padding and JPEG(97) steps as 01-preprocess.py.
"""
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_SIZE = 224


def load_crop(row, root: Path = Path(".")):
    """(RGBA array, mm per pixel) for one specimen image row of metadata_enriched.csv."""
    rgba = np.asarray(Image.open(root / row["ORIGINAL_IMAGE_FILENAME"]).convert("RGBA"))
    return rgba, 25.4 / float(row["DPI"])


def _tight(rgba: np.ndarray) -> np.ndarray:
    """Crop to the mask's bounding box."""
    ys, xs = np.nonzero(rgba[..., 3] > 0)
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _rescale(rgba: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1) < 1e-3:
        return rgba
    h, w = rgba.shape[:2]
    im = Image.fromarray(rgba).resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.LANCZOS)
    a = np.asarray(im).copy()
    a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)      # keep a binary mask, like the pipeline's
    return a


def _random_orientation(rgba: np.ndarray, rng: random.Random) -> np.ndarray:
    rgba = np.rot90(rgba, rng.randint(0, 3))
    return rgba[:, ::-1] if rng.random() < 0.5 else rgba


def _resize_and_pad(im: Image.Image) -> Image.Image:
    """Same as 01-preprocess.py: longest side to 224 px, centred on white."""
    w, h = im.size
    scale = OUTPUT_SIZE / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    canvas = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE), (255, 255, 255))
    canvas.paste(im.resize((nw, nh), resample=Image.LANCZOS), ((OUTPUT_SIZE - nw) // 2, (OUTPUT_SIZE - nh) // 2))
    return canvas


def make_chimera(crop_a, crop_b, rng: random.Random):
    """Compose two (rgba, mm_per_px) crops. Returns (224 px RGB image, L_C in mm, layout name)."""
    (a, pa), (b, pb) = crop_a, crop_b
    pc = max(pa, pb)                                         # common pixel size: the coarser one
    a = _random_orientation(_tight(_rescale(a, pa / pc)), rng)
    b = _random_orientation(_tight(_rescale(b, pb / pc)), rng)
    if rng.random() < 0.5:
        a, b = b, a
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    gap = int(round(rng.uniform(0.0, 0.15) * max(ha, wa, hb, wb)))
    layout = rng.choice(["row", "column", "diagonal"])
    if layout == "row":
        ya, yb = (rng.randint(0, max(0, hb - ha)), 0) if ha < hb else (0, rng.randint(0, max(0, ha - hb)))
        xa, xb = 0, wa + gap
    elif layout == "column":
        xa, xb = (rng.randint(0, max(0, wb - wa)), 0) if wa < wb else (0, rng.randint(0, max(0, wa - wb)))
        ya, yb = 0, ha + gap
    else:
        xa, ya, xb, yb = 0, 0, wa + gap, ha + gap
    H, W = max(ya + ha, yb + hb), max(xa + wa, xb + wb)
    canvas = np.full((H, W, 3), 255, np.uint8)
    for arr, x, y in ((a, xa, ya), (b, xb, yb)):
        m = arr[..., 3] > 0
        canvas[y:y + arr.shape[0], x:x + arr.shape[1]][m] = arr[..., :3][m]
    # the pipeline's crops keep a small margin around the detection
    margin = int(round(rng.uniform(0.0, 0.03) * max(H, W)))
    if margin:
        canvas = np.pad(canvas, ((margin, margin), (margin, margin), (0, 0)), constant_values=255)
    L_c = max(canvas.shape[:2]) * pc
    buf = io.BytesIO()
    _resize_and_pad(Image.fromarray(canvas)).save(buf, format="JPEG", quality=97, subsampling=0)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), L_c, layout


class PartnerSampler:
    """Draws a partner of similar mass (within a factor `max_ratio`), so neither insect dominates,
    from the same stratum (scanner), so a composite never mixes two imaging setups."""

    def __init__(self, masses: np.ndarray, groups=None, strata=None, max_ratio: float = 3.0):
        self.log_m = np.log(np.asarray(masses, dtype=float))
        # specimen id per row, so a partner is never another image of the same insect
        self.groups = np.asarray(groups) if groups is not None else np.arange(len(self.log_m))
        self.strata = np.asarray(strata) if strata is not None else np.zeros(len(self.log_m))
        self.order = np.argsort(self.log_m)
        self.sorted = self.log_m[self.order]
        self.half = np.log(max_ratio)

    def __call__(self, i: int, rng: random.Random) -> int:
        lo = np.searchsorted(self.sorted, self.log_m[i] - self.half, "left")
        hi = np.searchsorted(self.sorted, self.log_m[i] + self.half, "right")
        for _ in range(50):
            j = int(self.order[rng.randrange(lo, hi)])
            if self.groups[j] != self.groups[i] and self.strata[j] == self.strata[i]:
                return j
        # rare: no match in the mass window; take the nearest mass in the same stratum
        same = np.nonzero((self.strata == self.strata[i]) & (self.groups != self.groups[i]))[0]
        return int(same[np.argmin(np.abs(self.log_m[same] - self.log_m[i]))])
