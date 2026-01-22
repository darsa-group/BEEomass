from __future__ import annotations
from segment_utils import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
Image.MAX_IMAGE_PIXELS = None



if __name__ == "__main__":

    FB_WEIGHTS = "../utils/fb_M_2025-12-01_08-09-45.pt"
    IN_DIR_ROOT = "00_data/00_raw"
    OUT_DIR_ROOT = "00_data/01_segmented/crops"
    IMAGE_SCALE = 0.5
    FORCE =True
    LABEL_PATTERN = r"^SPI_\d{3}_[A-Z]{2}_[A-Z]{3}_\d{4}-\d{2}-\d{2}]$"
    segment_dir(FB_WEIGHTS, IN_DIR_ROOT, OUT_DIR_ROOT, IMAGE_SCALE, FORCE, LABEL_PATTERN)
