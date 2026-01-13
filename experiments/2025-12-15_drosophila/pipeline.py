from __future__ import annotations

import json
import logging
import os.path
import re
import shutil
# import easyocr
from pathlib import Path
from PIL import Image
from pathlib import Path
from pylibdmtx.pylibdmtx import decode as dmtx_decode

logger = logging.getLogger()
logger.setLevel(logging.INFO)



Image.MAX_IMAGE_PIXELS = None

def set_png_dpi(path: Path, dpi: tuple[float, float]):
    with Image.open(path) as im:
        tmp = path.with_suffix(".tmp.png")
        im.save(tmp, dpi=dpi)
    tmp.replace(path)

def get_dpi(path, default=(72, 72)):
    with Image.open(path) as im:
        im.draft("RGB", (512, 512))
        dpi = im.info.get("dpi", default)
        return dpi


def parse_barcode( barcode_path: Path) -> Dict[str, Any]:

    """
    Read a Data Matrix from 'barcode_path' and return a dict payload.
    Rules:
      1) Must decode exactly 1 symbol -> else raise ValueError
      2) Decoded payload must parse to a mapping (dictionary) -> else raise ValueError
    """
    # 1) Load image
    enforce_barcode = True

    try:
        try:
            img = Image.open(barcode_path)
        except Exception as e:
            raise ValueError(f"Failed to open barcode image '{barcode_path}': {e}")

        # Optional: convert to grayscale to help decoder on noisy images
        try:
            img = img.convert("L")
        except Exception:
            # If convert fails for some reason, proceed with original
            pass

        # 2) Decode Data Matrix
        try:
            results = dmtx_decode(img)
        except Exception as e:
            raise ValueError(f"pylibdmtx failed to decode '{barcode_path}': {e}")

        if not results or len(results) != 1:
            found = 0 if not results else len(results)
            raise ValueError(f"Expected exactly 1 Data Matrix, found {found} in '{barcode_path}'.")

        raw_bytes = results[0].data
        try:
            payload_text = raw_bytes.decode("utf-8")
        except Exception as e:
            raise ValueError(f"Decoded Data Matrix is not valid UTF-8: {e}")

        # 3) Parse into a dictionary (prefer JSON; fallback to YAML if you want)
        parsed: Any = None
        json_err = yaml_err = None

        try:
            parsed = json.loads(payload_text)
        except Exception as e:
            json_err = e
        if not isinstance(parsed, dict):
            details = []
            if json_err: details.append(f"json: {json_err}")
            if yaml_err: details.append(f"yaml: {yaml_err}")
            detail_txt = "; ".join(details) if details else "payload is not a mapping"
            parsed = {"sample_id": payload_text}
            logging.warning(f"Decoded Data Matrix did not yield a dictionary. Returning instead {parsed}. {detail_txt}, {payload_text}")


        # 4) Return the dictionary
        return payload_text
    except Exception as e:
        if enforce_barcode:
            raise e
        else:
            logging.warning(str(e))
            return {}



from flat_bug.predictor import Predictor

if __name__ == "__main__":

    FB_WEIGHTS = "/home/quentin/repos/flat-bug-git/scripts/training/runs/segment/fb_M_2025-12-01_08-09-45/weights/last.pt"
    IN_DIR_ROOT = "/run/media/quentin/STICK/droso/entoscan"
    OUT_DIR_ROOT = "/home/quentin/Desktop/droso/results"
    IMAGE_SCALE = 0.5
    FORCE =True

    # reader = easyocr.Reader(["en"], gpu=True)
    pred = Predictor(model=FB_WEIGHTS, device="cuda:0")

    import json
    import glob
    import shutil

    for f in glob.glob(str(Path(IN_DIR_ROOT) /"**"/ "*.json"), recursive=True):

        metadata_f = Path(f)
        with open(metadata_f, "r") as f:
            metadata = json.load(f)

        image_dir = metadata_f.parent
        thumbnail = image_dir / metadata["thumbnail"]
        image = image_dir /  metadata["filename"]

        assert thumbnail.exists()
        assert image.exists()
        out_full_root = Path(OUT_DIR_ROOT) / Path(IN_DIR_ROOT).name

        out_rel_path = Path(os.path.dirname(os.path.relpath(metadata_f, IN_DIR_ROOT)))

        target = out_full_root / out_rel_path

        if not FORCE and (target/metadata_f.name).exists():
            logging.info(f"Skipping existing target {target/metadata_f.name}")
            continue
        try:
            label = parse_barcode( thumbnail)

            logging.info(f"Found {label}")
            metadata["custom_label"] = label
            if target.exists():
                shutil.rmtree(target)
            results = pred(str(image),scale_before=IMAGE_SCALE,)

            # results.boxes
            # for i in range(len(results)):


            os.makedirs(target/"crops", exist_ok=True)
            crops = results.save_crops(target / "crops", mask=True, identifier=label)
            results.plot(outpath=str(target/"flatbug_results.jpg"), scale=IMAGE_SCALE)

            dpi = get_dpi(str(image))
            for crp,b,cnf,s in zip(crops, results.boxes, results.confs, results.scales):
                m = metadata
                set_png_dpi(Path(crp), dpi)
                m["crop"] = Path(crp).name
                m["box"] = b.tolist()
                m["conf"] = float(cnf)
                m["scale"] = float(s)

                with open(Path(crp).with_suffix(".json"), "w") as f:
                    json.dump(m, f)

            with open(target/metadata_f.name, "w") as f:
                json.dump(metadata, f)


        except ValueError as e:
            logging.error(e)
            pass
