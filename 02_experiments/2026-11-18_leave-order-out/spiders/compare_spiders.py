"""Evaluate saved BF and log-biomass CNNs on the same held-out spiders."""
import argparse
import csv
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path


def read_metadata(experiment):
    path = experiment / "metadata_leave_out_Araneae.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    spiders = [r for r in rows if r["order"].strip().casefold() == "araneae"]
    if not spiders:
        raise ValueError("No Araneae found in metadata.")
    keys = {(r["DATASET"], r["INSECT_ID"]) for r in spiders}
    if any(r["SPLIT"] != "test" for r in rows
           if (r["DATASET"], r["INSECT_ID"]) in keys):
        raise ValueError("Spider specimens occur outside test: possible leakage.")
    selected = [r for r in spiders if r["SPLIT"] == "test"
                and r["IS_VALID"].strip().lower() == "true"]
    if not selected:
        raise ValueError("No valid spider test images.")
    for r in selected:
        if any(not math.isfinite(float(r[c])) or float(r[c]) <= 0
               for c in ("DRYMASS_MG", "ROI_SIZE_MM")):
            raise ValueError(f"Invalid mass or physical scale: {r['IMAGE_FILENAME']}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path(
        "C:/Users/au738360/PycharmProjects/BEEomass/02_experiments/"
        "2026-11-18_leave-order-out/spiders"))
    parser.add_argument("--image-root", type=Path, default=Path(
        "I:/BEE_qg_2026-11-05/01_biomass_model/00_data/02_resized"))
    parser.add_argument("--check-only", action="store_true",
                        help="Check metadata and checkpoints without loading images or models")
    args = parser.parse_args()
    experiment = args.experiment_dir.resolve()
    rows = read_metadata(experiment)
    checkpoints = {
        "BF_CNN": experiment / "BF_full_01" / "selected_model.pt",
        "log_biomass_CNN": experiment / "log_full_01" / "selected_model.pt",
    }
    for path in [experiment / "models.py", *checkpoints.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    n_specimens = len({(r["DATASET"], r["INSECT_ID"]) for r in rows})
    print(f"Valid spider test images: {len(rows)}; specimens: {n_specimens}")
    print("No spider specimens occur in training or validation.")
    if args.check_only:
        print("Metadata and checkpoint checks passed; images not checked.")
        return

    # Load dependencies only for inference. No data-loader workers are spawned.
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import ToTensor

    paths = []
    for row in rows:
        path = Path(row["IMAGE_FILENAME"])
        if not path.is_absolute():
            path = args.image_root / row["DATASET"] / path.name
        paths.append(path)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing images; first: {missing[0]}")

    # Import the experiment's exact architecture, avoiding unrelated 'models' packages.
    spec = importlib.util.spec_from_file_location("experiment_models", experiment / "models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    to_tensor = ToTensor()  # Matches validation/test preprocessing in 02_train.py.
    truth = np.array([float(r["DRYMASS_MG"]) for r in rows])
    scale = np.array([float(r["ROI_SIZE_MM"]) for r in rows])
    summaries = []

    for name, checkpoint in checkpoints.items():
        model = module.build_efficientnet(variant="v2_s", pretrained=False).to(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        del state
        model.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(paths), 32):
                batch = []
                for path in paths[start:start + 32]:
                    with Image.open(path) as image:
                        batch.append(to_tensor(image.convert("RGB")))
                outputs.extend(model(torch.stack(batch).to(device)).cpu().numpy().reshape(-1))
        raw = np.array(outputs, dtype=float)
        mass = (raw * scale) ** 3 if name == "BF_CNN" else np.exp(raw)
        if not np.isfinite(mass).all():
            raise ValueError(f"Nonfinite predictions from {name}")
        errors = mass - truth
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        ss_total = np.sum((truth - truth.mean()) ** 2)
        summaries.append({
            "model": name, "n_images": len(rows), "n_specimens": n_specimens,
            "R2_mass": float(1 - np.sum(errors ** 2) / ss_total) if ss_total > 0 else "",
            "MAE_mg": float(np.mean(np.abs(errors))), "RMSE_mg": rmse,
            "CV_RMSE_percent": 100 * rmse / float(truth.mean()),
            "bias_mg": float(errors.mean()),
            "negative_mass_predictions": int(np.sum(mass < 0)),
        })
        for row, prediction, native in zip(rows, mass, raw):
            row[name + "_mass_mg"] = float(prediction)
            row[name + "_native_output"] = float(native)
        del model
        print(f"Finished {name}")

    # A unique new directory preserves all training outputs and previous evaluations.
    output = experiment / ("spider_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    for filename, records in [("spider_predictions.csv", rows),
                              ("spider_metrics.csv", summaries)]:
        with (output / filename).open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with (output / "evaluation_details.json").open("x", encoding="utf-8") as handle:
        json.dump({"metadata": str(experiment / "metadata_leave_out_Araneae.csv"),
                   "checkpoints": {k: str(v) for k, v in checkpoints.items()},
                   "image_root": str(args.image_root), "device": str(device),
                   "metric_unit": "mg", "metric_weighting": "equal weight per image",
                   "CV_RMSE_definition": "100 * RMSE / mean measured mass",
                   "prediction_clipping": "none"}, handle, indent=2)
    print("\nSpider-only results (all errors calculated in mg):")
    for result in summaries:
        print(f"{result['model']}: R2={result['R2_mass']:.4f}, "
              f"MAE={result['MAE_mg']:.4f} mg, RMSE={result['RMSE_mg']:.4f} mg, "
              f"CV(RMSE)={result['CV_RMSE_percent']:.2f}%")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
