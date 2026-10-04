"""How additive is a model? Score it on two-insect composites of held-out weighed specimens.

For each composite C of A and B (built by chimera.py, seeded, so every model sees the same set):
  accuracy:   log M_hat(C) - log(M_A + M_B)                 (true masses)
  additivity: log M_hat(C) - log(M_hat(A) + M_hat(B))       (the model's own part predictions;
                                                             needs no weights, so it also applies
                                                             to unweighed images such as BIOSCAN)
Parts are scored on their normal 224 px images, as in training.

Usage (from 01_biomass_model/):
  python ../tools/chimera-eval.py <checkpoint.pt> <logbf|logmass> <out_dir> [n_composites] [split]
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path.cwd()))
from chimera import load_crop, make_chimera, PartnerSampler
from models import load_gaussian_model, call_gaussian_model

Z95 = 1.959964


def log_mass(out, log_l, target):
    mu = out[:, 0]
    return 3 * (mu + log_l) if target == "logbf" else mu


def main():
    ckpt, target, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    split = sys.argv[5] if len(sys.argv) > 5 else "test"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_gaussian_model(ckpt, device)
    tt = T.ToTensor()

    d = pd.read_csv("metadata_enriched.csv")
    d = d[(d.IS_VALID == True) & (d.SPLIT == split)].reset_index(drop=True)
    sampler = PartnerSampler(d.DRYMASS_MG.to_numpy(float), d.INSECT_ID.to_numpy(), d.DATASET.to_numpy())
    rng = random.Random(0)

    rows, imgs, scales = [], [], []
    for k in range(n):
        i = rng.randrange(len(d)); j = sampler(i, rng)
        a, b = d.iloc[i], d.iloc[j]
        img, L_c, layout = make_chimera(load_crop(a), load_crop(b), rng)
        imgs.append(tt(img)); scales.append([np.log(L_c), np.log(a.AREA_MM2 + b.AREA_MM2)])
        rows.append({"i": i, "j": j, "layout": layout, "L_c": L_c, "M_true": a.DRYMASS_MG + b.DRYMASS_MG})
    comp = pd.DataFrame(rows)

    def predict(x, s):
        outs = []
        with torch.no_grad():
            for k in range(0, len(x), 32):
                outs.append(call_gaussian_model(model, x[k:k + 32].to(device), s[k:k + 32].to(device)).cpu().numpy())
        return np.concatenate(outs)

    # parts: every specimen image used in a composite, on its normal 224 px input
    used = sorted(set(comp.i) | set(comp.j))
    px = torch.stack([tt(Image.open(d.IMAGE_FILENAME[u]).convert("RGB")) for u in used])
    ps = torch.tensor(np.log(d.loc[used, ["ROI_SIZE_MM", "AREA_MM2"]].to_numpy(float)), dtype=torch.float32)
    part_lm = dict(zip(used, log_mass(predict(px, ps), ps[:, 0].numpy(), target)))

    oc = predict(torch.stack(imgs), torch.tensor(scales, dtype=torch.float32))
    comp["lm_hat"] = log_mass(oc, comp.L_c.map(np.log).to_numpy(), target)
    comp["sd_hat"] = (3.0 if target == "logbf" else 1.0) * np.exp(oc[:, 1])   # raw, uncalibrated
    comp["lm_parts"] = np.log(np.exp(comp.i.map(part_lm)) + np.exp(comp.j.map(part_lm)))
    comp["err_true"] = comp.lm_hat - np.log(comp.M_true)
    comp["err_parts_true"] = comp.lm_parts - np.log(comp.M_true)
    comp["additivity_gap"] = comp.lm_hat - comp.lm_parts
    comp.to_csv(out / "chimera_eval.csv", index=False)

    pct = lambda e: np.abs(np.expm1(e)) * 100
    print(f"{n} composites of {split} specimens, model {ckpt}")
    print(f"  composite vs true sum:        median % error {np.median(pct(comp.err_true)):5.1f}   "
          f"bias {np.expm1(comp.err_true.median()) * 100:+.1f}%")
    print(f"  sum of parts vs true sum:     median % error {np.median(pct(comp.err_parts_true)):5.1f}   "
          f"bias {np.expm1(comp.err_parts_true.median()) * 100:+.1f}%")
    print(f"  additivity gap (C vs parts):  median |gap| {np.median(pct(comp.additivity_gap)):5.1f}%   "
          f"bias {np.expm1(comp.additivity_gap.median()) * 100:+.1f}%")
    for lay, g in comp.groupby("layout"):
        print(f"    {lay:8s} n={len(g):4d}  median % error vs truth {np.median(pct(g.err_true)):5.1f}  "
              f"additivity bias {np.expm1(g.additivity_gap.median()) * 100:+.1f}%")


if __name__ == "__main__":
    main()
