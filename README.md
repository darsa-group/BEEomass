# BEEomass

Code for the [paper](https://ecoevorxiv.org/repository/view/13446/):

> **EntoScan and BEEomass: a standardized imaging system and a physically motivated model for high-throughput dry biomass estimation of arthropods**
> Melika Baghooee, Robert Thalheim, Fevziye Hasan, Søren Toft, Torsten Nygård Kristensen, and Quentin Geissmann

The training dataset (paired images and dry biomass measurements, ~2,100 specimens / ~8,400 images) is publicly available at [https://zenodo.org/records/20543262](https://zenodo.org/records/20543262).

---

## Overview

BEEomass (Biomass Estimation in Entomology) is a computer vision model for estimating the dry biomass of individual arthropods directly from images, without drying and weighing each specimen. It is designed to work with images acquired using [EntoScan](https://github.com/darsa-group/EntoScan), a modified flatbed scanner, but generalises to other calibrated imaging setups.

---

## The Biomass Factor model

Rather than predicting dry mass *M* directly — which would conflate genuine body composition with image scale — the model predicts a size-normalised **Biomass Factor (BF)**.

**Step 1 — physical scale.** Each segmented image is resized so its longest side equals 224 px (resize factor *s*). The physical width of the imaged field of view is then

$$L = 25.4 \times \frac{224/s}{\text{DPI}} \quad [\text{mm}]$$

where 25.4 converts inches to millimetres.

**Step 2 — Biomass Factor.** The model target is defined as

$$\text{BF} = \frac{M^{1/3}}{L} \quad [\text{mg}^{1/3}\,\text{mm}^{-1}]$$

This quantity is dimensionless up to units: under isometric scaling (*M* ∝ *L*³) BF is constant across body sizes, so variation in BF captures deviations from isometry — i.e. differences in morphological "compactness" — rather than absolute size. Elongated insects such as Culicidae have lower BF than compact ones such as Coleoptera of comparable length.

**Step 3 — inference.** Given the model's prediction BF̂ and the known physical scale *L* for the image, dry mass is recovered as

$$\hat{M} = \left(\widehat{\text{BF}} \times L\right)^3$$

The CNN backbone is EfficientNet v2-s with a single regression head (dropout 0.6 → linear → scalar). Training uses MSE loss on BF values, with data augmentation including random rotations, flips, colour jitter, Gaussian noise, elastic deformation, and a custom random-downscale transform that rescales the target by the cube of the scale factor to keep BF consistent. At inference, 8-view test-time augmentation (4 rotations × 2 flips, median-aggregated) is used.

---

## Repository structure

```
01_biomass_model/          Core model — corresponds to Section 2.3 of the paper
  01-preprocess.py           Segment images, compute L and BF for each specimen
  02-train.py                Train EfficientNet v2-s to predict BF
  03-predict.py              Run inference with 8-view TTA, recover M from BF̂
  models.py                  Model architecture definition
  split.py                   Specimen-level train / val / test split
  metadata.csv               Raw specimen metadata (~9,400 records)
  metadata_enriched.csv      Metadata with computed L and BF columns
  predictions.csv            Model predictions on the test set

02_experiments/            Case studies — Section 2.4 / 3 of the paper
  2025-12-15_drosophila_biomass/         Temperature × size experiment (Drosophila)
  2026-01-15_drosophila_classification/  Multi-task sex + species classification
  2026-01-15_drosophila_temp_prediction/ Predicting rearing temperature from images
  2026-01-20_spiders/                    Generalisation to spiders (Pachygnatha degeeri)
  utils/segment_utils.py                 Shared segmentation and barcode utilities

tools/                     Standalone utilities
  segment-images.py          Batch segmentation using FlatBug
  add-new-metadata.py        Append new specimen records
  data-selection.py          Dataset filtering helpers
  bf-sorting-images.py       Sort images by predicted BF for QC
```

The numbered prefix on `01_biomass_model` and `02_experiments` loosely mirrors the paper's Methods section order. Each experiment sub-directory under `02_experiments/` is self-contained and mirrors the case-study results reported in the paper.
