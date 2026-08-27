# BEEomass

Code for the [paper](https://ecoevorxiv.org/repository/view/13446/):

> **EntoScan and BEEomass: a standardized imaging system and a physically motivated model for high-throughput dry biomass estimation of arthropods**
> Melika Baghooee, Robert Thalheim, Fevziye Hasan, Søren Toft, Torsten Nygård Kristensen, and Quentin Geissmann

**Data and model.** Both are on Zenodo, linked by their concept DOIs, which always
resolve to the most recent version:

| | DOI |
|---|---|
| Dataset — paired images and dry biomass measurements | [10.5281/zenodo.20543261](https://doi.org/10.5281/zenodo.20543261) |
| Model — trained weights | [10.5281/zenodo.20624494](https://doi.org/10.5281/zenodo.20624494) |

The dataset holds three self-contained parts: the training data (9,481 images of
3,142 weighed specimens, from EntoScan and Biodiscover), the *Drosophila* experiment
(1,913 individuals), and the spider experiment (1,499 individuals). Each has its own
`metadata.csv` and `images/` directory.

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

The CNN backbone is EfficientNet v2-s with a single regression head (dropout 0.6 → linear → scalar). Training uses MSE loss on BF values, with augmentation by random 90° rotations, horizontal flips, colour jitter, a random choice of blur or sharpening, and a random-downscale transform.

The downscale transform rescales the target **linearly** in the scale factor *s*, not by its cube. The augmentation pastes the image, shrunk by *s*, onto a canvas of unchanged size, so *L* stays fixed while the specimen's mass under isometry becomes *s*³*M*; substituting into the definition above gives BF′ = (*s*³*M*)^(1/3)/*L* = *s*·BF. The cube root in BF already absorbs the cubic mass–length relationship.

Inference applies no augmentation. Test-time augmentation is available behind `--tta` but is off by default: it is worth about 1% of MAE, within noise, for eight times the compute.

---

## Repository structure

```
01_biomass_model/          Core model — corresponds to Section 2.3 of the paper
  01-preprocess.py           Segment images, compute L and BF for each specimen
  02-train.py                Train EfficientNet v2-s to predict BF
  03-predict.py              Run inference, recover M from BF̂ (--tta optional)
  models.py                  Model architecture definition
  split.py                   Specimen-level train / val / test split
  metadata.csv               Raw specimen metadata (9,481 records)
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
  bf-sortingi-images.py      Sort images by predicted BF for QC
```

The numbered prefix on `01_biomass_model` and `02_experiments` loosely mirrors the paper's Methods section order. Each experiment sub-directory under `02_experiments/` is self-contained and mirrors the case-study results reported in the paper.
