# TODO

Open items as of 2026-08-24. Companion to `HANDOVER_2026-08-24.md`, which records
how these were found. Numbers here were measured on this machine; the manuscript
referred to is the submitted version (`ecoevorxiv.org/repository/view/13446/`),
equations numbered as in that PDF.

---

## 1. Zenodo record — needs a new version

Record: [10.5281/zenodo.20543262](https://doi.org/10.5281/zenodo.20543262),
published 2026-06-04. Contains `EntoScan.zip` (5.9 GB) and `metadata.csv`.

The published `metadata.csv` was exported **before 2026-02-02**, so it predates
that day's corrections — four months stale at the time of publication. Two
concrete defects:

### 1a. 92 beetle images carry the wrong DPI

| | Zenodo | correct |
|---|---:|---:|
| DPI, the 92 `NOTES == "beetles"` rows | 3200 | **6400** |

Fixed in the repo by `1d23c69` (2026-02-02, "fix wrong dpi in beetles!"); the
Zenodo export missed it. All 92 are one batch, captured 2026-01-09, one image per
specimen, spread across all three splits (56 train / 17 val / 19 test).

**Why it matters:** the published `metadata.csv` has no `SCALE`, `ROI_SIZE_MM` or
`BF` column, so a user must derive them from DPI via Eq (1),
`L = 25.4·(224/s)/DPI`. Halving DPI doubles `L` and halves `BF`. Anyone training
on the release reproduces the exact bug that forced the February relabelling,
with no way to know.

Affected file list: `zenodo_wrong_dpi_beetles.csv` (regenerate by diffing the
Zenodo `metadata.csv` against `01_biomass_model/metadata.csv` on `IMAGE_FILENAME`).

That 6400 is the correct value rests on four independent lines of evidence:

1. `1d23c69` changes **only** those 92 DPI values across all 9,481 rows and does
   nothing else — a deliberate human correction, not a side effect.
2. The 3200 they replaced was never a measurement: `tools/add-new-metadata.py`
   hardcodes `"DPI": 3200` for every ingested batch (see §3a).
3. Compactness, `sqrt(area_mm²)/M^⅓`, against mass-matched non-beetle EntoScan
   images: at 6400 beetles are 21% **more compact** than average, as Fig 1d
   claims for Coleoptera; at 3200 they would be 58% **more elongated**.
4. Held-out model residuals (§ below) are 5–6 SD better under the 6400 labels.

Residual uncertainty: the physical tests cannot cleanly separate 4800 from 6400
(they bound the true value below at ~5,045 and rely on an assumed compactness
offset above that). **Confirm the scanner setting for the 2026-01-09 session**
before republishing — the segmented PNGs carry no embedded DPI metadata, so the
files cannot answer it.

### 1b. Four `IS_VALID = False` rows are in the release but not in the repo

All of specimen `2025-09-23_002_A1`, `NOTES = "Wrong data"`. The record has 7,448
rows of which 7,444 are usable; the local metadata has only the 7,444.

Not a defect in itself — `IS_VALID` marks them correctly — but decide whether the
release should carry them, and say so in the record description either way. A user
who filters on `IS_VALID` gets the right answer; one who does not, silently trains
on four known-bad images.

### 1c. State what the record does and does not contain

The record is **EntoScan only**: 7,444 valid images / 2,123 specimens. Verified
that every local EntoScan image is present and nothing else is. `biodiscover-S`
(2,037 images / 1,019 specimens) is **not** published there.

So the released dataset **cannot reproduce the published model**, which was
trained on all 9,481. Say this explicitly in the record description and in the
paper's data-availability statement, along with the reason biodiscover-S is not
redistributable.

### 1d. Verify the archive against its own index

Everything above was checked against the record's `metadata.csv`, the only file
readable without a full download. `EntoScan.zip` has **not** been opened — Zenodo
serves no `accept-ranges`, so counting its entries means pulling 5.9 GB. Given
that the index is demonstrably from a stale export, do not assume the archive
matches it. Check once before republishing.

---

## 2. Manuscript

### 2a. Corrections where the paper is wrong

- **Eq (6): `BF' = BF·s³` → `BF' = BF·s`.** `BF = M^⅓/L` is a cube root, so
  simulating a shrink by `s` gives `(s³M)^⅓/L = s·BF`; cubing applies the
  exponent twice. The submitted model was trained under the erroneous form, so
  also state that, with the sensitivity: a corrected retrain moves test MAE
  0.0824 → 0.0807 with overlapping bootstrap CIs, conclusions unaffected.
  *(Code already fixed in `783179b`; the paper is what needs updating.)*
- **Learning rate: 1×10⁻⁴ → 1.4×10⁻⁴.** The code uses the higher value
  (`sqrt` scaling from the batch-size increase). Update the paper to match.
- **`README.md:38`** repeats the cube-of-the-scale-factor error — same fix.
- **Dataset size, `02_methods.tex:40`.** "approximately 8,400 images of 2,100
  individual specimens" matches nothing. Correct figures:

  | | images | specimens |
  |---|---:|---:|
  | EntoScan | 7,444 | 2,123 |
  | biodiscover-S | 2,037 | 1,019 |
  | **total** | **9,481** | **3,142** |

  Figure 3a is **correct** — all twelve of its cells match the data exactly, and
  it sums to 9,481 / 3,142. It is only the prose that disagrees, with the paper's
  own figure. Split the sentence: one clause for what the model was trained on,
  one for what the DOI contains (§1c).
- **Abstract "R² > 0.95"** overstates. Results are 0.95 (EntoScan) and 0.89
  (biodiscover-S); neither exceeds 0.95 and the claim generalises from the better
  dataset. `04_discussion.tex:30` repeats it.
- **Spider case study describes two species, not one.** PD (*P. degeeri*, 1,913)
  and BG (1,383) across the four named sites. "1,913 individuals were imaged" is
  the PD count; 3,296 were imaged. Fig 3c's caption promises "species-specific
  patterns" while showing one species.

### 2b. Missing from Methods — needed for reproduction

- **Head dropout.** The paper describes the head as a bare linear layer
  ("the final layer was adapted for scalar regression"). The code has
  `Dropout(p)` → `Linear(→1)`. State the value. Note `aeaa98f` (2026-02-02
  16:43) changed it 0.4 → 0.6, **after** the submitted run `2026-02-02_15-11-40`
  started at 15:11 — so the submitted model used 0.4.
- **Colour jitter strengths** (0.2 each). "applied to brightness, contrast,
  saturation, and hue" is not reproducible without magnitudes.
- **Label smoothing**: state whether it was enabled for the reported model, and
  at what range.
- **The `conf > 0.90` detection filter** used in both case studies is
  undocumented. For spiders it discards 3,296 → 2,421 detections (26%).
- **Statistics section**: which R version, how the bootstrap was done, and the
  ANOVA (see §4a). `notes.tex:22` asks for the same thing.

### 2c. Augmentations removed from the code — no longer a mismatch

Gaussian noise (σ=7.0, p=0.2) and elastic transform (α=50, p=0.2) were in the
code but not in the paper's augmentation list. **Both removed** (commit
`4b6a2af`), so the paper's list is now exhaustive and no edit is needed. Recorded
here only so nobody re-adds them without also updating the manuscript.

The elastic transform was very unlikely to have been doing anything: torchvision
normalises the displacement field by image size, so α=50 gives a measured 0.67 px
mean / 3.6 px worst-case displacement on a 224 px image.

### 2d. Results worth adding

- **Specimen-level aggregation is an 11% free win.** Averaging a specimen's
  predictions takes test MAE 0.0807 → 0.0719 with no retraining. Report it
  per-source, not pooled: EntoScan is a 4-images-per-specimen protocol (1,690 of
  2,123 specimens have exactly 4) while biodiscover-S is exactly 2 (1,018 of
  1,019), so the gain is not uniform and the often-quoted "~3 images per
  specimen" describes neither.
- **Point-estimate mass is biased ~4.5% low.** Since `M = (BF·L)³` and
  `E[X³] ≠ E[X]³`: `E[M] ≈ (L·μ)³(1 + 3s²)` with `s ≈ 0.123`. Negligible per
  specimen; it does **not** average out in summed community biomass, where it
  accumulates as a one-sided underestimate — the direct target of both case
  studies. Correct it analytically or state it.

### 2e. Figure 3a

- Header typo: **`N Imgaes` → `N Images`**.
- Add a totals row (9,481 / 3,142) so the figure and the text can be checked
  against each other at a glance.
- `Biodiscover` vs `biodiscover-S` — settle on one name; if `-S` denotes a
  subset, say so.
- The table lists biodiscover rows but the DOI holds only EntoScan. Note it in
  the caption or the data-availability statement (§1c).

---

## 3. Code / pipeline

### 3a. `tools/add-new-metadata.py` hardcodes DPI — live bug

Line 178: `"DPI": 3200,` is stamped on **every** ingested batch regardless of the
resolution it was actually scanned at. This is how the beetles got 3200 on
2026-01-29 (`709014a`), and it will silently do the same to the next batch
scanned at any other resolution.

The manuscript states the acquisition software lets users "control acquisition
resolution" and that DPI is stored per image (`02_methods.tex:59`, feeding
Eq 1) — but the script that builds the training metadata discards that and
substitutes a constant. Read it from the EntoScan database, or make it a required
argument with no default.

### 3b. `--root` is dead in `02-train.py`

`__getitem__` uses `IMAGE_FILENAME` verbatim, and those values already carry the
`00_data/02_resized/` prefix, so `--root` has no effect and the script only works
when run from `01_biomass_model/`. The docstring claims resolution as
`<root>/<DATASET>/<IMAGE_FILENAME>`, which is not what happens.

### 3c. `01_biomass_model/best_model.pt` at the repo root is stale

It is still the February model. The current one lives under
`01_runs/regression_effnetv2_s/`. Either update it or delete it — as it stands it
is a trap.

---

## 4. Analysis / reproducibility

### 4a. The ANOVA behind Figure 3b is not in the repository

`02_methods.tex` lines 304–309 describe a Type III ANOVA with an `emmeans`
post-hoc, and Fig 3b's `***`/`ns` annotations depend on it. No `Anova()`,
`aov()`, `emmeans` or log-transform exists anywhere in the repo. Either write and
commit that code, or remove the annotations. **This is the largest
reproducibility gap.**

### 4b. Run-to-run seed variance is still unmeasured

Nothing below ~0.001 should be believed until it exists. `--seed` was added in
`4b6a2af` for exactly this; three seeds per arm is the minimum. The pair
originally used to estimate variance turned out to straddle two different
validation sets.

### 4c. Never compare across the validation-set eras

Three incompatible label sets exist in the run history — the dataset was
rewritten on 2026-02-02 (`9aa2dbb`, `1d23c69`). Recover a run's era from
`val_loss/(1-val_r2)`, which estimates the validation-set variance:

| implied val SD | runs |
|---|---|
| 0.05035 | `2026-01-30_15-00-26`, `2026-01-30_16-22-35` |
| 0.05632 | `2026-01-31_10-34-44`, `2026-02-02_09-40-39` |
| **0.05588** ← current | `2026-02-02_15-11-40`, `MSfix`, `NoSmooth` |

The often-quoted "0.01458 best ever" is from the old label set and is not a valid
target.

### 4d. `Rplots.pdf`, not `analysis.pdf`

`analysis.Rmd` in **both** case studies opens a second graphics device
(`pdf(w=16, h=9)` … `dev.off()`) mid-chunk. Those plots go to `Rplots.pdf`, not
into the knitted `analysis.pdf`. `Rplots.pdf` is the actual case-study figure.

### 4e. The case studies have no ground truth

Drosophila and spiders are applications, not validations. Swapping the model
changed their population biomass estimates by +6% and +19% with nothing to check
against. Treat the current PDFs as provisional until someone decides whether to
adopt the new model.

---

## 5. Decided, no action needed

- **The 92 beetles are not outliers in val or test.** Median absolute residual
  sits at the 56th (val) and 61st (test) percentile of the other images; in
  cube-root mass space they are predicted *better* than average in both splits.
  Dropping them moves the EntoScan MAE by +0.0005 (val) / +0.0003 (test).
  A small but real test-split BF bias of +0.0145 (95% CI [+0.0059, +0.0227])
  reads as a batch/taxon effect: a DPI error would show ~6 SD, not 0.7 SD.
- **The splits are clean.** Zero specimens span more than one split and zero
  `INSECT_ID`s are shared between the two data sources, so the grouping is
  genuinely by specimen and the val/test numbers are honest.
- **TTA removed** (`13c94b6`): worth +0.00079 MAE overall, 95% CI
  [−0.00020, +0.00179], for 8× the inference compute. Removing it also makes the
  manuscript's "training only" statement true.
