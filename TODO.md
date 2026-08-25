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

### 1c-bis. The published model is the February one, trained under the Eq (6) bug

The model is released separately at
[10.5281/zenodo.20624495](https://doi.org/10.5281/zenodo.20624495) ("BEEomass",
Baghooee & Geissmann, published 2026-06-10, CC-BY-4.0). It contains exactly one
file: **`2026-02-02_15-11-40.zip` (75.7 MB)** — byte-size matches the local
`01_runs/regression_effnetv2_s/2026-02-02_15-11-40.zip`.

So the publicly released weights are the February run, which means they were
trained with `BF' = BF·s³` (the erroneous exponent), dropout 0.4, and TTA still
in the inference path. Whatever the paper ends up saying about Eq (6) (§2a) must
also be true of this record, and if the model is ever replaced the record needs a
new version rather than a silent swap.

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
- **Learning rate: 1×10⁻⁴ → 1.4×10⁻⁴, and batch size: 32 → 64.** These go
  together: the LR is `sqrt(64/32)` scaling from the batch increase, so the paper
  must be updated for both or neither. Update to match the code.
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
- **Abstract "R² > 0.95" is not supported.** Under the paper's own bootstrap
  (one random image per specimen, resample, 500 reps), on the test split:

  | | R² (1−SS_res/SS_tot) | 95% CI | P(R² > 0.95) |
  |---|---:|---|---:|
  | EntoScan | 0.9553 | [0.9352, 0.9702] | 0.73 |
  | biodiscover-S | 0.8451 | [0.6948, 0.9371] | 0.00 |
  | pooled | 0.9495 | [0.9256, 0.9684] | **0.51** |

  Pooled is a coin flip, not "> 0.95". Only EntoScan alone clears it, and only
  73% of the time. The February model (the published one) is very slightly worse
  on both: pooled 0.9486, EntoScan 0.9505. `04_discussion.tex:30` repeats the
  claim. Say "≈0.95 on EntoScan, 0.85–0.89 on biodiscover-S" and drop the
  inequality.

- **Two different quantities are both called R².** `analysis.Rmd` uses
  `summary(lm(PRED ~ OBS))$r.squared` — squared correlation, which does not
  penalise bias or slope error. `02-train.py` uses `1 − SS_res/SS_tot`. On the
  test split they agree for EntoScan (0.9581 vs 0.9579) but differ by **+0.030
  for biodiscover-S** (0.8730 vs 0.8428), so the published 0.89 is the forgiving
  form and the coefficient of determination is 0.84. State which is used, and
  prefer the second. Changing it moves a published number, so decide deliberately.

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

### 4e. Recover the collection dates for five spider samples

Five vials carry `2022-00-00` in their data-matrix label — an invalid month and
day, i.e. a "date unknown" sentinel:

```
SPI_011_BG_HHJ_2022-00-00     SPI_038_PD_MSJ_2022-00-00
SPI_028_PD_FHJ_2022-00-00     SPI_047_PD_UTJ_2022-00-00
SPI_034_PD_MSJ_2022-00-00
```

They are five distinct field samples — those `sample_id`s never appear with a
real date — spanning all four sites (MSJ 268, HHJ 108, UTJ 97, FHJ 49 detections).

**The `2022` is a sentinel, not a year.** The dated samples run 2023-03-27 to
2025-05-09, and `02_methods.tex:168` states "collected between March 2023 and May
2025" — the paper's range is exactly the dated subset. If any sample were really
from 2022 the published range would be wrong. Combined with `00-00` being an
invalid month and day, these specimens are **undated**, not from 2022. So this
does not extend the study period; it is missing data inside it.

**Ask whoever ran the field sampling whether the dates are recoverable from a
field notebook or the sample database.** Nothing in the repository describes
these samples: `SPI_0*` appears only in the derived `metadata.csv`,
`metadata_enriched.csv` and `predictions.csv`, all generated by the pipeline
itself. The sole source of sample identity is the data-matrix barcode on the
vial, decoded at scan time by `parse_barcode()` in `segment_utils.py`. There is
no manifest, field-record file or README anywhere under the spider experiment —
so if the dates are not in an external notebook or database, they are
unrecoverable.

It costs 522 detections (15.8% of all), **354 of them surviving `conf > 0.90` —
14.6% of the analysed set.** That is a large piece of the seasonal series.

**Meanwhile, fix the silent inconsistency they cause.** `as.POSIXct` turns the
placeholder into `NA` (verified: a true `NA`, not a date coerced to 2021-11-30 —
some parsers do that and would silently place these animals at day-of-year 334).
`ggplot` then drops them from the two time-course density plots with only a
buried warning, **but the mass histogram at `analysis.Rmd:75` uses the unfiltered
`metadata` and includes all 354.** So one document reports seasonal figures over
2,067 animals and a distribution over 2,421, with nothing saying so.

Filter once, explicitly, up front:

```r
n_before <- nrow(metadata)
metadata <- metadata[!is.na(date)]
message(sprintf("dropped %d detections with unparseable dates", n_before - nrow(metadata)))
```

or keep them for the distribution plot deliberately and state the difference in
the caption.

### 4f. The case studies have no ground truth

Drosophila and spiders are applications, not validations. Swapping the model
changed their population biomass estimates by +6% and +19% with nothing to check
against. Treat the current PDFs as provisional until someone decides whether to
adopt the new model.

---

## 5. Publishing a new model — do it in this order

The current release (§1c-bis) is the February run, trained under the wrong Eq (6)
exponent. Replacing it is worth doing, but the steps are order-dependent and
several of them are easy to get wrong.

### 5a. Finish and evaluate the retrain

`MSaligned_2026-08-24_16-54-27` — manuscript-aligned except the exponent (`BF·s`)
and the batch/LR pair (64 / 1.4e-4), both of which the paper is being updated for.
Label smoothing re-enabled per Eq (7); Gaussian noise and elastic removed.

**Result (500 epochs, completed 2026-08-24 22:56).** Selection epoch 446;
best val MAE **0.01531**, the lowest of any run. Test set, paired bootstrap over
the same 972 images / 340 specimens:

| comparison | pooled ΔMAE | 95% CI | |
|---|---:|---|---|
| MS-aligned − Feb 02 (published) | −0.00057 | [−0.00660, +0.00519] | n.s. |
| MS-aligned − NoSmooth | +0.00045 | [−0.00410, +0.00554] | n.s. |

Every per-dataset comparison is also n.s. **The three models are statistically
indistinguishable on test.** Absolute test figures (bootstrap, paper's estimator):
MS-aligned pooled MAE 0.0800 / R² 0.9482; NoSmooth 0.0793 / 0.9500; Feb 02
0.0809 / 0.9477.

Note the paired CIs here are ±0.006, far wider than the ±0.001 of the TTA
comparison — two independent training runs disagree per-image much more than two
inference modes of one model, so 340 test specimens cannot resolve differences
below ~0.006 MAE. **No amount of re-running will make these separable on this
test set.**

**The useful conclusion:** re-enabling label smoothing per Eq (7) and removing the
two undocumented augmentations costs nothing measurable. The manuscript's
described method and the code can be reconciled without giving up performance —
which is the reason to adopt it, not a performance gain.

**Do not claim it is better.** It differs from `NoSmooth` in two ways at once
(smoothing on, two augmentations gone), so even a real difference could not be
attributed, and run-to-run seed variance is still unmeasured (§4b). If seed
variance is wanted for its own sake, 3 seeds at 250 epochs per arm via `--seed`;
but note this run selected at epoch 446, so 250 would have truncated it.

Compare with the paper's own estimator — one random image per specimen, resample,
500 reps — not raw per-image means, which read ~0.006 higher on R².

### 5b. DONE — adopted 2026-08-24

`MSaligned_2026-08-24_16-54-27/best_model.pt` is now the default `--weights` in
`03-predict.py` and both `04-inference.py`. All three `predictions.csv`
regenerated, all three `analysis.Rmd` re-knitted. Previous outputs preserved as
`predictions_nosmooth_backup.csv` alongside the existing `_feb02_` and `_tta_`
backups (all untracked, as before).

Test-set performance, bootstrap, adopted model: EntoScan MAE 0.0813
[0.0683, 0.0955], R² 0.9516 [0.9277, 0.9691]; biodiscover-S MAE 0.0769
[0.0583, 0.0999], R² 0.8522 [0.6926, 0.9471]. Statistically indistinguishable
from both predecessors (§5a).

**Note `01_biomass_model/best_model.pt` at the repo root is still the February
file and is now doubly stale.** Nothing references it (§3c) — delete it.

### 5b-bis. The case studies move far more than the test set does

Total predicted dry mass over each case study's full detection set:

| | drosophila (n=1,913) | spiders (n=3,296) |
|---|---:|---:|
| Feb 02 (published) | 510.1 mg | 3450.1 mg |
| NoSmooth | 539.8 mg (+5.8%) | 3178.9 mg (−7.9%) |
| **adopted (MS-aligned)** | **512.3 mg (+0.4%)** | **4056.4 mg (+17.6%)** |

Three models that cannot be told apart on the test set (all pairwise
comparisons n.s., §5a) produce **spider community biomass estimates spanning
27.6%** — NoSmooth to adopted. Drosophila is stable across all three; spiders
are not.

Two reasons, both worth stating in the paper. `M = (BF·L)³`, so a small BF shift
is cubed. And the case studies are out-of-distribution with no ground truth
(§4f), so nothing constrains the extrapolation — test-set equivalence does not
transfer to them.

**Consequence:** any absolute community-biomass number from the spider case study
carries a model-choice uncertainty far larger than its reported precision, and
that uncertainty is invisible in the test metrics. Report relative/temporal
patterns, which are what Fig 3c actually shows, or attach this sensitivity.

### 5c. Regenerating everything downstream (reference)

Order matters:

1. `03-predict.py` → `01_biomass_model/predictions.csv`
2. `02_experiments/*/04-inference.py` → both case-study `predictions.csv`
3. Re-knit all three `analysis.Rmd`. **The case-study figures are `Rplots.pdf`,
   not `analysis.pdf`** (§4d).
4. Re-derive every number in Results and Fig 3a/3b/3c.

The case studies have no ground truth (§4f): swapping the model moved their
population biomass estimates by +6% (drosophila) and +19% (spiders) with nothing
to check against. That shift is a reason for care, not evidence of improvement.

### 5d. Publish as a *new version*, not a new record

Use Zenodo's "New version" on
[10.5281/zenodo.20624495](https://doi.org/10.5281/zenodo.20624495). That keeps the
concept DOI resolving and leaves the old version citable, which matters because
the submitted paper reports the February weights. A fresh record would orphan
every existing citation and leave two unrelated DOIs for the same model.

Include in the upload, none of which the current record has:

- the weights (`best_model.pt`), not the whole run directory of checkpoints
- the **exact training command** and the resolved config line the run logged
  (`seed=… downscale_min=… label_smooth=(…) gaussian_noise=off elastic=off`)
- the **git commit hash** the run was launched from
- which `metadata_enriched.csv` produced the labels — the DPI fix and the split
  assignment are both baked into it (§1a, "Training does not read DPI" below)
- a one-line statement of what changed versus the previous version

### 5e. Republish the dataset record too, and keep the two consistent

§1a–1c. Do the dataset first or at the same time: a corrected model trained on
labels the public dataset still gets wrong would be worse than the current state,
because the two records would disagree with no way for a reader to tell which is
right.

### 5f. Then update the manuscript

§2. The Eq (6) correction, the LR/batch pair, and the model-record version all
have to tell the same story. Do not update the paper before the records exist —
the paper should cite versions that resolve.

---

## 6. Decided, no action needed

- **The 92 beetles are not outliers in val or test.** Median absolute residual
  sits at the 56th (val) and 61st (test) percentile of the other images; in
  cube-root mass space they are predicted *better* than average in both splits.
  Dropping them moves the EntoScan MAE by +0.0005 (val) / +0.0003 (test).
  A small but real test-split BF bias of +0.0145 (95% CI [+0.0059, +0.0227])
  reads as a batch/taxon effect: a DPI error would show ~6 SD, not 0.7 SD.
- **The splits are clean.** Zero specimens span more than one split and zero
  `INSECT_ID`s are shared between the two data sources, so the grouping is
  genuinely by specimen and the val/test numbers are honest.
- **Training does not read DPI, and stays that way.** `02-train.py` requires only
  `IMAGE_FILENAME`, `DATASET` and `BF_cbrMG_MM`; DPI is consumed once, upstream,
  in `01-preprocess.py:157`, where it produces `ROI_SIZE_MM`, `BF_cbrMG_MM` and
  `AREA_MM2`. Deliberate — do not plumb DPI into the trainer.

  The consequence to remember: correcting a DPI means **re-running
  `01-preprocess.py`** to regenerate the derived columns, not editing the DPI
  column in place. And since the model is scale-blind (every image is resized to
  224), a wrong DPI is invisible to it — it just learns a wrong target.
- **Only *P. degeeri* is reported for the spider case study — deliberate.** The
  data also contain a second species, BG (1,383 individuals against PD's 1,913,
  across all four sites). Excluding it from the manuscript is a decision, not an
  oversight; do not re-raise it. Consequently "1,913 individuals were imaged" is
  correct for the experiment as reported. The analysis code still generates a BG
  panel, so expect it in `Rplots.pdf` and ignore it.
- **TTA removed** (`13c94b6`): worth +0.00079 MAE overall, 95% CI
  [−0.00020, +0.00179], for 8× the inference compute. Removing it also makes the
  manuscript's "training only" statement true.
