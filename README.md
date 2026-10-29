# Helio-N \[/ˈhɛliən/\]
https://www.youtube.com/watch?v=eGtwgYt_QnA&list=RDeGtwgYt_QnA

Helio-N prepares solar-image datasets, experiments with filament suppression in
existing IDL coronal-hole (CH) masks, and calculates and evaluates solar-wind
forecasts. The active workflow on Miracle is:

`Dataset → filament features/classification → filamentless masks → CH areas → SW archive/statistics`

U-Net training, boundary inference, and segmentation comparisons are archived.
See [Archived U-Net workflow](#archived-u-net-workflow) for the branch, removal
commits, and recovery commands.

## Environment

Conda/YAML environments live in `Config/Environment Defs/`:

- CUDA / Miracle: `icme3.12-cuda.yml` (`icme3.12-cuda`)
- Apple Metal: `icme3.12-metal.yml` (`icme3.12-metal`)
- CPU: `icme3.12.yml` (`icme3.12`)

For example, create and activate the CUDA environment before running the Python
commands below:

```bash
conda env create -f "Config/Environment Defs/icme3.12-cuda.yml"
conda activate icme3.12-cuda
```

The PIP requirements are in `Config/Environment Defs/requirements.txt`.
Environment definitions retain their historical deep-learning dependencies;
the current filament baseline uses NumPy/SciPy and does not require TensorFlow
or scikit-learn.

## Machine configuration

`Config/Machine.json` selects paths and runtime settings by hostname. Set
`MACHINE=miracle` explicitly for Miracle; `inherits` lets another entry reuse
settings while overriding paths.

- `fits_root`, `masks_root`, `hmi_root`: AIA 193 FITS, existing final IDL masks,
  and HMI data.
- `aia304_root`: optional AIA 304 Å FITS root; set to `null` if unavailable.
- `artifact_root`: host-specific dataset indexes and helper tables, normally
  under `Outputs/Artifacts/<host>/`.
- `filament_feature_workers`: filament-feature process count.
- `plot_threads`: also supplies the CH-area worker count through `apply_config`.

Historical fields remain part of the configuration contract, including
`train_batch_size`, `train_workers`, `train_max_queue_size`,
`train_use_multiprocessing`, `apply_batch_size`, `chunk_size`, and
`max_inflight_plots`. `Library/Config.py` still reads them. Keep these fields,
`Config/Plot.json` (`target_px`, `dpi`), and `Config/Postprocessing/` presets;
archiving a workflow does not remove configuration fields or files.

CH–SW correspondence models remain in `Models/CH_SW_Correspondence/`:
`Shugay.py` and `Shugay_Slow_SW.py`.

## Data preparation

Build the dataset index from existing AIA FITS, final IDL masks, HMI, and
optional AIA 304 data:

```bash
MACHINE=miracle python Scripts/Make.py Dataset
```

- Roots come from `Config/Machine.json`.
- By default, keep all matches. The optional `hourly` argument keeps one
  observation per hour. Pass `--start YYYYMMDD --end YYYYMMDD` together to
  restrict the input dates, inclusively.
- Writes `Outputs/Artifacts/<host>/Paths.parquet` plus missing-data CSVs.
- Existing masks are inputs; this command does not determine CH boundaries.
- Run `python Scripts/Make.py` to list available Make commands.

## CH areas and SW inputs

Calculate a baseline from the original IDL masks before a suppression experiment:

```bash
MACHINE=miracle python Scripts/Make.py CH_Areas 20180101 20181231 \
  --area-mode idl-exact --validate-db \
  --output-parquet "Outputs/Filaments/CH Areas Baseline 20180101-20181231 idl-exact.parquet"
```

`--paths-parquet` selects an alternate dataset index, including one whose
`mask_path` column points to filament-suppressed masks. Keep baseline and
experimental area outputs separate with `--output-parquet`.

`idl-exact` writes `s_idl_exact_pct` in the database's percent units, exact FITS
`source_dt`, and the compatible `dt`, `ch_relative_area`, and `forecast_sw_speed`
columns. `--validate-db` compares original-mask speeds against the frozen
pre-2019 database series at exact observation timestamps. Use it for the
baseline, not as an equality gate on intentionally altered masks.

The default `projected` mode retains the older projected-area fraction `s_idl`.
It is a different area definition and unit convention from `idl-exact`.
Neither mode runs U-Net or writes U-Net comparison columns.

## Solar-wind CR archive

```bash
python Scripts/Make.py propagate_sw 2203 --input-source parquet \
  --input-parquet "Outputs/Filaments/from-miracle/CH Areas 20180101-20181231 idl-exact.parquet" \
  --validation-archive-root "Data/CDAWeb Validation" \
  --archive-root Outputs/SW/Samples/Archive
python Scripts/Make.py make_animation "2018-04-22 00:00" "2018-04-22 06:00" \
  --archive-root Outputs/SW/Samples/Archive
# The padded CR movie also needs CR 2202 and 2204:
python Scripts/Make.py propagate_sw --start 2018-04-12 --end 2018-05-23 \
  --input-source parquet \
  --input-parquet "Outputs/Filaments/from-miracle/CH Areas 20180101-20181231 idl-exact.parquet" \
  --validation-archive-root "Data/CDAWeb Validation" \
  --archive-root Outputs/SW/Samples/Archive
python Scripts/Make.py make_animation 2203 \
  --archive-root Outputs/SW/Samples/Archive
```

The production `propagate_sw` default is SQL and writes to
`Outputs/SW/Archive/CR####/`. Each CR contains rounded-core `series.parquet`,
source and prepared-input Parquets, a compressed HDF5 cube with separate
`speed` and `is_slow_wind` arrays, and a manifest. It refuses to overwrite an
existing CR. Astronomical CR boundaries are rounded to the nearest output
hour; the original bounds remain in each manifest. Adjacent products therefore
concatenate on one hourly lattice. `Library.SW.Archive.load_cube` and
`load_series` ingest arbitrary UTC ranges, including CR boundaries;
`load_cube(..., context_frames=1)` includes the preceding final snapshot when
an analysis starts at the next CR's first hourly centre. Both
`make_animation` and `sw_stats` accept either a single CR number or an
explicit start/end date range. In CR mode, animation frames span three days
before the CR through three days after it; the time-series panels also preload
their full look-back/look-ahead window so the opening frame is populated when
the neighboring archive exists. Date-range mode uses the requested interval
exactly. Every animation frame labels its current CR. The movie command never
runs propagation.
ACE, STEREO-A, STEREO-B, PSP, and Solar Orbiter validation observations come
from their post-merge CDAWeb products under `Data/CDAWeb Validation/merged/`;
intermediate `chunks/` are ignored. Propagation defaults to all five
validation spacecraft; `--satellites all` can make that selection explicit.
Each per-CR manifest records the merged product path and checksum. Sequential
reuse accepts an existing archive whose satellite set is a superset of the
request and verifies provenance for every requested spacecraft. If requested
satellites are missing, those CRs are recomputed in place; `--no-reuse`
recomputes every requested CR and replaces existing archives. This is the
default validation archive root and can be overridden with
`--validation-archive-root`.
`make_animation --satellites ace_earth,stereo_a` selects plotted spacecraft
in that order; `--satellites none` produces a polar-only movie. By default it
plots every spacecraft present in the archive. Selection removes both the
polar marker and time-series panel, without changing archived comparisons.
The direct `plot_polar_snapshot()` and `export_polar_animation()` calls accept
the same IDs as a `satellites=[...]` argument (or `[]` for none).
Each per-CR satellite series also stores `hee_beta_deg` and
`sdo_observation_age_days`. The latter is elapsed Carrington synodic rotation
time since the satellite-facing solar longitude last left SDO's visible disk
(zero while it remains visible). Thin tracks at the bottom of each animation
speed panel mark ages over 10 days in yellow and `abs(hee_beta_deg) > 10`
degrees in pink; compact badges mark flags at the current frame. Polar
satellite markers keep a black fill with a white outline for contrast.
`Scripts/Make/SW_Stats.py` reconstitutes an arbitrary interval from per-CR
archives and writes four workbooks: `SW <range>.xlsx`
with one hourly sheet per satellite plus Russian variable definitions, and
`SW <range> (Per-CR).xlsx` with one per-satellite sheet of CR-level forecast
scores, plus `SW <range> (Summary).xlsx` with a `Global` sheet and one
`<satellite> (Yearly)` sheet per satellite. Summary sheets contain valid-pair
counts, Pearson R, RMSE, MAE and bias for the same comparisons and ICME regimes
as the CR scores. They are computed directly from the hourly pairs, rather
than averaging CR or yearly metrics. Calendar years are split at January 1
UTC; `start_utc` and exclusive `end_utc` show the actual coverage, including
partial first/last years. Errors and bias are in km/s. `--summary-stats-out`
overrides the summary workbook path. Workbook names use date-only bounds, such as
`SW 20250101-20260101.xlsx`; animation files use the same `SW <range>` pattern.
`SW <range> (Event-based).xlsx` contains annual HSS counts, hit/miss rates,
and mean signed peak underreporting, with Yermolaev versus Earth and separate
B1 sheets for each satellite. Misses include the no-CH subset (raw speed
300 km/s throughout the event window). Peaks use the catalog interval padded
by 12 hours on each side; the hit threshold is strictly above 400 km/s.
Events belong to their start year. Partial model windows use the available
peak; windows with no finite prediction are unscored. No-CH classification
requires complete coverage at 300 km/s.
This report ignores ICME flags. Catalogs default to `Outputs/HSS List`;
`--hss-list-dir` and `--event-stats-out` override the inputs and output path.
See `Analysis/HSS Validation Spec.md` for the complete definitions.
Use `--events-only` to write just the event-based workbook, without rebuilding
the other reports or loading the original observations/ICME catalogs:

```bash
conda run -n icme3.12-metal python Scripts/Make/SW_Stats.py 2306 --events-only
```

For another machine, use its available Conda environment and copy the existing
CR archive directories (`manifest.json`, `series.parquet`, and `cube.h5`). The
six HSS CSVs in `Outputs/HSS List` are versioned with the code. No propagation
or catalog generation is needed. `--archive-root` selects a different archive
location. Available neighboring CRs supply padding; events remain unscored
when their entire padded window has no finite prediction.
The usual local environment/configuration is still required, including
`userpwd.py` for the existing library imports; event-only mode makes no database
connection.
`SW 4.ipynb` runs a CR 2306 SQL sample through the library functions directly,
with separate cells for inspecting source inputs, prepared inputs, and the
in-memory cube. Its sample archive and animation omit satellite comparisons.

`inputs.parquet` retains normalized source rows (including missing values) over
the source guard; `prepared_inputs.parquet` adds the empirical speeds used to
build launch knots. `series.parquet` keeps core-period CH-area/input speeds and
standard satellite comparisons. The satellite registry keeps `ace` (native
ACE spacecraft coordinates) separate from `ace_earth` (the propagated
ACE-at-Earth reference at the fixed Earth HEE point). The default comparison
uses native ACE. ACE comparisons use the native
`Data/ACE 1h.parquet` source and its time-dependent HEE position, rather than
the fixed Earth-point coordinate series. They always store both
`v_predict_raw` (the propagated forecast with the 300 km/s slow-wind baseline)
and `v_predict` (the same forecast with a time-varying empirical slow-wind patch),
plus `slow_sw_patch_mask`. The patch is applied to the ACE comparison series;
it does not modify the propagated speed cube. STEREO-A keeps the unpatched
series. Existing archives without the patch mask require `--no-reuse` to be
rebuilt; archives missing requested satellites are recomputed automatically.
The hourly satellite workbook uses the requested range reconstructed from the
per-CR archive and retains empty instrument fields rather than filling them
with model values; magnetic components retain the source instrument frame.
`propagate_ballistic()` returns the speed cube; `cube_stats()` leaves it
unchanged and computes the slow-wind mask, limits, and counts. The present mask
is inferred by output-time speed equality, so a future time-varying slow-wind
model will need source provenance carried by propagation.
Ballistic source knots are joined only across gaps of at most 12 hours;
longer gaps remain empty. This is separate from satellite gap handling:
STEREO-A speed and position are interpolated only between nonempty sampled bins
at most 6 hours apart; longer gaps remain missing. Comparison-frame position
sampling obeys the same limit. Previously written CR archives are unchanged
by this loader correction.

SQL input still uses upstream `sdo_fill_sw_193(...)`; the archive manifest
marks that unresolved provenance question. The 2018 Parquet above is a local
example artifact, not an included repository dataset.

## H-alpha-supervised filament removal

The filament classifier operates on connected components of the exact IDL
`*_CH_MASK_FINAL.png` masks. The baseline is an L2-regularized logistic model
using component geometry, AIA 193 intensity/contrast, AIA 304
intensity/contrast, and HMI field/polarity features. Catalog distances and match
flags generate labels but are not model inputs. If `--labels-parquet` is
provided, Kislovodsk and MAGFiLO positives are combined with an either-or rule;
frames without either catalog are never treated as negative examples.

After the exact final masks and AIA 304 files are present on Miracle:

```bash
MACHINE=miracle python Scripts/Make.py Dataset
MACHINE=miracle python Scripts/Train_Filament.py 20170101 20171231 \
  --training-only --features-only
MACHINE=miracle python Scripts/Train_Filament.py 20170101 20171231 --reuse-features
```

The first command rebuilds `Paths.parquet` using only exact final masks. The
feature-only run is a label-health gate: do not train unless it reports both
classes. The trainer uses a chronological day split, selects a
high-precision validation threshold, saves an inspectable JSON model, and
rejects any 2018 training data.
`--training-only` is intentionally limited to 2017 feature construction: it
reads each mask first, then skips empty masks and catalog-uncovered frames
before expensive AIA 304/HMI work. Do not use it for 2018 inference features.
Pass the same full region-level catalog table through `--labels-parquet` to
both trainer invocations when using the Kislovodsk-or-MAGFiLO label union.
Feature collection uses a bounded process pool: 8 workers on Miracle and 3 on
risc1, configurable with `--workers`.

Build 2018 features, remove predicted filament components without overwriting
the IDL masks, and calculate the compatible area/SW input:

```bash
MACHINE=miracle python Scripts/Train_Filament.py 20180101 20181231 --features-only
MACHINE=miracle python Scripts/Apply_Filament.py 20180101 20181231 \
  --model-path "Outputs/Filaments/Classifier 20170101-20171231.json" \
  --features-parquet "Outputs/Filaments/Features 20180101-20181231.parquet"
MACHINE=miracle python Scripts/Make/CH_Areas.py 20180101 20181231 \
  --area-mode idl-exact \
  --paths-parquet "Outputs/Filaments/Paths Filamentless 20180101-20181231.parquet" \
  --output-parquet "Outputs/Filaments/CH Areas Filamentless 20180101-20181231 idl-exact.parquet"
```

Validation gates:

- Before filament removal, exact-mask SW speeds must match the database-derived
  frozen series within ±2.5 km/s using `CH_Areas.py --validate-db`. Validation
  joins exact FITS observation timestamps only; hourly-bin matches are invalid
  because the SW builder interpolates the hourly series to 2.5-minute cadence.
- Report component classification metrics only on catalog-covered 2018 frames.
- Compare baseline and filamentless 2018 area parquets through the same frozen
  CH-SW and heliosphere pipeline; keep the frozen 2018 SWX result as a separate
  reference and do not fit any threshold or model on 2018.

## Filament experiment notebooks

`Analysis/Filament Removal.ipynb`, `Analysis/Filament Thresholding.ipynb`, and
`Analysis/Orientation Self-Test.ipynb` now load existing masks. For ISWAT cases,
configure the image and mask roots, then inspect the recorded same-day mask
time offsets before interpreting comparisons. Missing masks fail explicitly;
blank placeholders are not used as candidate boundaries.

The threshold notebook writes to `Outputs/Filaments/ISWAT IDL Thresholding/`,
separate from the historical boundary-detector cache. Old component IDs,
features, and manual labels must not be reused for different boundaries.
Saved notebook outputs were preserved as historical evidence; rerun the
relevant groups before treating them as results of the current IDL workflow.
`SW 2.ipynb` retains only the IDL comparison and exports `Paper Stats IDL.xlsx`.

## Synoptic copy helper
Sync FITS/masks/HMI trees between the main and “mini” hosts.

```bash
python Scripts/Make.py Synoptic up    # copy miracle -> miracle_mini
python Scripts/Make.py Synoptic down  # copy miracle_mini -> miracle (rsync entire roots)
python Scripts/Make.py Synoptic inplace  # build synoptic subset only
```

- Relies on `miracle` and `miracle_mini` entries in `Config/Machine.json`.
- Uses rsync; creates missing destination directories.
- In `up` mode, builds a synoptic subset (00/06/12/18) before copying.
- In `inplace` mode, only builds `Paths (Synoptic).parquet` from `Paths.parquet`.

## Archived U-Net workflow

The complete pre-removal source snapshot is on **`archive/unet-segmentation`**,
at commit `cc5b073bfcd0775053b9960cf250761a775a7978`. The branch was created
locally; if its name is unavailable in another clone, use that commit as the
starting ref.

The removal was split into two commits:

- **Stage 1 — `961c88759df8df8b72662b0e92738503f223b9eb`:** removed complete
  U-Net files, including `Library/Model.py`, `Library/Processing.py`,
  `Library/Plot.py`, `Models/Segmentation/`, training/inference scripts, and
  segmentation-specific notebooks and statistics.
- **Stage 2 — `7c39ec3be26368ef070d774aed3e38afaba6fdb2`:** edited the retained
  scripts and notebooks to remove U-Net portions. This is the commit to inspect
  or reverse when restoring shared integration code: `Library/IO.py`,
  `Library/CH.py`, `Library/Metrics.py`, `Models/__init__.py`,
  `Scripts/Make/CH_Areas.py`, the three retained filament/orientation notebooks,
  and `SW 2.ipynb`.

To resume from the original source snapshot in a separate checkout:

```bash
git worktree add -b resume/unet-segmentation ../helio-n-unet archive/unet-segmentation
git show archive/unet-segmentation:README.md
```

To restore the U-Net code onto a newer `main` instead, start a separate branch
and reverse the shared-code edits before restoring the deleted files:

```bash
git switch -c restore/unet-on-main main
git revert --no-commit 7c39ec3be26368ef070d774aed3e38afaba6fdb2
git revert --no-commit 961c88759df8df8b72662b0e92738503f223b9eb
git diff --cached
```

Review the staged restoration and resolve conflicts with later work before
committing. The archive's README contains the original training and inference
usage notes. Restoring only the deleted files is insufficient because stage 2
also changed their shared APIs and consumers.

Model checkpoints (`Outputs/Models/*.keras`), cached probability maps, and most
`Outputs/` research artifacts are ignored by Git. They need a separate copy or
backup; neither the archive branch nor a new worktree supplies those files.
Configuration fields, presets, and environment definitions were retained on
`main` throughout the removal.

## Notes
- The filament baseline uses NumPy/SciPy and does not require TensorFlow or
  scikit-learn.
- `Library/Config.py` selects the host section from `Machine.json` and exposes
  the retained path and runtime settings.
- `Scripts/Train_Filament_Per_Pixel.py` and
  `Scripts/Make/ISWAT_Filament_Gallery.py` remain experimental. Both currently
  import `CATALOG_MIN_ALIGNED_CENTERLINE_PX` and
  `CATALOG_MIN_ALIGNED_CENTERLINE_FRACTION`, which are absent from
  `Library/Filaments.py`. This pre-existing catalog-interface mismatch must be
  reconciled before those two tools can run; the region trainer above does not
  have this import error.
- Run `python -m Scripts` to list available commands.
