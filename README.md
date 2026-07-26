# Helio-N \[/ˈhɛliən/\]
https://www.youtube.com/watch?v=eGtwgYt_QnA&list=RDeGtwgYt_QnA

This repo contains a small set of scripts to prepare data, train the U-Net, run inference, and shuttle artifacts between machines.

## Environment
- Conda/YAML envs live in `Config/Environment Defs/`:
  - CUDA: `icme3.12-cuda.yml`
  - Apple Metal: `icme3.12-metal.yml`
  - CPU: `icme3.12.yml`
- Example: `conda env create -f "Config/Environment Defs/icme3.12-cuda.yml" && conda activate icme3.12`
- PIP fallback: `pip install -r "Config/Environment Defs/requirements.txt"`

## Machine configuration
- Per-host settings are in `Config/Machine.json`. The key **must** match `socket.gethostname()`. It's possible to override the host with $MACHINE
- Required keys:
  - `fits_root`, `masks_root`, `hmi_root`: where raw data lives
  - `aia304_root` (optional): AIA 304 Å FITS root (set to `null` if unavailable)
  - `artifact_root`: where parquet/plots/models are written (per host)
  - `train_batch_size`, `apply_batch_size`, `chunk_size`, `max_inflight_plots`, `plot_threads`
- Optional: `inherits` lets a host clone another entry and override only a few paths.
- Global plot settings: `Config/Plot.json` (`target_px`, `dpi`).
- Paths/parquet outputs live under `Outputs/Artifacts/<hostname>/`.
- Models save to `Outputs/Models/<architecture><date_range>.keras`.
- Segmentation model definitions and date ranges live in `Models/Segmentation/` (e.g., `Models/Segmentation/A1.py`).
- CH-SW correspondence model modules live in `Models/CH_SW_Correspondence/` (currently `Shugay.py`).

## Data preparation
Build the dataset parquet from raw FITS/masks/HMI roots.

```bash
python Scripts/Make.py Dataset [hourly]
```

- Uses roots from `Config/Machine.json`.
- Default is `hourly=False` (keep all matches). Pass `hourly` to keep one sample per hour per day.
- Writes `Outputs/Artifacts/<host>/Paths.parquet` plus helper CSVs for missing data.
- Run `python Scripts/Make.py` to list available Make scripts.

## Training
```bash
python Scripts/Train.py <architecture_id> <date_range_id>
```

- Example: `python Scripts/Train.py A2 D1`
- Loads `Models/Segmentation/<architecture_id>.py` and injects `train_batch_size` from `Machine.json` only if `batch_size` is not set in the model definition.
- Date ranges are defined in `Models/Segmentation/<architecture_id>.py` and selected by `<date_range_id>`.
- Uses generator-based training with optional `correct_steps_by_n`.
- Saves model to `Outputs/Models/<architecture_id><date_range_id>.keras`.

## Applying (inference + plots)
```bash
python Scripts/Apply.py <architecture_id> <date_range_id> <postprocessing> <start> <end>
```

- Example: `python Scripts/Apply.py A2 D1 P1 20170601 20170701`
- `<postprocessing>` must match a file in `Config/Postprocessing/` (e.g., `P0`, `P1`, `Custom`).
- `<start>`/`<end>` slice the Paths.parquet index (timestamp strings like `YYYYMMDD_HHMM`).
- Uses `apply_config` from `Machine.json` (`apply_batch_size`, `chunk_size`, `plot_threads`, `max_inflight_plots`).
- Outputs:
  - `.npy` pmaps via `Library.IO.pmap_path` (co-located with masks)
  - CH overlay PNGs and mask-only PNGs for requested and baseline `P0` postprocessing

## Stats
```bash
python Scripts/Make.py Stats <architecture_id> <date_range_id> <postprocessing> [synoptic]
```

- Example: `python Scripts/Make.py Stats A1 D1 P1`
- Add `synoptic` to read `Paths (Synoptic).parquet` instead of `Paths.parquet`.
- Writes `Outputs/Artifacts/<host>/Stats/<architecture><date_range><postprocessing>_stats.parquet`.

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

Build 2018 features, remove predicted filament components without overwriting
the IDL masks, and calculate the compatible area/SW input:

```bash
MACHINE=miracle python Scripts/Train_Filament.py 20180101 20181231 --features-only
MACHINE=miracle python Scripts/Apply_Filament.py 20180101 20181231 \
  --model-path "Outputs/Filaments/Classifier 20170101-20171231.json" \
  --features-parquet "Outputs/Filaments/Features 20180101-20181231.parquet"
MACHINE=miracle python Scripts/Make/CH_Areas.py 20180101 20181231 \
  --area-mode idl-exact \
  --paths-parquet "Outputs/Filaments/Paths Filamentless 20180101-20181231.parquet"
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

## Notes
- The filament baseline uses NumPy/SciPy and does not require TensorFlow or
  scikit-learn.
- `Library/Config.py` auto-selects the host section from `Machine.json` and exposes `paths`, `apply_config`, and `train_batch_size` to scripts.
- Run `python -m Scripts` to list available commands.
