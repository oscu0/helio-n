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
  - `artifact_root`: where parquet/plots/models are written (per host)
  - `train_batch_size`, `apply_batch_size`, `chunk_size`, `max_inflight_plots`, `plot_threads`
- Optional: `inherits` lets a host clone another entry and override only a few paths.
- Global plot settings: `Config/Plot.json` (`target_px`, `dpi`).
- Paths/parquet outputs live under `Outputs/Artifacts/<hostname>/`.
- Models save to `Outputs/Models/<architecture><date_range>.keras`.

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
- Loads `Config/Model/Architecture/<architecture_id>.json` and injects `train_batch_size` from `Machine.json`.
- Date slice comes from `Config/Model/Date Range/<date_range_id>.json`.
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
- TF/XLA logging is suppressed in scripts; GPU growth is enabled in Apply.
- `Library/Config.py` auto-selects the host section from `Machine.json` and exposes `paths`, `apply_config`, and `train_batch_size` to scripts.
- Run `python -m Scripts` to list available commands.
