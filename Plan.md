# Solar-Wind Operational Plan

## Scope and order

The current objective is an efficient per-Carrington-rotation
forecast/hindcast archive:

0. establish a reproducible plausibility gate and test suite;
1. refactor ballistic propagation;
2. isolate and constrain missing-input interpolation;
3. refactor propagation outputs and archive ingestion;
4. separate animation export;
5. validate and run the per-CR archive workflow.

The future common boundary-condition interface for ballistic, HUX, HUXt, and
SURF is out of scope. For now, preserve explicit coordinates, units, model
identity, and time conventions so this work does not obstruct it.

## 0. Establish the refactor gate

Keep three distinct validation layers. Exact semantic and implementation tests
are hard pass/fail checks. Broad numerical comparisons are diagnostics because
the propagation scheme is intentionally changing. The paired frames are a
human scientific review: the script produces and opens them, but does not
pretend to automate that judgment.

Run the current gate with:

```bash
conda run -n icme3.12-metal python Analysis/SW_Propagation_Smoke.py
```

It uses the frozen 2018 reproduction Parquet and records its SHA-256, uses the
single configured 60-minute production output step,
runs the fast test suite, and writes two separate same-scale polar PNGs for
`2018-04-22 00:00 UTC`: a frozen legacy swept-propagation reference and the
production continuous radial-shell candidate. Missing cells are white. It
also writes the paired frames as compressed NPZ and a JSON report, then opens
both PNGs for visual comparison.

The first zero-width, point-launch prototype failed the visual gate despite
low common-cell error: its coverage was only `89.8%` versus `99.8%` for the
reference. This remains the motivating regression case. The continuous
source-segment implementation fills the frame without white stippling and is
the production candidate; final visual acceptance remains a human scientific
judgment.

The hard contract suite currently covers:

- the analytic arrival equation, centre-labelled time bins, and complete shell
  visitation by a `2000 km/s` parcel;
- reconstruction at the inner boundary;
- continuous compression and rarefaction segments, faster-only collisions,
  source-order invariance, and exact longitude time shifts;
- finite output speed bounds and frame-artifact round trips;
- native hourly input retention, exactly-twelve-hour source segments, no segment
  over a longer gap, and no leading or trailing extrapolation.

The smoke report records frame coverage, finite-mask overlap, common-cell
error, Earth-series correlation and lag, and both legacy and candidate errors
against ACE. Conservative thresholds raise review flags but do not fail the
run; exact agreement with the old scheme is neither expected nor desired.

Continue growing the production suite with later stages:

- keep `Tests/SW/reference.py` slow, small, and independent as the semantic
  oracle;
- add archive round-trip, rounded half-open CR concatenation, cross-CR context,
  arbitrary-range loading, and renderer-from-archive tests in steps 3 and 4;
- keep warm runtime, peak memory, archive size, and read speed in a benchmark
  report rather than brittle unit-test timing assertions.

## 1. Refactor ballistic propagation

### Continuous radial-shell arrivals

Treat the native boundary observations as knots of piecewise-linear speed
segments. For each requested radial shell, invert the arrival-time mapping

```text
A_r(t_launch) = t_launch + (r - r0) * R_sun / v(t_launch)
```

over every valid source segment. A segment can have at most one turning point,
so split there when necessary and invert each monotonic branch by bisection.
For every output bin centred on its timestamp, retain the fastest valid root
over `[centre - step/2, centre + step/2)`. This represents the continuously
evolving boundary without manufacturing a sub-hourly launch series. The
single 60-minute output cadence bounds temporal quantization to 30 minutes in
either direction. Centre labels lie on one global UTC hourly lattice; round
each astronomical CR boundary to the nearest output hour and assign each CR
the half-open interval between consecutive rounded boundaries. Adjacent
products then concatenate without gaps or duplicate timestamps.

Compute the `(radius, time)` plane once at zero longitude. The configured
longitudes are exact integer time shifts of that plane under the current grid,
so materialize the cube with shifted slices rather than resolving identical
characteristics independently for every longitude.

This removes:

- swept-cell deposition;
- radial leapfrog repair and its diagnostic;
- the propagation time-step and horizon loop;
- `h_step_idx`, per-seed radial paths, and
  `seed_r_idx[seed, horizon]`;
- fractional-delay speed interpolation through `v_prev`, `v_next`, and
  `phi_delay_alpha`;
- CR-specific accumulator state;
- the full-size post-processing copy.

The dense result is one NaN-initialized `float32` cube. Characteristic solving
scales primarily with source segments, radial shells, and intersected output
bins; longitude adds cheap shifted copies. Keep the implementation direct and
compiled, and optimize further only against measured archive-run costs.

### Remove CR-reset semantics

The per-cell CR reset was introduced in commit `c373b92` together with swept
deposition and a one-hour-wide angular packet. It made a new source block
replace an old one even when the new speed was slower. The identifiers were
run-relative blocks computed from the simulation start, not true Carrington
rotation numbers.

Parcel widening was removed in `8e5278e`, but the CR accumulator remained.
Under the chosen faster-only rule, parcels on opposite sides of a CR boundary
must compete normally. Remove `cr_flat`, `seed_cr_idx_arr`, `cr_steps`,
and the CR-specialized kernels.

### Propagation validation

- Check hand-calculated arrival times at several shells and longitudes.
- Compare analytic arrivals with very-fine-step point propagation on a small
  domain.
- Include parcels launched on opposite sides of a CR boundary that reach the
  same output bin.
- Shuffle seed order and require an identical cube.
- Record warm runtime and peak memory before and after the replacement.

## 2. Separate missing-input interpolation

Missing-input handling defines which continuous boundary segments exist;
propagation performs no separate missing-cell filling. Keep the native hourly
knots rather than materializing two-minute launch samples.

Use this rule:

- if valid bounding observations are separated by at most twelve elapsed hours,
  their linear boundary segment is valid in full;
- this includes the normal adjacent-hour segment;
- if separation exceeds twelve hours, create no segment across the gap;
- never partially fill the first or last twelve hours of an over-limit gap;
- do not extrapolate before the first or after the last observation.

Determine gaps from the original observed timestamps before reindexing or
dropping missing rows. Preserve a provenance/quality field distinguishing:

- `observed`;
- `adjacent_interp`;
- `short_gap_interp`;
- `missing`.

Keep native source rows and the availability mask in reproduction outputs.
Remove the current unconditional
`.interpolate(method="time").ffill().bfill()` behavior. Observed knots still
launch their own characteristics even when no continuous segment crosses an
over-limit gap.

Do not mechanically remove unrelated interpolation. Satellite coordinates and
external comparison products such as ENLIL are separate concerns.

> **OPEN QUESTION — upstream SQL filling:** the current query obtains CH area
> through `sdo_fill_sw_193(...)`. Verify exactly what that database function
> fills and whether a raw/unfilled source must replace it. Removing pandas-side
> interpolation alone may not expose the original gaps.

Test one-hour, exactly-twelve-hour, over-twelve-hour, leading, and trailing gaps.

## 3. Refactor per-CR outputs and ingestion

Compute astronomical Carrington boundaries with SunPy, then round each one
mathematically to the nearest output hour. Canonical products cover half-open
rounded-core intervals:

```text
[rounded CR start, rounded next-CR start)
```

Per-CR propagation runs are independent. A new propagation run does not consume
the preceding cube or any active-parcel restart state.

When an analysis range crosses a boundary, the ingester reads the final time
snapshot of CR `n` as the initial continuity context for CR `n+1`. This is
an ingestion/analysis operation only and does not modify or seed propagation.
The final snapshot already lives in the preceding dense cube; no separate
restart-parcel product is required.

A roughly 40-day analysis or movie interval is assembled from adjacent
rounded-core CR products. Do not store overlapping padded cubes.

### Provisional archive

```text
CR####/
    manifest.json
    inputs.parquet
    prepared_inputs.parquet
    series.parquet
    cube.h5
index.parquet
```

`manifest.json` records:

- rounded ownership bounds, astronomical CR bounds, and forecast/hindcast mode;
- model and schema versions;
- time-bin convention and all coordinates with units;
- configuration and code revision;
- input checksums and source extraction time;
- interpolation policy and availability statistics;
- neighboring CR identities.

`inputs.parquet` retains normalized source rows, including missing rows and
their availability status. `prepared_inputs.parquet` retains the rows and
empirical velocities actually used for propagation. SQL's upstream fill
provenance remains the open question in step 2.

`series.parquet` contains rounded-core tabular products, including standard
satellite predictions and comparison data. Adjacent files must concatenate
without duplicated boundary timestamps.

`cube.h5` provisionally stores the logical dense
`speed[time, phi, r]` cube as chunked compressed `float32`, alongside a
separate Boolean `is_slow_wind` cube. Before locking
the format and chunk shape, compare HDF5 and Zarr on one representative CR for
size, write time, arbitrary-window reads, animation reads, and extraction along
a new satellite trajectory. The dense cube should not be flattened into a
pandas table.

Movies are derived convenience products and never inputs to propagation.

### Ingestion interface

Provide:

- `load_series(start, end) -> DataFrame`, concatenating the required rounded
  per-CR Parquets and slicing once;
- `load_cube(start, end)`, reading only the required dense chunks and
  returning an array with named coordinates.

Both loaders must handle ranges within one CR and across multiple boundaries.
They must preserve missing values and expose input-quality provenance.

## 4. Separate animation export

Replace the current combined `make sw` entry point with two explicit commands:
`make propagate_sw` writes the propagation archive and exits, while
`make_animation` renders from archived products. It accepts either a CR number
for a core movie padded by default by seven days on both sides, or two UTC
timestamps for an arbitrary half-open range. Every frame labels its owning CR.
The animation command:

1. resolves the requested CR or date range and requests it from the archive loaders;
2. reads the required dense frames and tabular comparison data;
3. renders without invoking propagation;
4. optionally writes the movie into the relevant archive product directory.

The same ingestion path supports live DataFrame analysis and later sampling of
a spacecraft that was not selected during propagation.

## 5. Operational validation

Before a large archive run:

- propagate and archive one known CR;
- reload it and reproduce its standard satellite series;
- ingest a range crossing two CRs and verify the preceding final snapshot is
  used as intended;
- require unique, gap-free canonical timestamps apart from explicitly missing
  source/model values;
- extract a previously unplanned satellite path from the archived cube;
- render an existing movie interval from archive products alone;
- record propagation time, peak memory, compressed size, and representative
  read speed.

Then add the sequential per-CR driver, with clear reporting of completed,
missing, or invalid products so interrupted archive work can resume without
rerunning valid rotations.

## Expected code scope

- `Library/SW/Ballistic.py`: continuous shell-arrival/max-only kernel.
- `Library/SW/Coords.py`: physical coordinates only; remove temporal transport
  and CR accumulator machinery.
- `Library/SW/Inputs.py`: native boundary knots and bounded-segment policy;
  add archive provenance when the archive schema lands.
- `Library/SW/Archive.py`: add when producer, ingester, and renderer genuinely
  share archive logic.
- `Scripts/Make/Propagate_SW.py` (`make propagate_sw`): replaces `make sw`;
  propagation and archive production only.
- `Scripts/Make/Make_Animation.py` (`make make_animation`): independent
  padded-CR and arbitrary-range rendering from the archive.
- `Library/SW/Visualization.py`: rendering only, with no propagation or
  archive discovery.
- `Analysis/SW_Resolution_Snapshot.py`: retain old propagation methods only
  for controlled validation, then remove them from the production path.
