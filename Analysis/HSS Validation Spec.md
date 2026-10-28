# Annual HSS validation specification

Status: specification only; the new statistics are not implemented.

Validate the model's continuous solar-wind speed against cataloged events. Write annual aggregates to one Excel workbook. Do not construct a predicted HSS catalog, match predicted events, or use ICME flags in this version.

## Reference lists and model locations

| Worksheet | Reference file | Model evaluation location |
| --- | --- | --- |
| `Yermolaev checked - Earth` | `Outputs/HSS List/yermolaev_hss.csv` | Earth |
| `B1 - ACE` | `Outputs/HSS List/ace_hss_b1.csv` | Native ACE position |
| `B1 - STEREO-A` | `Outputs/HSS List/stereo-a_hss_b1.csv` | Native STEREO-A position |
| `B1 - STEREO-B` | `Outputs/HSS List/stereo-b_hss_b1.csv` | Native STEREO-B position |
| `B1 - PSP` | `Outputs/HSS List/psp_hss_b1.csv` | Native PSP position |
| `B1 - Solar Orbiter` | `Outputs/HSS List/solo_hss_b1.csv` | Native Solar Orbiter position |

Paths above are relative to `/Users/aosh/Developer/helio-n`. The catalogs are also available through `/Users/aosh/Developer/Data Links/HSS Lists`.

Earth must be an actual Earth-target model series. Do not substitute native ACE predictions or relabel the `ace` series as Earth. Historical `ace_earth` forecasts may be used only after verifying their target is Earth. The Earth and ACE worksheets remain distinct.

The Yermolaev list is generated in the last cell of `Analysis/HSS B1.ipynb`. Its source is `/Users/aosh/Developer/ICME_list/out/source/iki_sw_types.pkl`, written by `/Users/aosh/Developer/ICME_list/downloader/iki_list.py`. Join consecutive, hourly-contiguous `SWfast`/`CIR` rows into events, including CIR-only runs. Other classifications and temporal gaps break events. The cached interval source has no separate Boolean `checked` field; the worksheet name refers to the SRI RAS classification catalog.

For Yermolaev events, `v_max` and `T_max` come from Earth/OMNI speed inside the original, unpadded event interval, using `/Users/aosh/Developer/ICME_list/out/source/omni_v_dst.pkl`. No interpolation is added. Three events in the current generated list have no valid OMNI speed and retain blank peak fields. B1 reference peaks use the existing catalog `v_max` unchanged.

## Per-event evaluation

For a reference event with `T_start` and `T_end`, define the inclusive UTC evaluation window:

```text
window_start = T_start - 12 hours
window_end   = T_end   + 12 hours
```

Use hourly model `v_predict_raw`, before the ACE slow-wind correction. If native model output is finer than hourly, use hourly arithmetic means consistently. Do not interpolate missing model values to create coverage.

```text
v_pred_peak = maximum model speed within the padded window
t_pred_peak = earliest timestamp attaining that maximum
```

Use the same window for the exported slow-SW mask. An event has no CH contribution when that mask is true throughout the window. Do not infer this category by comparing speed with 300 km/s. When converting a finer-cadence mask to hourly values, an hour is slow only if all its native samples are slow; missing mask values remain unknown.

The entire padded window must have model speed and mask coverage to score an event. A partial window, missing forecast, or unknown mask makes it `unscored`, not a miss. Load adjacent CR/year data as needed rather than truncating the window at an export boundary. A missing mask field is an export prerequisite to resolve, not a reason to infer a mask from speed.

Each reference event is evaluated independently. Overlapping padded windows may use the same model samples; there is no one-to-one event or peak matching.

## Outcomes

Use three mutually exclusive scored outcomes, in this order:

| Outcome | Definition |
| --- | --- |
| `miss_no_ch` | Slow-SW mask is true throughout the padded window |
| `hit` | Some CH contribution is present and `v_pred_peak > 400` km/s |
| `miss` | Some CH contribution is present and `v_pred_peak <= 400` km/s |

`miss` in the workbook means the weak-prediction category; it does not include `miss_no_ch`. The combined miss count, if needed, is `miss + miss_no_ch`.

The 400 km/s threshold is strict: exactly 400 is a miss. There is no 450 km/s test, minimum predicted duration, relative-amplitude test, or additional reference peak-speed cutoff. A single hourly value above 400 qualifies as a hit. Keep every reference event, including weaker events and events with blank reference peaks.

## Underreporting

For every scored event with a valid reference `v_max`:

```text
underreporting_km_s = reference v_max - v_pred_peak
```

Positive values indicate underprediction; negative values indicate overprediction. Do not clip negative values, take absolute values, or restrict the mean to hits. Both miss categories contribute when their reference peak is available. The reference peak uses the original interval; only the model peak uses the 12-hour padding.

An event with no reference peak can still receive a hit/miss outcome, but contributes no underreporting value. Report the number of valid underreporting values separately.

## Annual aggregation

Assign each event to the UTC calendar year of its original `T_start`, before padding. An event crossing New Year is counted once, in its start year; evaluate its full window across the year boundary.

Select events by `T_start` in the requested half-open date range. Write one row per calendar year intersecting that range, including years with zero events. Partial years use only selected events and retain the requested period in the workbook filename. Pool events across CRs before computing annual statistics; do not average CR-level means or percentages.

Each worksheet has these columns:

| Column | Definition |
| --- | --- |
| `year` | UTC event-start year |
| `n_events` | Selected reference events |
| `n_scored` | Events with complete required model/mask coverage |
| `n_unscored` | `n_events - n_scored` |
| `hit` | Hit count |
| `miss` | Weak-prediction miss count |
| `miss_no_ch` | No-CH miss count |
| `hit_rate` | `hit / n_scored` |
| `miss_rate` | `miss / n_scored` |
| `miss_no_ch_rate` | `miss_no_ch / n_scored` |
| `n_underreporting` | Scored events with a valid reference peak |
| `mean_underreporting_km_s` | Arithmetic mean of signed per-event underreporting |

Counts are integers, rates are displayed as percentages, and mean underreporting is displayed to one decimal place. Preserve full numerical precision in the cells. Rates are blank when `n_scored = 0`; the mean is blank when `n_underreporting = 0`. The three scored rates sum to 100% before display rounding.

## Delivery and implementation boundary

Proposed output: `Outputs/SW/SW <start>-<end> (HSS Yearly).xlsx`, with the six worksheets listed above. This is a separate workbook; existing hourly, per-CR, and continuous-speed summary reports remain unchanged. Include only annual tables in this workbook. Keep source-path and method documentation in this specification rather than adding provenance columns or sidecars.

The intended archive-stats entry point is `Scripts/Make/SW_Stats.py` on `main`, with calculations in `Library/SW/Stats.py` and workbook writing alongside `Library/SW/Report.py`. These sources differ from the current `archive/aatr-2026-r0-0` checkout; implementation must use the active archive-stats code rather than transplanting the older branch's pipeline.

Before scoring, make the slow-SW mask available for Earth and every selected satellite. The inspected CR2306 `series.parquet` exports `slow_sw_patch_mask` for ACE only; CR2297 has no such field. Earth-target availability must likewise be verified. This specification does not authorize interpreting absent fields as false or generating substitute ACE-based Earth predictions.

Implementation checks should cover the 400 km/s boundary, all-slow versus mixed masks, padded-window boundaries, cross-year assignment, incomplete coverage, mean underreporting including misses/negative values, and preservation of the reference event counts. Existing `icme_flag` values are ignored completely: no exclusion, grouping, or weighting by ICME status.
