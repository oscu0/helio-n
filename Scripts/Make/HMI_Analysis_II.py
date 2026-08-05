#!/usr/bin/env python3
"""Generate a portable HTML review report for K10 filament contacts."""

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import ndimage
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from Library.Config import paths
from Library.Filaments import (
    HMI_STRONG_FIELD_G,
    compute_hmi_input,
    load_kislovodsk_catalog,
    rasterize_catalog_segments,
    select_catalog_segments,
)
from Library.IO import prepare_fits, prepare_mask


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Analyze HMI polarity distributions for every region on K10 "
            "Kislovodsk-contact frames and write a portable HTML report."
        ),
    )
    parser.add_argument(
        "--contacts-parquet",
        type=Path,
        default=(
            ROOT_DIR
            / "Outputs"
            / "Filaments"
            / "Overlap Scan 20180101-20181231.parquet"
        ),
    )
    parser.add_argument(
        "--paths-parquet",
        type=Path,
        default=Path(paths["artifact_root"]) / "Paths.parquet",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT_DIR / "Data" / "Kislovodsk Filaments.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT_DIR
            / "Outputs"
            / "Filaments"
            / "HMI Analysis II K10 2018"
        ),
    )
    parser.add_argument("--catalog-window-hours", type=float, default=12.0)
    parser.add_argument("--hmi-display-limit-g", type=float, default=100.0)
    parser.add_argument("--histogram-limit-g", type=float, default=150.0)
    parser.add_argument("--histogram-bin-g", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    assert args.catalog_window_hours > 0.0
    assert args.hmi_display_limit_g > 0.0
    assert args.histogram_limit_g > 0.0
    assert args.histogram_bin_g > 0.0
    assert args.max_frames is None or args.max_frames > 0
    return args


def summarize_hmi_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "hmi_mean_G": np.nan,
            "hmi_std_G": np.nan,
            "third_central_moment_G3": np.nan,
            "skewness": np.nan,
            "positive_flux_G": 0.0,
            "negative_flux_G": 0.0,
            "polarity_imbalance": np.nan,
            "polarity_balance": np.nan,
        }

    mean = float(values.mean())
    std = float(values.std())
    centered = values - mean
    third_moment = float(np.mean(centered**3)) if values.size >= 3 else np.nan
    skewness = third_moment / std**3 if values.size >= 3 and std > 0 else np.nan
    positive_flux = float(values[values > 0].sum())
    negative_flux = float(-values[values < 0].sum())
    unsigned_flux = positive_flux + negative_flux
    imbalance = (
        abs(positive_flux - negative_flux) / unsigned_flux
        if unsigned_flux > 0
        else np.nan
    )
    return {
        "hmi_mean_G": mean,
        "hmi_std_G": std,
        "third_central_moment_G3": third_moment,
        "skewness": skewness,
        "positive_flux_G": positive_flux,
        "negative_flux_G": negative_flux,
        "polarity_imbalance": imbalance,
        "polarity_balance": 1.0 - imbalance,
    }


def analyze_frame(
    frame_key,
    observation,
    contact_ids,
    kislovodsk_catalog,
    catalog_window_hours,
):
    observation_dt = pd.to_datetime(frame_key, format="%Y%m%d_%H%M")
    aia_map, aia193 = prepare_fits(observation.fits_path)
    candidate_mask = prepare_mask(observation.mask_path).astype(bool)
    labels, component_count = ndimage.label(
        candidate_mask,
        structure=np.ones((3, 3), dtype=int),
    )
    assert candidate_mask.shape == aia193.shape == aia_map.data.shape

    hmi_los, hmi_radial, hmi_valid = compute_hmi_input(
        aia_map,
        observation.hmi_path,
    )
    filaments = select_catalog_segments(
        kislovodsk_catalog,
        observation_dt,
        catalog_window_hours,
    )
    filament_mask, rasterized_segments, projected_segments = (
        rasterize_catalog_segments(
            aia_map,
            filaments,
            return_projected_segments=True,
        )
    )
    assert contact_ids.issubset(set(range(1, component_count + 1)))

    rows = []
    distributions = {}
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        valid_values = hmi_radial[component & hmi_valid]
        strong = (
            component
            & hmi_valid
            & (np.abs(hmi_los) >= HMI_STRONG_FIELD_G)
        )
        strong_values = hmi_radial[strong]
        distributions[component_id] = {
            "valid": valid_values,
            "strong": strong_values,
        }
        centroid_row, centroid_column = ndimage.center_of_mass(component)
        rows.append(
            {
                "frame_key": frame_key,
                "observation_dt": observation_dt,
                "component_id": component_id,
                "k10_contact": component_id in contact_ids,
                "area_px": int(component.sum()),
                "centroid_row": float(centroid_row),
                "centroid_column": float(centroid_column),
                "hmi_valid_px": int(valid_values.size),
                "hmi_strong_px": int(strong_values.size),
                "hmi_strong_fraction": (
                    float(strong_values.size / valid_values.size)
                    if valid_values.size
                    else np.nan
                ),
                **summarize_hmi_distribution(strong_values),
            }
        )

    return {
        "frame_key": frame_key,
        "observation_dt": observation_dt,
        "aia193": aia193,
        "labels": labels,
        "component_count": component_count,
        "hmi_radial": hmi_radial,
        "hmi_valid": hmi_valid,
        "filaments": filaments,
        "filament_mask": filament_mask,
        "rasterized_segments": rasterized_segments,
        "projected_segments": projected_segments,
        "contact_ids": contact_ids,
        "metrics": pd.DataFrame(rows),
        "distributions": distributions,
    }


def plot_frame_overview(frame, output_path, hmi_display_limit_g):
    height = frame["labels"].shape[0]
    labels_fits = np.flipud(frame["labels"])
    aia193_fits = np.flipud(frame["aia193"])
    hmi_fits = np.flipud(frame["hmi_radial"])
    hmi_valid_fits = np.flipud(frame["hmi_valid"])
    segments_fits = frame["projected_segments"].copy()
    segments_fits[:, [1, 3]] = height - 1 - segments_fits[:, [1, 3]]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6.5),
        layout="constrained",
    )
    axes[0].imshow(aia193_fits, cmap="sdoaia193", origin="lower")
    hmi_display = np.ma.array(hmi_fits, mask=~hmi_valid_fits)
    hmi_cmap = plt.get_cmap("RdBu_r").copy()
    hmi_cmap.set_bad("black")
    hmi_image = axes[1].imshow(
        hmi_display,
        cmap=hmi_cmap,
        norm=TwoSlopeNorm(
            vmin=-hmi_display_limit_g,
            vcenter=0.0,
            vmax=hmi_display_limit_g,
        ),
        origin="lower",
    )

    for component_id in range(1, frame["component_count"] + 1):
        component = labels_fits == component_id
        is_contact = component_id in frame["contact_ids"]
        color = "magenta" if is_contact else "cyan"
        linewidth = 2.0 if is_contact else 1.1
        centroid_row, centroid_column = ndimage.center_of_mass(component)
        for axis in axes:
            axis.contour(
                component,
                levels=[0.5],
                colors=[color],
                linewidths=linewidth,
            )
            axis.text(
                centroid_column,
                centroid_row,
                f"R{component_id}",
                color="white",
                ha="center",
                va="center",
                fontsize=8,
                bbox={
                    "facecolor": "black",
                    "edgecolor": color,
                    "alpha": 0.7,
                },
            )

    for x1, y1, x2, y2 in segments_fits:
        for axis in axes:
            axis.plot(
                [x1, x2],
                [y1, y2],
                color="lime",
                linewidth=1.5,
            )

    axes[0].set_title(f"AIA 193 Å — {frame['frame_key']}")
    axes[1].set_title(
        f"Reprojected HMI $B_r$ (blue −, red +; ±{hmi_display_limit_g:g} G)"
    )
    for axis in axes:
        axis.set_xlabel("FITS x pixel")
        axis.set_ylabel("FITS y pixel")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="lime", label="Kislovodsk centreline"),
            Line2D([0], [0], color="magenta", label="K10 contact region"),
            Line2D([0], [0], color="cyan", label="Other segmented region"),
        ],
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
        framealpha=0.88,
    )
    figure.colorbar(hmi_image, ax=axes[1], label="$B_r$ [G]", shrink=0.82)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_frame_histograms(
    frame,
    output_path,
    histogram_limit_g,
    histogram_bin_g,
):
    region_metrics = frame["metrics"].set_index("component_id")
    component_ids = list(region_metrics.index)
    n_columns = 3
    n_rows = int(np.ceil(len(component_ids) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(14, 3.5 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    bins = np.arange(
        -histogram_limit_g,
        histogram_limit_g + histogram_bin_g,
        histogram_bin_g,
    )
    for axis, component_id in zip(axes.flat, component_ids):
        metric = region_metrics.loc[component_id]
        valid_values = frame["distributions"][component_id]["valid"]
        strong_values = frame["distributions"][component_id]["strong"]
        valid_values = valid_values[np.abs(valid_values) <= histogram_limit_g]
        strong_values = strong_values[np.abs(strong_values) <= histogram_limit_g]
        axis.hist(
            valid_values,
            bins=bins,
            histtype="step",
            color="0.4",
            linewidth=1.0,
            label="all valid",
        )
        axis.hist(
            strong_values[strong_values < 0],
            bins=bins,
            color="#4477aa",
            alpha=0.7,
            label="strong −",
        )
        axis.hist(
            strong_values[strong_values > 0],
            bins=bins,
            color="#cc6677",
            alpha=0.7,
            label="strong +",
        )
        contact_label = "K10" if metric["k10_contact"] else "other"
        axis.set_title(
            f"R{component_id} [{contact_label}] | "
            f"balance={metric['polarity_balance']:.2f}, "
            f"μ₃={metric['third_central_moment_G3']:.2g}, "
            f"skew={metric['skewness']:.2f}"
        )
        axis.set_yscale("log")
        axis.set_xlabel("$B_r$ [G]")
        axis.set_ylabel("pixels")
        axis.grid(alpha=0.2)

    for axis in axes.flat[len(component_ids) :]:
        axis.remove()
    axes.flat[0].legend()
    figure.suptitle(
        f"Region HMI distributions — {frame['frame_key']}; "
        f"strong selection: |B_LOS| ≥ {HMI_STRONG_FIELD_G:g} G; "
        f"display limited to ±{histogram_limit_g:g} G"
    )
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_summary(metrics, output_path):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        layout="constrained",
    )
    groups = (
        (False, "Other regions on K10 frames", "0.55", "o"),
        (True, "K10 contact regions", "magenta", "X"),
    )
    for is_contact, label, color, marker in groups:
        selected = metrics.loc[metrics["k10_contact"] == is_contact]
        axes[0].scatter(
            selected["polarity_balance"],
            selected["skewness"],
            s=30 if is_contact else 18,
            marker=marker,
            facecolors=color if is_contact else "none",
            edgecolors=color,
            alpha=0.85 if is_contact else 0.5,
            label=label,
        )
        axes[1].hist(
            selected["polarity_balance"].dropna(),
            bins=np.linspace(0, 1, 21),
            histtype="step",
            linewidth=2 if is_contact else 1.5,
            color=color,
            label=label,
        )

    axes[0].axhline(0, color="0.75", linewidth=0.8)
    axes[0].set(
        xlabel="Polarity balance",
        ylabel="Standardized third moment",
        title="Region polarity balance and skewness",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].set(
        xlabel="Polarity balance",
        ylabel="Regions",
        title="Polarity-balance distributions",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def format_metric(value, digits=3):
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def metrics_table_html(metrics):
    columns = [
        "component_id",
        "k10_contact",
        "area_px",
        "hmi_valid_px",
        "hmi_strong_px",
        "hmi_strong_fraction",
        "polarity_balance",
        "polarity_imbalance",
        "third_central_moment_G3",
        "skewness",
    ]
    table = metrics[columns].copy()
    table.columns = [
        "Region",
        "K10",
        "Area px",
        "Valid HMI px",
        "Strong HMI px",
        "Strong fraction",
        "Polarity balance",
        "Polarity imbalance",
        "Third moment G³",
        "Skewness",
    ]
    for column in (
        "Strong fraction",
        "Polarity balance",
        "Polarity imbalance",
        "Skewness",
    ):
        table[column] = table[column].map(lambda value: format_metric(value, 3))
    table["Third moment G³"] = table["Third moment G³"].map(
        lambda value: "—" if pd.isna(value) else f"{value:.4g}"
    )
    table["K10"] = table["K10"].map({True: "yes", False: "no"})
    return table.to_html(index=False, classes="metrics-table", border=0)


def write_report_html(
    metrics,
    coverage,
    case_records,
    output_path,
    parameters,
):
    contacts = metrics.loc[metrics["k10_contact"]]
    other_regions = metrics.loc[~metrics["k10_contact"]]
    contact_balance = contacts["polarity_balance"].median()
    other_balance = other_regions["polarity_balance"].median()
    contact_skewness = contacts["skewness"].median()
    included_frames = int(coverage["included"].sum())
    missing_frames = coverage.loc[
        coverage["reason"] == "missing HMI in current Paths tolerance",
        "frame_key",
    ].tolist()

    cards = []
    for record in case_records:
        frame_metrics = record["metrics"]
        contact_rows = frame_metrics.loc[frame_metrics["k10_contact"]]
        balance = contact_rows["polarity_balance"].median()
        skewness = contact_rows["skewness"].median()
        contact_components = ", ".join(
            f"R{component_id}"
            for component_id in contact_rows["component_id"].astype(int)
        )
        cards.append(
            "<article class='case-card' "
            f"data-frame='{html.escape(record['frame_key'])}' "
            f"data-balance='{balance if pd.notna(balance) else ''}'>"
            "<header>"
            f"<h3>{html.escape(record['frame_key'])}</h3>"
            "<div class='badges'>"
            f"<span>K10 {html.escape(contact_components)}</span>"
            f"<span>{len(frame_metrics)} regions</span>"
            f"<span>balance {format_metric(balance, 3)}</span>"
            f"<span>skew {format_metric(skewness, 3)}</span>"
            "</div></header>"
            f"<a href='{record['overview_path']}'><img "
            f"src='{record['overview_path']}' loading='lazy' "
            f"alt='AIA and HMI overview for {html.escape(record['frame_key'])}'></a>"
            "<details><summary>Region polarity histograms</summary>"
            f"<a href='{record['histogram_path']}'><img "
            f"src='{record['histogram_path']}' loading='lazy' "
            f"alt='HMI histograms for {html.escape(record['frame_key'])}'></a>"
            "</details>"
            "<details><summary>Exact region metrics</summary>"
            f"<div class='table-wrap'>{metrics_table_html(frame_metrics)}</div>"
            "</details></article>"
        )

    missing_cards = []
    for frame_key in missing_frames:
        missing_cards.append(
            "<article class='case-card missing' "
            f"data-frame='{html.escape(frame_key)}' data-balance=''>"
            f"<header><h3>{html.escape(frame_key)}</h3>"
            "<div class='badges'><span>missing HMI in Paths tolerance</span>"
            "</div></header><p>No HMI metrics or plots were generated; the "
            "K10 contact remains in the coverage table.</p></article>"
        )

    all_metrics_html = metrics_table_html(
        metrics.assign(
            component_id=(
                metrics["frame_key"]
                + " R"
                + metrics["component_id"].astype(str)
            )
        )
    )
    missing_text = ", ".join(missing_frames) if missing_frames else "none"
    report_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>HMI Analysis II — K10 filament contacts</title>
<style>
:root{{--bg:#f5f6f8;--surface:#fff;--ink:#1b1e23;--muted:#626a76;--line:#d9dde4;--accent:#a00088;--soft:#f0e8ef}}
@media(prefers-color-scheme:dark){{:root{{--bg:#15171a;--surface:#22252a;--ink:#f2f3f5;--muted:#b8bec8;--line:#3b4048;--accent:#ff67de;--soft:#372f38}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,sans-serif;line-height:1.45}}
.page{{max-width:1500px;margin:auto;padding:28px}} h1{{margin:0 0 8px}} h2{{margin-top:36px}} h3{{margin:0}} p{{max-width:1000px}}
.summary{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px;margin:18px 0}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}}
.stat{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px}} .stat strong{{display:block;font-size:1.55rem}}
.muted{{color:var(--muted)}} .hero{{width:100%;height:auto;background:white;border-radius:8px}}
.controls{{position:sticky;top:0;z-index:2;display:flex;gap:10px;flex-wrap:wrap;background:var(--bg);padding:12px 0}}
input,select{{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:9px 11px}}
input{{min-width:260px}} .gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}}
.case-card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0}}
.case-card header{{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;flex-wrap:wrap;margin-bottom:10px}}
.case-card img{{display:block;width:100%;height:auto;background:white;border-radius:6px}} .case-card.missing{{border-style:dashed}}
.badges{{display:flex;gap:6px;flex-wrap:wrap}} .badges span{{background:var(--soft);border-radius:999px;padding:3px 8px;font-size:.8rem}}
details{{margin-top:10px}} summary{{cursor:pointer;font-weight:600;padding:5px 0}} .table-wrap{{overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:.82rem}} th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
code{{background:var(--soft);padding:2px 5px;border-radius:4px}} .hidden{{display:none}}
@media(max-width:650px){{.page{{padding:16px}}.gallery{{grid-template-columns:1fr}}input{{min-width:100%}}}}
</style>
</head>
<body><main class="page">
<h1>HMI Analysis II — K10 filament contacts</h1>
<section class="summary"><h2>Technical summary</h2>
<p>The report applies the repository's existing HMI reprojection and radial-field preprocessing to every segmented region on K10-contact frames. K10 contacts are catalog-proximity candidates, not confirmed filaments; other regions on the same frames are shown as a matched comparison group.</p>
<p><strong>Observed medians:</strong> K10 polarity balance {format_metric(contact_balance, 3)} versus {format_metric(other_balance, 3)} for other regions, with K10 standardized third moment {format_metric(contact_skewness, 3)}. These are descriptive diagnostics only; the case images and distributions below are the primary evidence.</p></section>
<div class="stats">
<div class="stat"><strong>{len(coverage)}</strong><span>K10 contact frames</span></div>
<div class="stat"><strong>{included_frames}</strong><span>frames with HMI</span></div>
<div class="stat"><strong>{len(contacts)}</strong><span>K10 components analyzed</span></div>
<div class="stat"><strong>{len(metrics)}</strong><span>all regions analyzed</span></div>
</div>
<section><h2>Population-level metric comparison</h2>
<p>K10 candidates are highlighted in magenta; open gray points and outlines represent other segmented regions on the same observations. Polarity balance approaches one for balanced positive/negative unsigned flux and zero for a unipolar distribution.</p>
<a href="summary.png"><img class="hero" src="summary.png" alt="Population comparison of polarity balance and skewness"></a></section>
<section><h2>Definitions and scope</h2>
<p>Strong pixels satisfy <code>|B_LOS| ≥ {HMI_STRONG_FIELD_G:g} G</code> after masked Gaussian smoothing. Metrics use corrected <code>B_r</code>: imbalance is <code>|ΣB_r| / Σ|B_r|</code>, balance is one minus imbalance, the third central moment is <code>E[(B_r−E[B_r])³]</code>, and skewness is that moment divided by <code>σ³</code>. Histogram displays are limited to ±{parameters['histogram_limit_g']:g} G, while metrics use the full strong-field values.</p>
<p class="muted">Missing HMI within the current Paths tolerance: {html.escape(missing_text)}.</p></section>
<section><h2>Frame-by-frame evidence</h2>
<p>Each overview uses FITS pixel coordinates with (0,0) at bottom left. Magenta contours mark saved K10 components, cyan contours mark other regions, and green lines are the nearest projected Kislovodsk annotations.</p>
<div class="controls"><input id="frame-search" type="search" placeholder="Filter frame key…" aria-label="Filter frame key"><select id="balance-filter" aria-label="Filter K10 polarity balance"><option value="all">All balances</option><option value="balanced">Balance ≥ 0.5</option><option value="unipolar">Balance &lt; 0.5</option><option value="missing">Missing HMI</option></select><span id="visible-count" class="muted"></span></div>
<div class="gallery" id="gallery">{''.join(cards)}{''.join(missing_cards)}</div></section>
<section><h2>All region metrics</h2><p>The table is included for exact lookup; the Parquet and CSV files in this folder contain the same rows with full numeric precision.</p><details><summary>Open complete metrics table</summary><div class="table-wrap">{all_metrics_html}</div></details></section>
<section><h2>Limitations and next check</h2><p>This analysis measures magnetic-distribution differences but does not establish filament identity. The next useful check is whether visually credible filament cases consistently show higher balance or a characteristic bimodal/skewed distribution without incorrectly rejecting ordinary coronal holes.</p></section>
</main>
<script>
const search=document.getElementById('frame-search');const select=document.getElementById('balance-filter');const cards=[...document.querySelectorAll('.case-card')];const count=document.getElementById('visible-count');
function applyFilters(){{const q=search.value.trim().toLowerCase();const mode=select.value;let visible=0;for(const card of cards){{const frame=card.dataset.frame.toLowerCase();const raw=card.dataset.balance;const balance=raw===''?null:Number(raw);const matchesText=frame.includes(q);const matchesMode=mode==='all'||(mode==='missing'&&balance===null)||(mode==='balanced'&&balance!==null&&balance>=0.5)||(mode==='unipolar'&&balance!==null&&balance<0.5);card.classList.toggle('hidden',!(matchesText&&matchesMode));if(matchesText&&matchesMode)visible++;}}count.textContent=`${{visible}} / ${{cards.length}} frames`;}}
search.addEventListener('input',applyFilters);select.addEventListener('change',applyFilters);applyFilters();
</script></body></html>"""
    output_path.write_text(report_html)


def main(argv=None):
    args = parse_args(argv)
    assert args.contacts_parquet.exists(), args.contacts_parquet
    assert args.paths_parquet.exists(), args.paths_parquet
    assert args.catalog.exists(), args.catalog

    contacts = pd.read_parquet(args.contacts_parquet)
    contacts = contacts.loc[
        contacts["kislovodsk_supported_centerline_px"] > 0
    ].copy()
    paths_df = pd.read_parquet(args.paths_parquet)
    catalog = load_kislovodsk_catalog(args.catalog)
    all_frame_keys = sorted(contacts["frame_key"].unique())
    assert set(all_frame_keys).issubset(paths_df.index)

    coverage = paths_df.loc[all_frame_keys, ["fits_path", "mask_path", "hmi_path"]].copy()
    coverage.insert(0, "frame_key", coverage.index)
    coverage["included"] = coverage[["fits_path", "mask_path", "hmi_path"]].notna().all(axis=1)
    coverage["reason"] = np.where(
        coverage["included"],
        "included",
        "missing HMI in current Paths tolerance",
    )
    frame_keys = coverage.loc[coverage["included"], "frame_key"].tolist()
    if args.max_frames is not None:
        frame_keys = frame_keys[: args.max_frames]
        coverage["included"] = coverage["frame_key"].isin(frame_keys)
        coverage["reason"] = np.where(
            coverage["included"],
            "included",
            np.where(
                coverage["hmi_path"].isna(),
                "missing HMI in current Paths tolerance",
                "excluded by --max-frames",
            ),
        )

    contact_ids = {
        frame_key: set(frame_rows["component_id"].astype(int))
        for frame_key, frame_rows in contacts.groupby("frame_key")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    case_records = []
    for frame_key in tqdm(frame_keys, desc="K10 HMI report"):
        frame = analyze_frame(
            frame_key,
            paths_df.loc[frame_key],
            contact_ids[frame_key],
            catalog,
            args.catalog_window_hours,
        )
        overview_name = f"cases/{frame_key}-overview.png"
        histogram_name = f"cases/{frame_key}-histograms.png"
        plot_frame_overview(
            frame,
            args.output_dir / overview_name,
            args.hmi_display_limit_g,
        )
        plot_frame_histograms(
            frame,
            args.output_dir / histogram_name,
            args.histogram_limit_g,
            args.histogram_bin_g,
        )
        metric_frames.append(frame["metrics"])
        case_records.append(
            {
                "frame_key": frame_key,
                "overview_path": overview_name,
                "histogram_path": histogram_name,
                "metrics": frame["metrics"],
            }
        )

    assert metric_frames, "No HMI-covered K10 frames selected."
    metrics = (
        pd.concat(metric_frames, ignore_index=True)
        .sort_values(["frame_key", "component_id"])
        .reset_index(drop=True)
    )
    metrics.to_parquet(args.output_dir / "HMI K10 Region Metrics.parquet", index=False)
    metrics.to_csv(args.output_dir / "HMI K10 Region Metrics.csv", index=False)
    coverage.to_csv(args.output_dir / "K10 HMI Coverage.csv", index=False)
    plot_summary(metrics, args.output_dir / "summary.png")

    parameters = {
        "contacts_parquet": str(args.contacts_parquet),
        "paths_parquet": str(args.paths_parquet),
        "catalog": str(args.catalog),
        "catalog_window_hours": args.catalog_window_hours,
        "hmi_strong_field_G": HMI_STRONG_FIELD_G,
        "hmi_display_limit_G": args.hmi_display_limit_g,
        "histogram_limit_g": args.histogram_limit_g,
        "histogram_bin_g": args.histogram_bin_g,
        "metric_definitions": {
            "polarity_imbalance": "abs(sum(B_r)) / sum(abs(B_r))",
            "polarity_balance": "1 - polarity_imbalance",
            "third_central_moment_G3": "mean((B_r - mean(B_r)) ** 3)",
            "skewness": "third_central_moment_G3 / std(B_r) ** 3",
        },
        "chart_map": {
            "summary.png": (
                "relationship scatter and comparison histogram for K10 contacts "
                "versus other regions on the same frames"
            ),
            "cases/*-overview.png": (
                "AIA 193 and reprojected HMI with Kislovodsk lines and region contours"
            ),
            "cases/*-histograms.png": (
                "per-region valid and strong-field B_r distributions"
            ),
        },
    }
    (args.output_dir / "Report Metadata.json").write_text(
        json.dumps(parameters, indent=2) + "\n"
    )
    write_report_html(
        metrics,
        coverage,
        case_records,
        args.output_dir / "index.html",
        parameters,
    )
    print(
        f"Saved report for {len(frame_keys)} HMI-covered frames, "
        f"{int(metrics['k10_contact'].sum())} K10 contacts, and "
        f"{len(metrics)} total regions to {args.output_dir}"
    )
    print(f"Open {args.output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
