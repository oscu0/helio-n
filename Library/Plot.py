import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
import sunpy.visualization.colormaps.color_tables as ct
from astropy.visualization import AsinhStretch, ImageNormalize
from sunpy.coordinates import frames
import tempfile
import os
from PIL import Image


from .Processing import *
from .IO import (
    prepare_fits,
    prepare_pmap,
    prepare_mask,
    pmap_path,
    base_output_stem,
    plot_path,
    unet_mask_path,
)
from .Metrics import generate_omask


cmap = ct.aia_color_table(u.Quantity(193, "Angstrom"))


def px_to_pt(px: float) -> float:
    return px * 72.0 / DPI


def _setup_hgs_overlay(ax, m, grid_color="white"):
    ax.coords.grid(False)
    hgs_frame = frames.HeliographicStonyhurst()
    hgs = ax.get_coords_overlay(hgs_frame)
    hgs.grid(color=grid_color, alpha=0.4, linestyle="--", linewidth=px_to_pt(0.7))
    for c in hgs:
        c.set_ticks(spacing=10 * u.deg, color=grid_color, number=None)
        c.set_ticklabel_visible(False)
        c.set_axislabel("")
    return hgs


def plot_ch_map(
    row,
    source="unet",
    model=None,
    pmap=None,
    postprocessing="P0",
    oval=None,
    show_fits=True,
    multiplot_ax=None,
    set_title=False,
    map_obj=None,
    arch_id=None,
    date_id=None,
    base_fig_ax=None,
):
    if oval is None:
        oval = generate_omask(row)

    if source.lower() == "unet":
        smoothing_params = (
            postprocessing
            if isinstance(postprocessing, dict)
            else get_postprocessing_params(postprocessing)
        )
        if pmap is None:
            if model is None:
                raise ValueError("model required when pmap is None for source='unet'")
            pmap = find_or_make_pmap(row, model)
        mask = pmap_to_mask(pmap, smoothing_params)
        title = "helio-n (U-Net)"

    elif source.lower() == "idl":
        mask = prepare_mask(row.mask_path)
        title = "IDL"
    else:
        raise ValueError("source must be 'unet' or 'idl'.")

    mask = np.flipud(mask)
    oval = np.flipud(oval)

    fig, ax = plot_ch_base(
        row,
        mask,
        oval,
        show_fits=show_fits,
        multiplot_ax=multiplot_ax,
        set_title=set_title,
        map_obj=map_obj,
        title=title,
        arch_id=arch_id,
        date_id=date_id,
        base_fig_ax=base_fig_ax,
    )

    ax.contour(
        mask,
        levels=[0.5],
        colors="red",
        linewidths=px_to_pt(2),
        antialiased=True,
        transform=ax.get_transform("pixel"),
    )
    ax.contour(
        oval.astype(float),
        levels=[0.5],
        colors="yellow",
        linewidths=px_to_pt(3),
        antialiased=True,
        transform=ax.get_transform("pixel"),
    )

    return fig, ax


def plot_ch_base(
    row,
    mask,
    oval,
    show_fits=True,
    multiplot_ax=None,
    set_title=False,
    map_obj=None,
    title="",
    arch_id=None,
    date_id=None,
    base_fig_ax=None,
):
    """
    Draw the base CH plot (background + grid) without contours.
    Returns (fig, ax) ready for overlay contours.
    """
    FIGSIZE_IN = (TARGET_PX / DPI, TARGET_PX / DPI)

    m = map_obj if map_obj is not None else sunpy.map.Map(row.fits_path)

    if base_fig_ax is not None:
        fig, ax = base_fig_ax
        created_fig = False
    else:
        created_fig = False
        if multiplot_ax is None:
            fig = plt.figure(figsize=FIGSIZE_IN, dpi=DPI)
            fig.set_constrained_layout(False)
            ax = fig.add_axes([0, 0, 1, 1], projection=m)
            created_fig = True
        else:
            ax = multiplot_ax
            fig = ax.figure  # embed into existing figure

    if show_fits:
        m.plot(
            axes=ax,
            cmap=cmap,
            norm=ImageNormalize(stretch=AsinhStretch()),
            annotate=False,
        )
        grid_color = "white"
    else:
        ax.imshow(
            mask,
            origin="lower",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            transform=ax.get_transform("pixel"),
            interpolation="nearest",
        )
        grid_color = "lightgreen"

    if set_title:
        ax.set_title(title)
    else:
        ax.set_title("")
        if created_fig:
            fig.suptitle("")

    for c in ax.coords:
        c.set_ticks_visible(False)
        c.set_ticklabel_visible(False)
        c.set_axislabel("")

    ax.set_xlim(-0.5, TARGET_PX - 0.5)
    ax.set_ylim(-0.5, TARGET_PX - 0.5)
    ax.set_aspect("equal", adjustable="box")

    if created_fig:
        ax.set_position([0, 0, 1, 1])

    _setup_hgs_overlay(ax, m, grid_color=grid_color)
    return fig, ax


def save_ch_map_unet(
    row,
    model,
    postprocessing="P0",
    pmap=None,
    oval=None,
    map_obj=None,
    arch_id=None,
    date_id=None,
):
    fig, ax = plot_ch_map(
        row,
        source="unet",
        pmap=pmap,
        model=model,
        postprocessing=postprocessing,
        oval=oval,
        set_title=False,
        map_obj=map_obj,
        arch_id=arch_id,
        date_id=date_id,
    )

    architecture_id = arch_id if arch_id is not None else model.architecture_id
    date_range_id = date_id if date_id is not None else model.date_range_id

    base = base_output_stem(row.mask_path)
    out_path = plot_path(base, architecture_id, date_range_id, postprocessing)

    # Save the correct figure at exact size; no resizing step needed
    fig.savefig(out_path, dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def plot_ch_mask_only(
    row,
    source="unet",  # "unet" or "idl"
    model=None,  # required if source="unet" and no pmap_path/pmap provided
    pmap=None,  # optional precomputed pmap for source="unet"
    postprocessing="P0",  # used only for source="unet"
    ax=None,
):
    """
    Plot ONLY the CH mask, pixel-perfect:
    - no WCS grid
    - no FITS background
    - no contours
    - no padding
    - no interpolation

    Output pixels == mask pixels.
    """

    # --- obtain mask ---
    if source.lower() == "unet":
        smoothing_params = (
            postprocessing
            if isinstance(postprocessing, dict)
            else get_postprocessing_params(postprocessing)
        )

        if pmap is None:
            if model is None:
                raise ValueError("model required when pmap is None for source='unet'")
            pmap = find_or_make_pmap(row, model)

        mask = pmap_to_mask(pmap, smoothing_params)

    elif source.lower() == "idl":
        mask = prepare_mask(row.mask_path)

    else:
        raise ValueError("source must be 'unet' or 'idl'")

    # Match FITS/map orientation
    mask = np.flipud(mask)

    # --- figure / axes ---
    FIGSIZE_IN = (TARGET_PX / DPI, TARGET_PX / DPI)

    if ax is None:
        fig = plt.figure(figsize=FIGSIZE_IN, dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
    else:
        fig = ax.figure

    # --- draw mask ---
    ax.imshow(
        mask,
        origin="lower",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    # --- enforce exact pixel bounds ---
    ax.set_xlim(-0.5, TARGET_PX - 0.5)
    ax.set_ylim(-0.5, TARGET_PX - 0.5)
    ax.set_aspect("equal", adjustable="box")

    # --- remove everything else ---
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])

    return fig, ax


def save_ch_mask_only_unet(
    row,
    model,
    postprocessing="P0",
    pmap=None,
    fast=False,
    arch_id=None,
    date_id=None,
):
    base = base_output_stem(row.mask_path)
    out_path = unet_mask_path(
        base,
        arch_id if arch_id is not None else model.architecture_id,
        date_id if date_id is not None else model.date_range_id,
        postprocessing,
    )

    if fast:
        smoothing_params = (
            postprocessing
            if isinstance(postprocessing, dict)
            else get_postprocessing_params(postprocessing)
        )
        if pmap is None:
            pmap = find_or_make_pmap(row, model)
        mask = pmap_to_mask(pmap, smoothing_params)
        mask = np.flipud(mask)
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        img = PIL.Image.fromarray(mask_u8, mode="L")
        if img.size != (TARGET_PX, TARGET_PX):
            img = img.resize((TARGET_PX, TARGET_PX), resample=PIL.Image.NEAREST)
        img.save(out_path)
        return

    fig, ax = plot_ch_mask_only(
        row, source="unet", pmap=pmap, model=model, postprocessing=postprocessing
    )

    # Save the correct figure at exact size; no resizing step needed
    fig.savefig(out_path, dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)
