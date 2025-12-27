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
from .IO import prepare_fits, prepare_pmap, prepare_mask
from .Metrics import generate_omask





cmap = ct.aia_color_table(u.Quantity(193, "Angstrom"))


def plot_ch_map(
    row,
    source="unet",          # "unet" or "idl"
    model=None,             # required if source="unet" and no pmap_path/pmap provided
    pmap=None,              # optional precomputed pmap for source="unet"
    postprocessing="P0",    # used only for source="unet"
    oval=None,
    show_fits=True,
    ax=None,
    set_title=False,
):
    # see Library/Config.py for TARGET_PX, DPI
    FIGSIZE_IN = (TARGET_PX / DPI, TARGET_PX / DPI)

    def _setup_hgs_overlay(ax, m, grid_color="white"):
        """
        Add a Heliographic Stonyhurst grid overlay with 10° spacing,
        no tick marks, no labels. Grid color is configurable.
        """
        # Disable default HPC grid
        ax.coords.grid(False)

        # HGS frame consistent with the map (observer + obstime)
        hgs_frame = frames.HeliographicStonyhurst(
            # observer=m.observer_coordinate,
            # obstime=m.date,
        )
        hgs = ax.get_coords_overlay(hgs_frame)

        # Grid lines
        hgs.grid(color=grid_color, alpha=0.4, linestyle="--", linewidth=px_to_pt(0.7))

        # Ticks only to drive grid placement; hide them visually
        for c in hgs:
            c.set_ticks(spacing=10 * u.deg, color=grid_color, number=None)
            c.set_ticklabel_visible(False)
            c.set_axislabel("")

        return hgs

    def px_to_pt(px: float) -> float:
        return px * 72.0 / DPI

    # FITS map (for WCS/projection)
    m = sunpy.map.Map(row.fits_path)

    # Oval
    if oval is None:
        oval = generate_omask(row)

    # Obtain mask depending on source
    if source.lower() == "unet":
        smoothing_params = get_postprocessing_params(postprocessing)

        if pmap is None:
            if getattr(row, "pmap_path", None):
                pmap = prepare_pmap(row.pmap_path)
            else:
                if model is None:
                    raise ValueError("plot_ch_map(source='unet') requires model if pmap is not provided.")
                pmap = fits_to_pmap(model, prepare_fits(row.fits_path))

        mask = pmap_to_mask(pmap, smoothing_params)
        title = "helio-n (U-Net)"

    elif source.lower() == "idl":
        mask = prepare_mask(row.mask_path)
        title = "IDL"

    else:
        raise ValueError("source must be 'unet' or 'idl'.")

    # Match map orientation
    mask = np.flipud(mask)
    oval = np.flipud(oval)

    # ---- Pixel-perfect figure/axes: NO subplots ----
    fig = plt.figure(figsize=FIGSIZE_IN, dpi=DPI)
    fig.set_constrained_layout(False)

    # Fill entire canvas; this eliminates margins at the figure level
    if ax is None:
        ax = fig.add_axes([0, 0, 1, 1], projection=m)

    # Draw base layer
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
        fig.suptitle("")

    # Hide WCS outside ticks/labels
    for c in ax.coords:
        c.set_ticks_visible(False)
        c.set_ticklabel_visible(False)
        c.set_axislabel("")

    # Hide any remaining matplotlib ticks
    # ax.tick_params(
    #     which="both",
    #     bottom=False, top=False, left=False, right=False,
    #     labelbottom=False, labeltop=False, labelleft=False, labelright=False,
    # )

    # IMPORTANT: lock axes to exact pixel bounds (prevents WCS padding/aspect games)
    # Note: pixel coordinates are [0..1023] with pixel centers at integers; -0.5..1023.5 fills exactly 1024 pixels.
    ax.set_xlim(-0.5, TARGET_PX - 0.5)
    ax.set_ylim(-0.5, TARGET_PX - 0.5)
    ax.set_aspect("equal", adjustable="box")
    # Kill axes padding
    ax.set_position([0, 0, 1, 1])  

    # Overlay grid (ensure your helper doesn't re-enable ticks/labels)
    _setup_hgs_overlay(ax, m, grid_color=grid_color)

    # Contours with linewidth specified in pixels
    ax.contour(
        mask,
        levels=[0.5],
        colors="red",
        linewidths=px_to_pt(2),   # 2 px
        antialiased=True,
        transform=ax.get_transform("pixel"),
    )
    ax.contour(
        oval.astype(float),
        levels=[0.5],
        colors="yellow",
        linewidths=px_to_pt(3),   # 3 px
        antialiased=True,
        transform=ax.get_transform("pixel"),
    )
    return fig, ax


def save_ch_map_unet(row, model, postprocessing="P0", pmap=None, oval=None):
    fig, ax = plot_ch_map(
        row,
        source="unet",
        pmap=pmap,
        model=model,
        postprocessing=postprocessing,
        oval=oval,
        set_title=False,
    )

    out_path = row.mask_path.replace(
        "CH_MASK",
        "CH_" + model.architecture_id + model.date_range_id + postprocessing,
    )
    print(out_path)

    # Save the correct figure at exact size; no resizing step needed
    fig.savefig(out_path, dpi=1024, bbox_inches=None, pad_inches=0)
    plt.close(fig)
