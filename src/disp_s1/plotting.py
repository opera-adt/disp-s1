import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from dolphin._types import Filename


def plot_product(
    filename: Filename,
    downsample=3,
    mask_on_conncomp: bool = True,
    figsize: tuple[float, float] = (9, 6),
    unwrapped_phase_cmap: str = "RdBu",
    unwrapped_phase_limits: tuple[float, float] = (-20, 20),
):
    """Plot the raster layers from one DISP product.

    Parameters
    ----------
    filename : Filename
        Path to the DISP product.
    downsample : int, optional
        Downsample factor, by default 3.
    mask_on_conncomp : bool, optional
        Mask the data where connected component label = 0, by default True.
        Otherwise, Mask the data where the data value = 0/nan.
    figsize : tuple[float, float], optional
        Figure size, by default (9, 6).
    unwrapped_phase_cmap : str, optional
        Colormap for the unwrapped phase, by default "RdBu".
    unwrapped_phase_limits : tuple[float, float], optional
        Limits for the unwrapped phase colormap, by default (-20, 20).

    Returns
    -------
    fig, axes : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        Figure and axes objects.
    """
    ds = rioxarray.open_rasterio(filename).sel(band=1)

    dsets = [
        "unwrapped_phase",
        "connected_component_labels",
        "temporal_correlation",
        "interferometric_correlation",
    ]
    cmaps = [unwrapped_phase_cmap, "jet", "viridis", "plasma"]

    vms = [unwrapped_phase_limits, (0, None), (0, 1), (0, 1)]

    if mask_on_conncomp:
        bad_mask = ds["connected_component_labels"][::downsample, ::downsample] == 0

    fig, axes = plt.subplots(
        ncols=2, nrows=2, sharex=True, sharey=True, figsize=figsize
    )

    for ax, dset_name, cmap, vm in zip(axes.ravel(), dsets, cmaps, vms):
        dset = ds[dset_name][::downsample, ::downsample]
        if not mask_on_conncomp:
            bad_mask = dset == 0

        dset.where(~bad_mask, np.nan).plot.imshow(
            ax=ax,
            cmap=cmap,
            vmin=vm[0],
            vmax=vm[1],
            cbar_kwargs={"label": dset.attrs["units"]},
        )
        ax.set_title(dset.attrs["long_name"])
        ax.set_aspect("equal")

    fig.tight_layout()
    return fig, axes
