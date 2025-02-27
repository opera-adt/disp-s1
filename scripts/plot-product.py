#!/usr/bin/env python
# /// script
# dependencies = ["cartopy","cmap", "matplotlib", "numpy", "rioxarray", "tyro"]
# ///
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import tyro
from cartopy.io import img_tiles
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cmap import Colormap
from mpl_toolkits.axes_grid1 import AxesGrid


def plot(
    filename: Path | str,
    *,
    output_name: Path | str | None = None,
    dset_name: str = "displacement",
    subsample: int = 4,
    apply_recommended_mask: bool = True,
    pad_pct: float = 0.3,
    zoom_level: int = 8,
    cmap: str = "vik",
    vm: float = 0.1,
    cbar_label: str | None = None,
    figsize: tuple[float, float] | None = (9, 4),
    cartopy_crs_name: str = "PlateCarree",
    tick_resolution: float = 1.0,
    interpolation: str | None = "nearest",
) -> None:
    """Plot a DISP product image on top of background map tiles.

    Parameters
    ----------
    filename : Path | str
        Name of DISP product to plot
    output_name : Path | str | None, optional
        Name of the output file to save the plot
    dset_name : str, default "displacement"
        Name of the dataset to plot
    subsample : int, default 4
        Subsampling factor for the dataset
    apply_recommended_mask : bool, default True
        Whether to apply the recommended mask to the dataset
    pad_pct : float, default 0.3
        Percentage to pad the bounding box.
    zoom_level : int, default 8
        Zoom level for the background tiles.
    cmap : str, default "vik"
        Colormap to use for the plot
    vm : float, default 0.1
        Maximum absolute value for the colorbar range
    cbar_label : str, optional
        Label for the colorbar.
    figsize : tuple of float, optional
        Figure size (width, height) in inches.
    cartopy_crs_name : str, default "PlateCarree"
        Name of the Cartopy coordinate reference system to use.
    tick_resolution : float, default 1.0
        Resolution of the ticks.
    interpolation : str | None, default "nearest"
        Interpolation method to use for imshow

    """
    if output_name is None:
        output_name = Path(filename).with_suffix(".plot.pdf")

    cmap = Colormap(cmap).to_mpl()

    ds = rioxarray.open_rasterio(filename).sel(band=1)
    da = getattr(ds, dset_name)
    if subsample > 1:
        da = da[::subsample, ::subsample]
    if apply_recommended_mask:
        da = da.where(ds["recommended_mask"][::subsample, ::subsample])
    da_4326 = da.rio.reproject("EPSG:4326")

    tiler = img_tiles.GoogleTiles(style="satellite")
    crs = getattr(ccrs, cartopy_crs_name)()

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, {"projection": projection})

    fig = plt.figure(figsize=figsize)
    # https://scitools.org.uk/cartopy/docs/latest/gallery/miscellanea/axes_grid_basic.html
    axgr = AxesGrid(
        fig,
        111,
        axes_class=axes_class,
        nrows_ncols=(1, 1),
        axes_pad=0.6,
        cbar_location="right",
        cbar_mode="single",
        cbar_pad=0.1,
        cbar_size="3%",
        label_mode="keep",
    )
    ax = axgr[0]
    cbar_loc = axgr.cbar_axes[0]

    bbox = da_4326.rio.bounds()

    # matplotlib wants `extent`, different than gdal/rasterio convention
    pad_pct = pad_pct or 0.0
    extent = _padded_extent(bbox, pad_pct)
    ax.set_extent(extent, crs=crs)

    ax.add_image(tiler, zoom_level)
    extent_img = _padded_extent(bbox, 0.0)
    axim = ax.imshow(
        np.asarray(da_4326 - da_4326.mean()),
        transform=crs,
        extent=extent_img,
        origin="upper",
        zorder=2,
        interpolation=interpolation,
        vmax=vm,
        vmin=-vm,
        cmap=cmap,
        # **imshow_kwargs,
    )
    cbar = cbar_loc.colorbar(axim)
    tick_values = np.linspace(-vm, vm, 5)

    # Create the colorbar with specified ticks
    cbar = cbar_loc.colorbar(axim, ticks=tick_values)

    cbar_label = cbar_label or da.attrs["units"]
    cbar.set_label(cbar_label)

    _add_ticks(ax, resolution=tick_resolution)
    fig.tight_layout()
    fig.savefig(output_name, dpi=250)


def _generate_ticks(bounds, resolution, offset=0):
    """Create ticks for a cartopy axis.

    Parameters
    ----------
    bounds : tuple of float
        The bounds of the raster image in the form (left, bottom, right, top).
    resolution : float
        The spacing/rounding resolution for the ticks.
    offset : float, optional
        The offset to be added to the tick positions, by default 0.

    Returns
    -------
    xticks : numpy.ndarray
        The generated xticks adjusted to the specified resolution and offset.
    yticks : numpy.ndarray
        The generated yticks adjusted to the specified resolution and offset.

    """

    def _snap_bounds_to_res(bounds, resolution):
        left, bottom, right, top = bounds
        # Adjust the extents
        new_left = np.ceil(left / resolution) * resolution
        new_right = np.floor(right / resolution) * resolution
        new_bottom = np.ceil(bottom / resolution) * resolution
        new_top = np.floor(top / resolution) * resolution

        return [new_left, new_bottom, new_right, new_top]

    left, bottom, right, top = _snap_bounds_to_res(bounds, resolution)

    # Generate xticks from left to right bounds
    xticks = np.arange(left, right + resolution, resolution) + offset
    # Filter xticks to be within the bounds
    xticks = xticks[(xticks >= left) & (xticks <= right)]

    # Generate yticks from bottom to top bounds
    yticks = np.arange(bottom, top + resolution, resolution) + offset
    # Filter yticks to be within the bounds
    yticks = yticks[(yticks >= bottom) & (yticks <= top)]

    return xticks, yticks


def _padded_extent(bbox, pad_pct):
    """Return a padded extent, given a bbox and a percentage of padding."""
    left, bot, right, top = bbox
    padx = pad_pct * (right - left) / 2
    pady = pad_pct * (top - bot) / 2
    return (left - padx, right + padx, bot - pady, top + pady)


def _add_ticks(ax, resolution: float = 1, projection=ccrs.PlateCarree()):
    left, right, bot, top = ax.get_extent(projection)
    bounds = (left, bot, right, top)
    lon_ticks, lat_ticks = _generate_ticks(bounds, resolution=resolution)
    ax.set_xticks(lon_ticks, crs=projection)
    ax.set_yticks(lat_ticks, crs=projection)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()


if __name__ == "__main__":
    tyro.cli(plot)
