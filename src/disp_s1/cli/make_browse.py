from __future__ import annotations

import functools

import click

from disp_s1.browse_image import DEFAULT_CMAP
from disp_s1.product_info import DISPLACEMENT_PRODUCTS

# Always show defaults
click.option = functools.partial(click.option, show_default=True)


@click.command()
@click.option("-o", "--out-fname", help="Path to output png file")
@click.option("-i", "--in-fname", required=True, help="Path to input NetCDF file")
@click.option(
    "-n",
    "--dataset-name",
    default=DISPLACEMENT_PRODUCTS.short_wavelength_displacement.name,
    type=click.Choice(DISPLACEMENT_PRODUCTS.names),
    help="Name of dataset to plot from NetCDF file",
)
@click.option(
    "-m",
    "--max-img-dim",
    default=2048,
    help="Maximum dimension allowed for either length or width of browse image",
)
@click.option("--cmap", default=DEFAULT_CMAP, help="Colormap to use for the image")
@click.option(
    "--vmin", default=-0.15, type=float, help="Minimum value for color scaling"
)
@click.option(
    "--vmax", default=0.15, type=float, help="Maximum value for color scaling"
)
def make_browse(out_fname, in_fname, dataset_name, max_img_dim, cmap, vmin, vmax):
    """Create browse images for displacement products from command line."""
    import disp_s1.browse_image

    if out_fname is None:
        out_fname = in_fname.replace(".nc", f".{dataset_name}.png")

    disp_s1.browse_image.make_browse_image_from_nc(
        out_fname, in_fname, dataset_name, max_img_dim, cmap, vmin, vmax
    )
