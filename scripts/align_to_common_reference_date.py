#!/usr/bin/env python3
"""Convert a series of OPERA DISP-S1 products to a single-reference stack.

The OPERA L3 InSAR displacement netCDF files have reference dates which
move forward in time. Each displacement is relative between two SAR acquisition dates.

This script converts these files into a single continuous displacement time series.
The current format is a stack of geotiff rasters.

Usage:
    python align_to_common_reference_date.py single-reference-out/ OPERA_L3_DISP-S1_*.nc
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from dolphin import io
from dolphin.timeseries import get_incidence_matrix
from dolphin.utils import flatten, format_dates, numpy_to_gdal_type
from osgeo import gdal
from pydantic import BaseModel
from tqdm.auto import tqdm

from disp_s1.product_info import DISPLACEMENT_PRODUCTS, ProductInfo


@click.group
def app():
    """Command line tool for running both conversion scripts."""
    pass


class OperaDispFile(BaseModel):
    """Class for information from one DISP-S1 production filename."""

    frame_id: int
    reference_dt: datetime
    secondary_dt: datetime
    version: str
    generation_dt: datetime

    @classmethod
    def from_filename(cls, name: Path | str) -> "OperaDispFile":
        """Create a OperaDispFile from a filename."""
        pattern = re.compile(
            r"OPERA_L3_DISP-S1_IW_F(?P<frame_id>\d{5})_VV_"
            r"(?P<reference_dt>\d{8}T\d{6}Z)_"
            r"(?P<secondary_dt>\d{8}T\d{6}Z)_"
            r"v(?P<version>[\d.]+)_"
            r"(?P<generation_dt>\d{8}T\d{6}Z)"
        )

        if not (match := pattern.match(Path(name).name)):
            raise ValueError(f"Invalid filename format: {name}")

        data = match.groupdict()
        data["reference_dt"] = datetime.fromisoformat(data["reference_dt"])
        data["secondary_dt"] = datetime.fromisoformat(data["secondary_dt"])
        data["generation_dt"] = datetime.fromisoformat(data["generation_dt"])

        return cls(**data)


def _make_gtiff_writer(output_dir, all_dates, like_filename, dataset: str):
    ref_date = all_dates[0]
    suffix = ".tif"

    out_paths = [
        Path(output_dir) / f"{dataset}_{format_dates(ref_date, d)}{suffix}"
        for d in all_dates[1:]
    ]
    output_dir.mkdir(exist_ok=True, parents=True)
    return io.BackgroundStackWriter(
        out_paths,
        like_filename=like_filename,
        # Using np.nan for the residuals, since it's not a valid phase
        nodata=np.nan,
    )


@app.command()
@click.argument("output_dir", type=Path)
@click.argument("nc_files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(["displacement", "short_wavelength_displacement"]),
    default="displacement",
)
@click.option("--block-shape", type=tuple[int, int], default=(256, 256))
@click.option("--nodata", type=float, default=np.nan)
def rereference(
    output_dir: Path,
    nc_files: list[str],
    dataset: str = "displacement",
    block_shape: tuple[int, int] = (256, 256),
    nodata: float = np.nan,
):
    """Create a single-reference stack from a list of OPERA displacement files.

    Parameters
    ----------
    output_dir : str
        File path to the output directory.
    nc_files : list[str]
        One or more netCDF files, each containing a 'displacement' dataset
        for a reference_date -> secondary_date interferogram.
    dataset : str
        Name of HDF5 dataset within product to convert.
    block_shape : tuple[int, int]
        Size of chunks of data to load at once.
        Default is (256, 256)
    nodata : float
        Value to use in translated rasters as nodata value.
        Default is np.nan

    """
    ifg_date_pairs = []
    for f in nc_files:
        odf = OperaDispFile.from_filename(f)
        ifg_date_pairs.append((odf.reference_dt, odf.secondary_dt))

    # Flatten all dates, find unique sorted list of SAR epochs
    all_dates = sorted(set(flatten(ifg_date_pairs)))

    # Build incidence matrix A and its pseudo-inverse
    A = get_incidence_matrix(ifg_date_pairs, all_dates)
    assert _is_full_column_rank(A)
    A_pinv = np.linalg.pinv(A)
    # Normally we have M interferograms, N = (len(all_dates) - 1) unknowns
    # Here the "inversion" is actually a trivial inversion, essentially a running sum,
    # so that M = N-1

    # open a GDAL dataset for the first file just to get the shape/geoinformation
    # All netCDF files for a frame are on the same grid.
    gdal_str = io.format_nc_filename(nc_files[0], dataset)
    ncols, nrows = io.get_raster_xysize(gdal_str)

    # Create the main displacement dataset.
    writer = _make_gtiff_writer(
        output_dir, all_dates=all_dates, like_filename=gdal_str, dataset=dataset
    )

    # Blockwise reading of each interferogram and accumulation into output
    # We'll define a block manager for reading data in 256x256 chunks
    block_iter = io.iter_blocks(arr_shape=(nrows, ncols), block_shape=block_shape)

    reader = io.HDF5StackReader.from_file_list(
        nc_files, dset_names=dataset, nodata=nodata
    )

    for row_slice, col_slice in tqdm(block_iter):
        # Read all 3D array of shape (M, block_rows, block_cols)
        block_ifg_data = reader[:, row_slice, col_slice]
        if isinstance(block_ifg_data, np.ma.MaskedArray):
            block_ifg_data = block_ifg_data.filled(0)
        M, cur_nrows, cur_ncols = block_ifg_data.shape

        # Now we combine them into date-wise displacement using the pseudo-inverse.
        # A_pinv has shape (N_out_dates, M_ifgs).
        # So for each pixel in the block, we do a dot product:
        #   displacement_for_all_dates = A_pinv.dot( [ifg_vals...] )
        # Reshape block_ifg_data to (M, block_size) so we can do one matmul
        npixels = cur_nrows * cur_ncols
        block_ifg_2d = block_ifg_data.reshape(M, npixels)  # (M, npixels)
        # (N, M) dot (M, npixels) -> (N, npixels)
        block_disp_2d = A_pinv @ block_ifg_2d
        # Reshape back to (n_out_dates, block_rows, block_cols)
        block_disp_3d = block_disp_2d.reshape(writer.shape[0], cur_nrows, cur_ncols)

        # Write block of data to output
        writer[:, row_slice, col_slice] = block_disp_3d

    writer.notify_finished()

    print(f"Saved displacement stack to {output_dir}")


def _is_full_column_rank(A):
    return np.linalg.matrix_rank(A) == A.shape[1]


QUALITY_LAYERS = [p.name for p in DISPLACEMENT_PRODUCTS]
# Remove the two that need to be inverted
QUALITY_LAYERS.pop(QUALITY_LAYERS.index("displacement"))
QUALITY_LAYERS.pop(QUALITY_LAYERS.index("short_wavelength_displacement"))
# For use in newer pythons, if we want to type the dataset arg:
# QUALITY_CHOICES = StrEnum(
#     "QualityLayer", [(value, auto()) for value in list(QUALITY_LAYERS)]
# )


@app.command()
@click.argument("output_dir", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("nc_files", nargs=-1, type=click.Path(exists=True))
@click.option("--dataset", type=click.Choice(QUALITY_LAYERS))
def translate(output_dir: Path, nc_files: list[Path | str], dataset):
    """Convert an auxiliary layer to the correct single-reference name."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for nc_path in tqdm(nc_files):
        p: ProductInfo = getattr(DISPLACEMENT_PRODUCTS, dataset)
        disp_file = OperaDispFile.from_filename(nc_path)
        date_str = format_dates(disp_file.reference_dt, disp_file.secondary_dt)
        out_tif = output_dir / f"{dataset}_{date_str}.tif"

        #  Use GDAL Translate to copy the recommended_mask dataset to GeoTIFF
        gdal.Translate(
            str(out_tif),
            io.format_nc_filename(nc_path, dataset),
            outputType=numpy_to_gdal_type(p.dtype),
            creationOptions=io.DEFAULT_TIFF_OPTIONS,
        )

        tqdm.write(f"Created {out_tif}")


if __name__ == "__main__":
    app()
