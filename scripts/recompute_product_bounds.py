#!/usr/bin/env python
"""Recompute the bounding polygon of an OPERA DISP-S1 product.

The bounding polygon stored in some DISP-S1 products does not match the actual
extent of the valid displacement data (see opera-sds nasa/disp-s1#376). This
script recomputes the footprint directly from the `displacement` layer:

  1. Build a validity mask of the displacement data (finite and non-zero).
  2. Write the mask to a temporary GeoTIFF in the product's CRS.
  3. Run `disp_s1._utils.extract_footprint` — the same function used to write the
     bounding polygon during forward product generation — to get the minimum
     rotated rectangle of the valid-data footprint (antimeridian-split).
  4. Write the result back into the product metadata:
       /identification/bounding_polygon  (WKT, degrees)

Using `extract_footprint` keeps this retroactive fix byte-for-byte consistent
with forward processing.

Example:
-------
    $ python scripts/recompute_product_bounds.py input.nc -o output.nc

"""

from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tyro
from dolphin import io
from pyproj import CRS

from disp_s1._utils import extract_footprint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DISPLACEMENT_DATASET = "displacement"
BOUNDING_POLYGON_PATH = "/identification/bounding_polygon"


def compute_bounding_polygon(input_file: Path) -> str:
    """Recompute the bounding polygon from a product's ``/displacement`` layer.

    Writes the valid-data mask to a temporary GeoTIFF and runs the shared
    ``disp_s1._utils.extract_footprint`` (the same call used in forward product
    generation), returning the minimum rotated rectangle of the footprint as an
    EPSG:4326 MULTIPOLYGON WKT.

    Notes
    -----
    The displacement is read with h5py rather than ``dolphin.io.load_gdal`` on
    purpose. Forward processing runs ``extract_footprint`` on the GeoTIFF timeseries,
    but this script reads ``/displacement`` back from the NetCDF.
    GDAL flips the NetCDF y-axis (gdal 3.12.3 reads it reversed relative
    to the stored /y axis; 3.12.2 does not) while leaving GeoTIFFs untouched, so
    a ``load_gdal`` read would disagree with forward output.
    Reading natively with h5py (CF ``(y, x)`` layout, where
    ``/displacement[i, j]`` maps to ``/y[i]``, ``/x[j]``) is what keeps the
    recompute consistent with forward production and independent of GDAL's
    NetCDF orientation behavior. The mask is then written to a GeoTIFF, which
    GDAL reads unambiguously, before calling ``extract_footprint``.

    """
    logger.info(f"Reading displacement from {input_file}")
    with h5py.File(input_file) as f:
        disp = f[f"/{DISPLACEMENT_DATASET}"][:]
        x = f["/x"][:]
        y = f["/y"][:]
        crs = CRS.from_wkt(f["/spatial_ref"].attrs["crs_wkt"])

    # Valid-data mask -> temp GeoTIFF -> shared extract_footprint. Same idea as
    # `disp_s1.main._create_nodata_mask` (nonzero -> write_arr); the array is read
    # with h5py and georeferenced from the native /x, /y axes (see Notes).
    mask = (np.isfinite(disp) & (disp != 0)).astype("uint8")
    if not mask.any():
        raise ValueError(f"{input_file}: no valid displacement pixels found")
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    geotransform = (float(x[0]) - dx / 2, dx, 0.0, float(y[0]) - dy / 2, 0.0, dy)

    with tempfile.TemporaryDirectory() as tmp_dir:
        mask_tif = Path(tmp_dir) / "valid_mask.tif"
        io.write_arr(
            arr=mask,
            output_name=mask_tif,
            geotransform=geotransform,
            projection=crs.to_wkt(),
            dtype="uint8",
            nodata=0,
        )
        return extract_footprint(mask_tif)


def update_metadata_timestamps(
    output_file: Path,
    processing_datetime: datetime | None = None,
    update_processing_time: bool = True,
    update_version: bool = False,
    new_version: str | None = None,
) -> None:
    """Update metadata timestamps in the corrected file.

    Parameters
    ----------
    output_file : Path
        Path to the output file to update.
    processing_datetime : datetime, optional
        Processing datetime to use. If None, uses current time.
    update_processing_time : bool
        Whether to update the processing_start_datetime field.
        Default = True
    update_version : bool
        Whether to update the product version.
    new_version : str, optional
        New version string. If None and update_version=True, appends ".1".

    """
    if processing_datetime is None:
        processing_datetime = datetime.now()

    with h5py.File(output_file, "a") as f:
        # Update processing_start_datetime
        if update_processing_time:
            old_datetime = f["/identification/processing_start_datetime"][()].decode(
                "utf-8"
            )
            logger.info(
                f"Updating processing_start_datetime from {old_datetime} to "
                f"{processing_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            proc_str = (processing_datetime.strftime("%Y-%m-%d %H:%M:%S")).encode(
                "utf-8"
            )
            f["/identification/processing_start_datetime"][()] = proc_str

        # Optionally update product version
        if update_version:
            if new_version is None:
                raise ValueError("new_version must be specified if update_version=True")

            old_version = f["/identification/product_version"][()].decode("utf-8")
            logger.info(f"Updating product_version from {old_version} to {new_version}")
            f["/identification/product_version"][()] = new_version.encode("utf-8")


def recompute_product_bounds(
    input_file: Path,
    output_file: Path | None = None,
    update_metadata: bool = True,
    update_version: bool = False,
    new_version: str | None = None,
) -> Path:
    """Recompute product bounds from the displacement layer and update metadata.

    Parameters
    ----------
    input_file : Path
        Path to the input OPERA DISP-S1 NetCDF file.
    output_file : Path, optional
        Path to the output file. If not provided, will add "_corrected" suffix.
    update_metadata : bool
        Whether to update the processing_start_datetime field.
        Default = True
    update_version : bool
        Whether to update the product version field.
        Default = False
    new_version : str, optional
        New version string to replace in /identification/product_version.
        Required if update_version=True.

    Returns
    -------
    Path
        Path to the output file.

    """
    input_file = Path(input_file)

    if output_file is None:
        output_file = input_file.with_stem(f"{input_file.stem}_corrected")

    output_file = Path(output_file)

    footprint_wkt = compute_bounding_polygon(input_file)

    # Copy the input file to output so we don't overwrite the original.
    logger.info(f"Copying {input_file} to {output_file}")
    shutil.copy(input_file, output_file)

    logger.info("Writing recomputed bounding polygon to product metadata")
    with h5py.File(output_file, "a") as f:
        old_polygon = f[BOUNDING_POLYGON_PATH][()].decode("utf-8")
        logger.info(f"Updating {BOUNDING_POLYGON_PATH}")
        logger.info(f"  old: {old_polygon}")
        logger.info(f"  new: {footprint_wkt}")
        f[BOUNDING_POLYGON_PATH][()] = footprint_wkt.encode("utf-8")

    logger.info("Updating metadata timestamps")
    update_metadata_timestamps(
        output_file,
        processing_datetime=datetime.now(),
        update_processing_time=update_metadata,
        update_version=update_version,
        new_version=new_version,
    )

    logger.info(f"Successfully wrote recomputed product bounds to {output_file}")
    return output_file


if __name__ == "__main__":
    tyro.cli(recompute_product_bounds)
