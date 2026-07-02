#!/usr/bin/env python
"""Recompute the perpendicular baseline AND the bounding polygon in one pass.

Combines the two single-purpose scripts into one, updating both layers of an
OPERA DISP-S1 NetCDF in a single output file:

  - ``recompute_perpendicular_baseline.py`` -> /corrections/perpendicular_baseline
    (recomputed from the saved reference/secondary orbits)
  - ``recompute_product_bounds.py``         -> /identification/bounding_polygon
    (minimum rotated rectangle of the valid-data footprint, via the shared
    ``disp_s1._utils.extract_footprint`` used in forward production)

The input is copied to the output once, both layers are rewritten, and the
metadata timestamps / version are updated a single time at the end.

Example
-------
    $ python scripts/recompute_bperp_bbounds.py input.nc -o output.nc

"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import h5py
import tyro

from recompute_perpendicular_baseline import recompute_perpendicular_baseline
from recompute_product_bounds import (
    BOUNDING_POLYGON_PATH,
    compute_bounding_polygon,
    update_metadata_timestamps,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def recompute_bperp_bbounds(
    input_file: Path,
    output_file: Path | None = None,
    subsample: int = 50,
    update_metadata: bool = True,
    update_version: bool = False,
    new_version: str | None = None,
) -> Path:
    """Recompute perpendicular baseline + bounding polygon and update the NetCDF.

    Parameters
    ----------
    input_file : Path
        Path to the input OPERA DISP-S1 NetCDF file.
    output_file : Path, optional
        Path to the output file. If not provided, adds a "_corrected" suffix.
    subsample : int
        Subsampling factor for the perpendicular-baseline computation.
        Default = 50
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

    # 1) Perpendicular baseline. This copies input -> output and rewrites
    #    /corrections/perpendicular_baseline. Defer the metadata/version bump to
    #    the single update at the end so it is not applied twice.
    logger.info("=== Recomputing perpendicular baseline ===")
    recompute_perpendicular_baseline(
        input_file,
        output_file,
        subsample=subsample,
        update_metadata=False,
        update_version=False,
    )

    # 2) Bounding polygon, recomputed from the (untouched) displacement layer.
    logger.info("=== Recomputing bounding polygon ===")
    footprint_wkt = compute_bounding_polygon(input_file)
    with h5py.File(output_file, "a") as f:
        old_polygon = f[BOUNDING_POLYGON_PATH][()].decode("utf-8")
        logger.info(f"Updating {BOUNDING_POLYGON_PATH}")
        logger.info(f"  old: {old_polygon}")
        logger.info(f"  new: {footprint_wkt}")
        f[BOUNDING_POLYGON_PATH][()] = footprint_wkt.encode("utf-8")

    # 3) Single metadata / version update covering both edits.
    logger.info("Updating metadata timestamps")
    update_metadata_timestamps(
        output_file,
        processing_datetime=datetime.now(),
        update_processing_time=update_metadata,
        update_version=update_version,
        new_version=new_version,
    )

    logger.info(
        f"Successfully recomputed baseline + bounding polygon -> {output_file}"
    )
    return output_file


if __name__ == "__main__":
    tyro.cli(recompute_bperp_bbounds)
