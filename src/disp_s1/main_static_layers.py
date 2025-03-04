from __future__ import annotations

import logging
from pathlib import Path

import opera_utils.geometry
from dolphin import Bbox, PathOrStr
from dolphin._log import log_runtime, setup_logging
from dolphin.utils import get_max_memory_usage
from opera_utils.geometry import Layer

from disp_s1 import __version__
from disp_s1.pge_runconfig import StaticLayersRunConfig

logger = logging.getLogger(__name__)

__all__ = ["run_static_layers"]


@log_runtime
def run_static_layers(
    pge_runconfig: StaticLayersRunConfig,
) -> None:
    """Run the Static Layers workflow for one of the DISP-S1 frames.

    Parameters
    ----------
    pge_runconfig : RunConfig
        PGE-specific metadata for the output product.

    """
    setup_logging(logger_name="disp_s1", filename=pge_runconfig.log_file)
    epsg, bounds = opera_utils.get_frame_bbox(pge_runconfig.input_file_group.frame_id)
    logger.info("Creating geometry layers Workflow")
    output_dir = pge_runconfig.product_path_group.scratch_path

    _geom_files = opera_utils.geometry.stitch_geometry_layers(
        local_hdf5_files=pge_runconfig.dynamic_ancillary_file_group.geometry_files,
        layers=[Layer.LOS_EAST, Layer.LOS_NORTH, Layer.LAYOVER_SHADOW_MASK],
        strides={"x": 6, "y": 3},
        output_dir=output_dir,
        out_bounds=bounds,
        out_bounds_epsg=epsg,
    )
    # TODO: add DEM for now
    warp_dem_to_utm(
        pge_runconfig.dynamic_ancillary_file_group.dem_file,
        epsg,
        bounds,
        output_dir=output_dir,
    )

    logger.info(f"Product type: {pge_runconfig.primary_executable.product_type}")
    logger.info(f"Product version: {pge_runconfig.product_path_group.product_version}")
    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Current running disp_s1 version: {__version__}")


def warp_dem_to_utm(
    dem_file: PathOrStr, epsg: int, bounds: Bbox, output_dir: Path
) -> Path:
    """Warp a lat/lon EPSG:4326 DEM into a UTM grid using cubic interpolation.

    Parameters
    ----------
    dem_file : PathOrStr
        Path to the input DEM file in EPSG:4326 projection.
    epsg : int
        EPSG code for the target UTM projection.
    bounds : Bbox
        Bounding box of the area of interest in the target UTM projection.
    output_dir : Path
        Directory where the output warped DEM will be saved.

    Returns
    -------
    Path
        Path to the output warped DEM file.

    Notes
    -----
    This function uses GDAL's Warp functionality to perform the reprojection
    and resampling of the input DEM.

    """
    from osgeo import gdal

    output_path = output_dir / "dem_warped_utm.tif"

    warp_options = gdal.WarpOptions(
        dstSRS=f"EPSG:{epsg}",
        outputBounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        resampleAlg=gdal.GRA_Cubic,
        format="GTiff",
        srcNodata=None,
        dstNodata=None,
    )

    gdal.Warp(str(output_path), str(dem_file), options=warp_options)

    return output_path
