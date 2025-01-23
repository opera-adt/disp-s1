from __future__ import annotations

import logging

import opera_utils.geometry
from dolphin._log import log_runtime, setup_logging
from dolphin.utils import get_max_memory_usage
from dolphin.workflows.config import DisplacementWorkflow
from opera_utils.geometry import Layer

from disp_s1 import __version__
from disp_s1.pge_runconfig import RunConfig

logger = logging.getLogger(__name__)

__all__ = ["run_static_layers"]


@log_runtime
def run_static_layers(
    cfg: DisplacementWorkflow,
    pge_runconfig: RunConfig,
) -> None:
    """Run the Static Layers workflow for one of the DISP-S1 frames.

    Parameters
    ----------
    cfg : DisplacementWorkflow
        `DisplacementWorkflow` object for controlling the workflow.
    pge_runconfig : RunConfig
        PGE-specific metadata for the output product.

    """
    setup_logging(logger_name="disp_s1", filename=cfg.log_file)
    epsg, bounds = opera_utils.get_frame_bbox(pge_runconfig.input_file_group.frame_id)
    logger.info("Creating geometry layers Workflow")
    output_dir = pge_runconfig.product_path_group.scratch_path

    _geom_files = opera_utils.geometry.stitch_geometry_layers(
        local_hdf5_files=cfg.correction_options.geometry_files,
        layers=[Layer.LOS_EAST, Layer.LOS_NORTH, Layer.LAYOVER_SHADOW_MASK],
        strides={"x": 6, "y": 3},
        output_dir=output_dir,
        out_bounds=bounds,
        out_bounds_epsg=epsg,
    )
    print(__version__)
    # TODO: add DEM for now
    logger.info(f"Product type: {pge_runconfig.primary_executable.product_type}")
    logger.info(f"Product version: {pge_runconfig.product_path_group.product_version}")
    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Current running disp_s1 version: {__version__}")
