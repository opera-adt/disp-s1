from __future__ import annotations

import logging

from dolphin._log import log_runtime, setup_logging
from dolphin.io import load_gdal
from dolphin.utils import get_max_memory_usage
from dolphin.workflows.config import DisplacementWorkflow
from dolphin.workflows.displacement import run as run_displacement
from opera_utils import get_dates, group_by_date

from disp_s1 import __version__, product
from disp_s1.pge_runconfig import RunConfig


@log_runtime
def run(
    cfg: DisplacementWorkflow,
    pge_runconfig: RunConfig,
    debug: bool = False,
):
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : DisplacementWorkflow
        `DisplacementWorkflow` object for controlling the workflow.
    debug : bool, optional
        Enable debug logging, by default False.
    pge_runconfig : RunConfig, optional
        PGE-specific metadata for the output product.

    """
    setup_logging(logger_name="disp_s1", debug=debug, filename=cfg.log_file)

    logger = logging.getLogger(__name__)

    # ######################################
    # 1. Run dolphin's displacement workflow
    # ######################################
    out_paths = run_displacement(cfg=cfg, debug=debug)

    # #########################################
    # 2. Finalize the output as an HDF5 product
    # #########################################
    # Group all the non-compressed SLCs by date
    date_to_slcs = group_by_date(
        [p for p in cfg.cslc_file_list if "compressed" not in p.name],
        # We only care about the product date, not the Generation Date
        date_idx=0,
    )

    out_dir = pge_runconfig.product_path_group.output_directory
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Creating {len(out_paths.unwrapped_paths)} outputs in {out_dir}")

    # group dataset based on date to find corresponding files and set None
    # for the layers that do not exist: correction layers specifically
    grouped_unwrapped_paths = group_by_date(out_paths.unwrapped_paths)
    grouped_conncomp_paths = group_by_date(out_paths.conncomp_paths)
    grouped_cor_paths = group_by_date(out_paths.stitched_cor_paths)
    grouped_tropospheric_corrections = None
    grouped_ionospheric_corrections = None

    if out_paths.tropospheric_corrections is not None:
        grouped_tropospheric_corrections = group_by_date(
            out_paths.tropospheric_corrections
        )

    if out_paths.ionospheric_corrections is not None:
        grouped_ionospheric_corrections = group_by_date(
            out_paths.ionospheric_corrections
        )

    corrections = {}

    # Package the existing layers for each interferogram
    # TODO: we need to think about how to package if a network
    # inversion is applied and interferograms are not single reference
    for key in grouped_unwrapped_paths.keys():
        if grouped_tropospheric_corrections is not None:
            corrections["troposphere"] = load_gdal(
                grouped_tropospheric_corrections.get(key)[0]
            )
        else:
            logger.warning("Missing tropospheric correction. Creating empty layer.")

        if grouped_ionospheric_corrections is not None:
            corrections["ionosphere"] = load_gdal(
                grouped_ionospheric_corrections.get(key)[0]
            )
        else:
            logger.warning("Missing ionospheric correction. Creating empty layer.")

        unw_p = grouped_unwrapped_paths.get(key)[0]
        output_name = out_dir / unw_p.with_suffix(".nc").name
        # Get the current list of acq times for this product
        dair_pair = get_dates(output_name)
        secondary_date = dair_pair[1]
        cur_slc_list = date_to_slcs[(secondary_date,)]

        product.create_output_product(
            output_name=output_name,
            unw_filename=unw_p,
            conncomp_filename=grouped_conncomp_paths.get(key)[0],
            temp_coh_filename=out_paths.stitched_temp_coh_file,
            ifg_corr_filename=grouped_cor_paths.get(key)[0],
            ps_mask_filename=out_paths.stitched_ps_file,
            pge_runconfig=pge_runconfig,
            cslc_files=cur_slc_list,
            corrections=corrections,
        )

    if pge_runconfig.product_path_group.save_compressed_slc:
        logger.info(f"Saving {len(out_paths.comp_slc_dict.items())} compressed SLCs")
        output_dir = out_dir / "compressed_slcs"
        output_dir.mkdir(exist_ok=True)
        product.create_compressed_products(
            comp_slc_dict=out_paths.comp_slc_dict,
            output_dir=output_dir,
        )

    logger.info(f"Product type: {pge_runconfig.primary_executable.product_type}")
    logger.info(f"Product version: {pge_runconfig.product_path_group.product_version}")
    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Config file dolphin version: {cfg._dolphin_version}")
    logger.info(f"Current running disp_s1 version: {__version__}")
