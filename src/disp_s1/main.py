from __future__ import annotations

from dolphin._log import get_log, log_runtime
from dolphin.io import load_gdal
from dolphin.workflows.config import DisplacementWorkflow
from dolphin.workflows.displacement import run as run_displacement
from opera_utils import get_dates, group_by_date

from disp_s1 import product
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

    logger = get_log(name="disp_s1", debug=debug)
    out_dir = pge_runconfig.product_path_group.output_directory
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Creating {len(out_paths.unwrapped_paths)} outputs in {out_dir}")
    for unw_p, cc_p, s_corr_p, tropo_p, iono_p in zip(
        out_paths.unwrapped_paths,
        out_paths.conncomp_paths,
        out_paths.stitched_cor_paths,
        out_paths.tropospheric_corrections,
        out_paths.ionospheric_corrections,
    ):
        output_name = out_dir / unw_p.with_suffix(".nc").name
        # Get the current list of acq times for this product
        dair_pair = get_dates(output_name)
        secondary_date = dair_pair[1]
        cur_slc_list = date_to_slcs[(secondary_date,)]

        if tropo_p and iono_p:
            corrections = {
                "troposphere": load_gdal(tropo_p),
                "ionosphere": load_gdal(iono_p),
            }
        else:
            logger.error(f"Missing {tropo_p = }, {iono_p = }. Creating empty layer.")

        product.create_output_product(
            output_name=output_name,
            unw_filename=unw_p,
            conncomp_filename=cc_p,
            temp_coh_filename=out_paths.stitched_temp_coh_file,
            ifg_corr_filename=s_corr_p,
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
