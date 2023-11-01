#!/usr/bin/env python
from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pprint import pformat

from dolphin import __version__ as dolphin_version
from dolphin import utils
from dolphin._background import DummyProcessPoolExecutor
from dolphin._log import get_log, log_runtime
from dolphin.opera_utils import group_by_burst
from dolphin.workflows import stitch_and_unwrap, wrapped_phase
from dolphin.workflows._utils import _create_burst_cfg, _remove_dir_if_empty
from dolphin.workflows.config import Workflow

from disp_s1 import _log, product
from disp_s1.pge_runconfig import RunConfig


@log_runtime
def run(
    cfg: Workflow,
    pge_runconfig: RunConfig,
    debug: bool = False,
):
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : Workflow
        [`Workflow`][dolphin.workflows.config.Workflow] object for controlling the
        workflow.
    debug : bool, optional
        Enable debug logging, by default False.
    pge_runconfig : RunConfig, optional
        PGE-specific metadata for the output product.
    """
    # Set the logging level for all `dolphin.` modules
    logger = get_log(name="disp_s1", debug=debug)
    if cfg.log_file:
        _log.setup_file_logging(cfg.log_file)

    logger.debug(pformat(cfg.model_dump()))
    cfg.create_dir_tree(debug=debug)

    utils.set_num_threads(cfg.worker_settings.threads_per_worker)

    try:
        grouped_slc_files = group_by_burst(cfg.cslc_file_list)
    except ValueError as e:
        # Make sure it's not some other ValueError
        if "Could not parse burst id" not in str(e):
            raise e
        # Otherwise, we have SLC files which are not OPERA burst files
        grouped_slc_files = {"": cfg.cslc_file_list}

    if cfg.amplitude_dispersion_files:
        grouped_amp_dispersion_files = group_by_burst(cfg.amplitude_dispersion_files)
    else:
        grouped_amp_dispersion_files = defaultdict(list)
    if cfg.amplitude_mean_files:
        grouped_amp_mean_files = group_by_burst(cfg.amplitude_mean_files)
    else:
        grouped_amp_mean_files = defaultdict(list)

    # ######################################
    # 1. Burst-wise Wrapped phase estimation
    # ######################################
    if len(grouped_slc_files) > 1:
        logger.info(f"Found SLC files from {len(grouped_slc_files)} bursts")
        wrapped_phase_cfgs = [
            (
                burst,  # Include the burst for logging purposes
                _create_burst_cfg(
                    cfg,
                    burst,
                    grouped_slc_files,
                    grouped_amp_mean_files,
                    grouped_amp_dispersion_files,
                ),
            )
            for burst in grouped_slc_files
        ]
        for _, burst_cfg in wrapped_phase_cfgs:
            burst_cfg.create_dir_tree()
        # Remove the mid-level directories which will be empty due to re-grouping
        _remove_dir_if_empty(cfg.phase_linking._directory)
        _remove_dir_if_empty(cfg.ps_options._directory)

    else:
        # grab the only key (either a burst, or "") and use that
        b = list(grouped_slc_files.keys())[0]
        wrapped_phase_cfgs = [(b, cfg)]

    ifg_file_list: list[Path] = []
    tcorr_file_list: list[Path] = []
    ps_file_list: list[Path] = []
    # The comp_slc tracking object is a dict, since we'll need to organize
    # multiple comp slcs by burst (they'll have the same filename)
    comp_slc_dict: dict[str, Path] = {}
    # Now for each burst, run the wrapped phase estimation
    # Try running several bursts in parallel...
    Executor = (
        ProcessPoolExecutor
        if cfg.worker_settings.n_parallel_bursts > 1
        else DummyProcessPoolExecutor
    )
    mw = cfg.worker_settings.n_parallel_bursts
    ctx = mp.get_context("spawn")
    with Executor(max_workers=mw, mp_context=ctx) as exc:
        fut_to_burst = {
            exc.submit(
                wrapped_phase.run,
                burst_cfg,
                debug=debug,
            ): burst
            for burst, burst_cfg in wrapped_phase_cfgs
        }
        for fut in fut_to_burst:
            burst = fut_to_burst[fut]

            cur_ifg_list, comp_slc, tcorr, ps_file = fut.result()
            ifg_file_list.extend(cur_ifg_list)
            comp_slc_dict[burst] = comp_slc
            tcorr_file_list.append(tcorr)
            ps_file_list.append(ps_file)

    # ###################################
    # 2. Stitch and unwrap interferograms
    # ###################################

    (
        unwrapped_paths,
        conncomp_paths,
        ifg_corr_paths,
        stitched_tcorr_file,
        stitched_ps_file,
    ) = stitch_and_unwrap.run(
        ifg_file_list=ifg_file_list,
        tcorr_file_list=tcorr_file_list,
        ps_file_list=ps_file_list,
        cfg=cfg,
        debug=debug,
    )

    # ######################################
    # 3. Finalize the output as an HDF5 product
    # ######################################
    # Group all the non-compressed SLCs by date
    date_to_slcs = utils.group_by_date(
        [p for p in cfg.cslc_file_list if "compressed" not in p.name]
    )

    out_dir = pge_runconfig.product_path_group.output_directory
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Creating {len(unwrapped_paths)} outputs in {out_dir}")
    for unw_p, cc_p, s_corr_p in zip(
        unwrapped_paths,
        conncomp_paths,
        ifg_corr_paths,
    ):
        output_name = out_dir / unw_p.with_suffix(".nc").name
        # Get the current list of acq times for this product
        dair_pair = utils.get_dates(output_name)
        secondary_date = dair_pair[1]
        cur_slc_list = date_to_slcs[(secondary_date,)]

        product.create_output_product(
            output_name=output_name,
            unw_filename=unw_p,
            conncomp_filename=cc_p,
            tcorr_filename=stitched_tcorr_file,
            ifg_corr_filename=s_corr_p,
            ps_mask_filename=stitched_ps_file,
            pge_runconfig=pge_runconfig,
            cslc_files=cur_slc_list,
            corrections={},
        )

    if pge_runconfig.product_path_group.save_compressed_slc:
        logger.info(f"Saving {len(comp_slc_dict.items())} compressed SLCs")
        output_dir = out_dir / "compressed_slcs"
        output_dir.mkdir(exist_ok=True)
        product.create_compressed_products(
            comp_slc_dict=comp_slc_dict,
            output_dir=output_dir,
        )

    # Print the maximum memory usage for each worker
    max_mem = utils.get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Config file dolphin version: {cfg._dolphin_version}")
    logger.info(f"Current running dolphin version: {dolphin_version}")
    logger.info(f"Product type: {pge_runconfig.primary_executable.product_type}")
    logger.info(f"Product version: {pge_runconfig.product_path_group.product_version}")
