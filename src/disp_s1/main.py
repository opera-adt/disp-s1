from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from dolphin._log import log_runtime, setup_logging
from dolphin.io import load_gdal
from dolphin.utils import get_max_memory_usage
from dolphin.workflows.config import DisplacementWorkflow
from dolphin.workflows.displacement import OutputPaths
from dolphin.workflows.displacement import run as run_displacement
from opera_utils import get_dates, group_by_date
from tqdm.contrib.concurrent import thread_map

from disp_s1 import __version__, product
from disp_s1.pge_runconfig import RunConfig

logger = logging.getLogger(__name__)


class ProductFiles(NamedTuple):
    """Named tuple to hold the files for each NetCDF product."""

    unwrapped: Path
    conncomp: Path
    temp_coh: Path
    correlation: Path
    ps_mask: Path
    troposphere: Path | None
    ionosphere: Path | None


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
    unw_date_keys = list(grouped_unwrapped_paths.keys())
    _assert_dates_match(unw_date_keys, out_paths.conncomp_paths, "connected components")
    _assert_dates_match(unw_date_keys, out_paths.stitched_cor_paths, "correlation")

    if out_paths.tropospheric_corrections is not None:
        _assert_dates_match(
            unw_date_keys, out_paths.tropospheric_corrections, "troposphere"
        )

    if out_paths.ionospheric_corrections is not None:
        _assert_dates_match(
            unw_date_keys, out_paths.ionospheric_corrections, "ionosphere"
        )

    logger.info(f"Creating {len(out_paths.unwrapped_paths)} outputs in {out_dir}")
    run_parallel_processing(out_paths, out_dir, date_to_slcs, pge_runconfig)
    logger.info("Finished creating output products.")

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


def _assert_dates_match(
    unw_date_keys: list[datetime], test_paths: list[Path], name: str
):
    if list(group_by_date(test_paths).keys()) != unw_date_keys:
        msg = f"Mismatch of dates found for {name}:"
        msg += f"{unw_date_keys = }, but {name} has {test_paths}"
        raise ValueError(msg)


def process_product(
    files: ProductFiles,
    out_dir: Path,
    date_to_slcs: dict,
    pge_runconfig: RunConfig,
) -> Path:
    """Process a single interferogram and create the output product.

    Parameters
    ----------
    files : ProductFiles
        NamedTuple containing paths for all interferogram-related files.
    out_dir : Path
        Output directory for the product.
    date_to_slcs : dict
        Dictionary mapping dates to corresponding SLC files.
    pge_runconfig : RunConfig
        Configuration object for the PGE run.

    Returns
    -------
    Path
        Path to the processed output.

    """
    corrections = {}

    if files.troposphere is not None:
        corrections["troposphere"] = load_gdal(files.troposphere)
    else:
        logger.warning(
            "Missing tropospheric correction for %s. Creating empty layer.",
            files.unwrapped,
        )

    if files.ionosphere is not None:
        corrections["ionosphere"] = load_gdal(files.ionosphere)
    else:
        logger.warning(
            "Missing ionospheric correction for %s. Creating empty layer.",
            files.unwrapped,
        )

    output_name = out_dir / files.unwrapped.with_suffix(".nc").name
    dair_pair = get_dates(output_name)
    secondary_date = dair_pair[1]
    cur_slc_list = date_to_slcs[(secondary_date,)]

    product.create_output_product(
        output_name=output_name,
        unw_filename=files.unwrapped,
        conncomp_filename=files.conncomp,
        temp_coh_filename=files.temp_coh,
        ifg_corr_filename=files.correlation,
        ps_mask_filename=files.ps_mask,
        pge_runconfig=pge_runconfig,
        cslc_files=cur_slc_list,
        corrections=corrections,
    )

    return output_name


def run_parallel_processing(
    out_paths: OutputPaths,
    out_dir: Path,
    date_to_slcs: dict,
    pge_runconfig: RunConfig,
    max_workers: int = 3,
) -> None:
    """Run parallel processing for all interferograms.

    Parameters
    ----------
    out_paths : OutputPaths
        Object containing paths for various output files.
    out_dir : Path
        Output directory for the products.
    date_to_slcs : dict
        Dictionary mapping dates to corresponding SLC files.
    pge_runconfig : RunConfig
        Configuration object for the PGE run.
    max_workers : int
        Number of parallel products to process.
        Default is 3.

    """
    tropo_files = out_paths.tropospheric_corrections or [None] * len(
        out_paths.unwrapped_paths
    )
    iono_files = out_paths.ionospheric_corrections or [None] * len(
        out_paths.unwrapped_paths
    )
    files = [
        ProductFiles(
            unwrapped=unw,
            conncomp=cc,
            temp_coh=out_paths.stitched_temp_coh_file,
            correlation=cor,
            ps_mask=out_paths.stitched_ps_file,
            troposphere=tropo,
            ionosphere=iono,
        )
        for unw, cc, cor, tropo, iono in zip(
            out_paths.unwrapped_paths,
            out_paths.conncomp_paths,
            out_paths.stitched_cor_paths,
            tropo_files,
            iono_files,
        )
    ]

    _results = thread_map(
        lambda x: process_product(x, out_dir, date_to_slcs, pge_runconfig),
        files,
        max_workers=max_workers,
        desc="Processing products",
    )
