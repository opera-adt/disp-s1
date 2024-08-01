from __future__ import annotations

import logging
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from tqdm import tqdm

from disp_s1 import __version__, product
from disp_s1.pge_runconfig import RunConfig

logger = logging.getLogger(__name__)


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

    # Run dolphin's displacement workflow
    out_paths = run_displacement(cfg=cfg, debug=debug)

    # Create the short wavelength layer for the product
    if hasattr(cfg, "spatial_wavelength_cutoff"):
        wavelength_cutoff = cfg.spatial_wavelength_cutoff
    else:
        wavelength_cutoff = 50_000

    # Finalize the output as an HDF5 product
    # Group all the CSLCs by date to pick out ref/secondaries
    date_to_cslc_files = group_by_date(cfg.cslc_file_list, date_idx=0)

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
    create_displacement_products(
        out_paths,
        out_dir=out_dir,
        date_to_cslc_files=date_to_cslc_files,
        pge_runconfig=pge_runconfig,
        wavelength_cutoff=wavelength_cutoff,
    )
    logger.info("Finished creating output products.")

    if pge_runconfig.product_path_group.save_compressed_slc:
        logger.info(f"Saving {len(out_paths.comp_slc_dict.items())} compressed SLCs")
        output_dir = out_dir / "compressed_slcs"
        output_dir.mkdir(exist_ok=True)
        product.create_compressed_products(
            comp_slc_dict=out_paths.comp_slc_dict,
            output_dir=output_dir,
            cslc_file_list=cfg.cslc_file_list,
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


class ProductFiles(NamedTuple):
    """Named tuple to hold the files for each NetCDF product."""

    unwrapped: Path
    conncomp: Path
    temp_coh: Path
    correlation: Path
    ps_mask: Path
    troposphere: Path | None
    ionosphere: Path | None
    unwrapper_mask: Path | None


def process_product(
    files: ProductFiles,
    out_dir: Path,
    date_to_cslc_files: Mapping[tuple[datetime], list[Path]],
    pge_runconfig: RunConfig,
    wavelength_cutoff: float,
) -> Path:
    """Create a single displacement product.

    Parameters
    ----------
    files : ProductFiles
        NamedTuple containing paths for all displacement-related files.
    out_dir : Path
        Output directory for the product.
    date_to_cslc_files: Mapping[tuple[datetime], list[Path]]
        Dictionary mapping dates to real/compressed SLC files.
    pge_runconfig : RunConfig
        Configuration object for the PGE run.
    wavelength_cutoff : float
        Wavelength cutoff for filtering long wavelengths.

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
    ref_date, secondary_date = get_dates(output_name)[:2]
    # The reference one could be compressed, or real
    # Also possible to have multiple compressed fiels with same reference date
    ref_slc_files = date_to_cslc_files[(ref_date,)]
    logger.debug(f"Found {len(ref_slc_files)} for reference date {ref_date}")
    secondary_slc_files = date_to_cslc_files[(secondary_date,)]
    assert len(secondary_slc_files) == 1

    product.create_output_product(
        output_name=output_name,
        unw_filename=files.unwrapped,
        conncomp_filename=files.conncomp,
        temp_coh_filename=files.temp_coh,
        ifg_corr_filename=files.correlation,
        ps_mask_filename=files.ps_mask,
        unwrapper_mask_filename=files.unwrapper_mask,
        pge_runconfig=pge_runconfig,
        reference_cslc_file=ref_slc_files[-1],
        secondary_cslc_file=secondary_slc_files[0],
        corrections=corrections,
        wavelength_cutoff=wavelength_cutoff,
    )

    return output_name


def create_displacement_products(
    out_paths: OutputPaths,
    out_dir: Path,
    date_to_cslc_files: Mapping[tuple[datetime], list[Path]],
    pge_runconfig: RunConfig,
    wavelength_cutoff: float = 50_000.0,
    max_workers: int = 5,
) -> None:
    """Run parallel processing for all interferograms.

    Parameters
    ----------
    out_paths : OutputPaths
        Object containing paths for various output files.
    out_dir : Path
        Output directory for the products.
    date_to_cslc_files: Mapping[tuple[datetime], list[Path]]
        Dictionary mapping dates to real/compressed SLC files.
    pge_runconfig : RunConfig
        Configuration object for the PGE run.
    wavelength_cutoff : float
        Wavelength cutoff (in meters) for filtering long wavelengths.
        Default is 50_000.
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
            # TODO: we should save/use the "combined_mask" from dolphin
            # This may have other pixels used in addition to the water mask provided
            unwrapper_mask=pge_runconfig.dynamic_ancillary_file_group.mask_file,
        )
        for unw, cc, cor, tropo, iono in zip(
            out_paths.unwrapped_paths,
            out_paths.conncomp_paths,
            out_paths.stitched_cor_paths,
            tropo_files,
            iono_files,
        )
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_product,
                file,
                out_dir,
                date_to_cslc_files,
                pge_runconfig,
                wavelength_cutoff,
            )
            for file in files
        ]

        with tqdm(total=len(files), desc="Processing products") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)
