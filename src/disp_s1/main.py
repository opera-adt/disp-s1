from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path
from typing import NamedTuple

from dolphin import interferogram, stitching
from dolphin._log import log_runtime, setup_logging
from dolphin.io import load_gdal
from dolphin.unwrap import grow_conncomp_snaphu
from dolphin.utils import DummyProcessPoolExecutor, full_suffix, get_max_memory_usage
from dolphin.workflows.config import DisplacementWorkflow, UnwrapOptions
from dolphin.workflows.displacement import OutputPaths
from dolphin.workflows.displacement import run as run_displacement
from opera_utils import get_dates, group_by_date

from disp_s1 import __version__, product
from disp_s1._masking import create_mask_from_distance
from disp_s1.pge_runconfig import RunConfig

from ._reference import ReferencePoint, read_reference_point

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
    cfg.work_directory.mkdir(exist_ok=True, parents=True)
    # Setup the binary mask as dolphin expects
    if pge_runconfig.dynamic_ancillary_file_group.mask_file:
        water_binary_mask = cfg.work_directory / "water_binary_mask.tif"
        create_mask_from_distance(
            water_distance_file=pge_runconfig.dynamic_ancillary_file_group.mask_file,
            output_file=water_binary_mask,
            # Set a little conservative for the general processing
            land_buffer=2,
            ocean_buffer=2,
        )
        cfg.mask_file = water_binary_mask

    # Run dolphin's displacement workflow
    out_paths = run_displacement(cfg=cfg, debug=debug)

    # Ensure the wavelength is set for the short wavelength layer
    if hasattr(cfg, "spatial_wavelength_cutoff"):
        wavelength_cutoff = cfg.spatial_wavelength_cutoff
    else:
        wavelength_cutoff = 25_000

    # Read the reference point
    assert out_paths.timeseries_paths is not None
    ref_point = read_reference_point(out_paths.timeseries_paths[0].parent)

    # Find the geometry files, if created
    try:
        los_east_file = next(cfg.work_directory.rglob("los_east.tif"))
        los_north_file = los_east_file.parent / "los_north.tif"
    except StopIteration:
        los_east_file = los_north_file = None

    # Finalize the output as an HDF5 product
    # Group all the CSLCs by date to pick out ref/secondaries
    date_to_cslc_files = group_by_date(cfg.cslc_file_list, date_idx=0)

    out_dir = pge_runconfig.product_path_group.output_directory
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Creating {len(out_paths.timeseries_paths)} outputs in {out_dir}")

    # Check for a network unwrapping approach:
    grouped_timeseries_paths = group_by_date(out_paths.timeseries_paths)
    disp_date_keys = set(grouped_timeseries_paths.keys())

    # Check and update correlation paths
    if set(group_by_date(out_paths.stitched_cor_paths).keys()) != disp_date_keys:
        out_paths.stitched_cor_paths = (
            interferogram.estimate_interferometric_correlations(
                ifg_filenames=out_paths.timeseries_paths,
                window_size=(11, 11),
            )
        )

    # Check and update connected components paths
    if set(group_by_date(out_paths.conncomp_paths).keys()) != disp_date_keys:
        method = cfg.unwrap_options.unwrap_method
        if method == "snaphu":
            out_paths.conncomp_paths = _update_snaphu_conncomps(
                out_paths.timeseries_paths, out_paths, cfg
            )
        elif method == "spurt":
            out_paths.conncomp_paths = _update_spurt_conncomps(
                out_paths.conncomp_paths, out_paths.timeseries_paths
            )
        else:
            raise NotImplementedError(
                f"Regrowing connected components not implemented for {method}"
            )

    # Check tropospheric corrections
    if out_paths.tropospheric_corrections is not None:
        _assert_dates_match(
            disp_date_keys, out_paths.tropospheric_corrections, "troposphere"
        )

    # Check ionospheric corrections
    if out_paths.ionospheric_corrections is not None:
        _assert_dates_match(
            disp_date_keys, out_paths.ionospheric_corrections, "ionosphere"
        )

    if pge_runconfig.dynamic_ancillary_file_group.mask_file:
        aggressive_water_binary_mask = (
            cfg.work_directory / "water_binary_mask_nobuffer.tif"
        )
        tmp_outfile = aggressive_water_binary_mask.with_suffix(".temp.tif")
        create_mask_from_distance(
            water_distance_file=pge_runconfig.dynamic_ancillary_file_group.mask_file,
            # Make the file in lat/lon
            output_file=tmp_outfile,
            # Still don't trust the land water 100%
            land_buffer=1,
            # Trust the ocean buffer
            ocean_buffer=0,
        )
        # Then need to warp to match the output UTM files
        stitching.warp_to_match(
            input_file=tmp_outfile,
            match_file=out_paths.timeseries_paths[0],
            output_file=aggressive_water_binary_mask,
        )
    else:
        aggressive_water_binary_mask = None

    logger.info(f"Creating {len(out_paths.timeseries_paths)} outputs in {out_dir}")
    create_displacement_products(
        out_paths,
        out_dir=out_dir,
        date_to_cslc_files=date_to_cslc_files,
        pge_runconfig=pge_runconfig,
        dolphin_config=cfg,
        wavelength_cutoff=wavelength_cutoff,
        reference_point=ref_point,
        los_east_file=los_east_file,
        los_north_file=los_north_file,
        water_mask=aggressive_water_binary_mask,
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
    disp_date_keys: set[datetime], test_paths: Iterable[Path], name: str
) -> None:
    """Assert that the dates in `paths_to_check` match the reference dates.

    Parameters
    ----------
    disp_date_keys : list[str]
        list of reference dates to compare against.
    test_paths : list[Path]
        list of paths to check for date consistency.
    name : str
        Description of the paths being checked (for error message).

    Raises
    ------
    AssertionError
        If the dates in the paths to check do not match the reference dates.

    """
    if set(group_by_date(test_paths).keys()) != disp_date_keys:
        msg = f"Mismatch of dates found for {name}:"
        msg += f"{disp_date_keys = }, but {name} has {test_paths}"
        raise ValueError(msg)


class ProductFiles(NamedTuple):
    """Named tuple to hold the files for each NetCDF product."""

    unwrapped: Path
    conncomp: Path
    temp_coh: Path
    correlation: Path
    shp_counts: Path
    ps_mask: Path
    troposphere: Path | None
    ionosphere: Path | None
    unwrapper_mask: Path | None
    similarity: Path
    water_mask: Path | None


def process_product(
    files: ProductFiles,
    out_dir: Path,
    date_to_cslc_files: Mapping[tuple[datetime], list[Path]],
    pge_runconfig: RunConfig,
    dolphin_config: DisplacementWorkflow,
    wavelength_cutoff: float,
    reference_point: ReferencePoint | None = None,
    los_east_file: Path | None = None,
    los_north_file: Path | None = None,
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
    dolphin_config : dolphin.workflows.DisplacementWorkflow
        Configuration object run by `dolphin`.
    wavelength_cutoff : float
        Wavelength cutoff for filtering long wavelengths.
    reference_point : ReferencePoint, optional
        Reference point recorded from dolphin after unwrapping.
        If none, leaves product attributes empty.
    los_east_file : Path, optional
        Path to the east component of line of sight unit vector
    los_north_file : Path, optional
        Path to the north component of line of sight unit vector

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

    output_name = files.unwrapped.name.replace(full_suffix(files.unwrapped), ".nc")
    output_path = out_dir / output_name
    ref_date, secondary_date = get_dates(output_name)[:2]
    # The reference one could be compressed, or real
    # Also possible to have multiple compressed files with same reference date
    ref_slc_files = date_to_cslc_files[(ref_date,)]
    logger.info(f"Found {len(ref_slc_files)} for reference date {ref_date}")
    secondary_slc_files = date_to_cslc_files[(secondary_date,)]
    logger.info(f"Found {len(secondary_slc_files)} for secondary date {secondary_date}")

    product.create_output_product(
        output_name=output_path,
        unw_filename=files.unwrapped,
        conncomp_filename=files.conncomp,
        temp_coh_filename=files.temp_coh,
        ifg_corr_filename=files.correlation,
        ps_mask_filename=files.ps_mask,
        shp_count_filename=files.shp_counts,
        unwrapper_mask_filename=files.unwrapper_mask,
        similarity_filename=files.similarity,
        water_mask_filename=files.water_mask,
        los_east_file=los_east_file,
        los_north_file=los_north_file,
        pge_runconfig=pge_runconfig,
        dolphin_config=dolphin_config,
        reference_cslc_files=ref_slc_files,
        secondary_cslc_files=secondary_slc_files,
        corrections=corrections,
        wavelength_cutoff=wavelength_cutoff,
        reference_point=reference_point,
    )

    return output_path


def create_displacement_products(
    out_paths: OutputPaths,
    out_dir: Path,
    date_to_cslc_files: Mapping[tuple[datetime], list[Path]],
    pge_runconfig: RunConfig,
    dolphin_config: DisplacementWorkflow,
    wavelength_cutoff: float = 25_000.0,
    reference_point: ReferencePoint | None = None,
    los_east_file: Path | None = None,
    los_north_file: Path | None = None,
    water_mask: Path | None = None,
    max_workers: int = 3,
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
    dolphin_config : dolphin.workflows.DisplacementWorkflow
        Configuration object run by `dolphin`.
    reference_point : ReferencePoint, optional
        Named tuple with (row, col, lat, lon) of selected reference pixel.
        If None, will record empty in the dataset's attributes
    wavelength_cutoff : float
        Wavelength cutoff (in meters) for filtering long wavelengths.
        Default is 25_000.
    reference_point : ReferencePoint, optional
        Reference point recorded from dolphin after unwrapping.
        If none, leaves product attributes empty.
    los_east_file : Path, optional
        Path to the east component of line of sight unit vector
    los_north_file : Path, optional
        Path to the north component of line of sight unit vector
    water_mask : Path, optional
        Binary water mask to use for output product.
        If provided, is used in the `recommended_mask`.
    max_workers : int
        Number of parallel products to process.
        Default is 3.

    """
    # Extra logging for product creation
    setup_logging(logger_name="disp_s1", debug=True)
    tropo_files = out_paths.tropospheric_corrections or [None] * len(
        out_paths.timeseries_paths
    )
    iono_files = out_paths.ionospheric_corrections or [None] * len(
        out_paths.timeseries_paths
    )
    unwrapper_mask_files = [
        # TODO: probably relying too much on dolphin's internals to be safe
        # we should figure out how to save/use the "combined_mask" from dolphin
        Path(str(p).replace(".cor.tif", ".mask.tif"))
        for p in out_paths.stitched_cor_paths
    ]
    files = [
        ProductFiles(
            unwrapped=unw,
            conncomp=cc,
            temp_coh=out_paths.stitched_temp_coh_file,
            correlation=cor,
            ps_mask=out_paths.stitched_ps_file,
            shp_counts=out_paths.stitched_shp_count_file,
            troposphere=tropo,
            ionosphere=iono,
            unwrapper_mask=mask_f,
            similarity=out_paths.stitched_similarity_file,
            water_mask=water_mask,
        )
        for unw, cc, cor, tropo, iono, mask_f in zip(
            out_paths.timeseries_paths,
            out_paths.conncomp_paths,
            out_paths.stitched_cor_paths,
            tropo_files,
            iono_files,
            unwrapper_mask_files,
        )
    ]

    executor_class = (
        ProcessPoolExecutor if max_workers > 1 else DummyProcessPoolExecutor
    )
    ctx = get_context("spawn")
    with executor_class(max_workers=max_workers, mp_context=ctx) as executor:
        list(  # Force evaluation to retrieve results/raise exceptions
            executor.map(
                process_product,
                files,
                repeat(out_dir),
                repeat(date_to_cslc_files),
                repeat(pge_runconfig),
                repeat(dolphin_config),
                repeat(wavelength_cutoff),
                repeat(reference_point),
                repeat(los_east_file),
                repeat(los_north_file),
            )
        )


def _update_snaphu_conncomps(
    timeseries_paths: Sequence[Path],
    out_paths: OutputPaths,
    unwrap_options: UnwrapOptions,
) -> list[Path]:
    """Update connected components using SNAPHU unwrapping method.

    Parameters
    ----------
    timeseries_paths : list[Path]
        list of paths to the timeseries files.
    out_paths : OutPaths
        Object containing various output paths.
    unwrap_options : [dolphin.workflows.config.UnwrapOptions][]
        Configuration object containing unwrapping options.

    Returns
    -------
    list[Path]
        list of updated connected component paths.

    """
    new_paths = []
    for unw_f, cor_f in zip(timeseries_paths, out_paths.stitched_cor_paths):
        mask_file = Path(str(cor_f).replace(".cor.tif", ".mask.tif"))
        new_path = grow_conncomp_snaphu(
            unw_filename=unw_f,
            corr_filename=cor_f,
            nlooks=50,
            mask_filename=mask_file,
            cost=unwrap_options.snaphu_options.cost,
            scratchdir=unwrap_options._directory / "scratch2",
        )
        new_paths.append(new_path)
    return new_paths


def _update_spurt_conncomps(
    conncomp_paths: Sequence[Path], timeseries_paths: Sequence[Path]
) -> list[Path]:
    """Update connected components using SPURT unwrapping method.

    Parameters
    ----------
    conncomp_paths : list[Path]
        list of original connected component paths.
    timeseries_paths : list[Path]
        list of paths to the timeseries files.

    Returns
    -------
    list[Path]
        list of updated connected component paths.

    """
    new_conncomp_paths: list[Path] = []
    for cc_p, ts_p in zip(conncomp_paths, timeseries_paths, strict=False):
        new_name = cc_p.parent / str(ts_p.name).replace(
            full_suffix(ts_p), full_suffix(cc_p)
        )
        new_conncomp_paths.append(cc_p.rename(new_name))
    return new_conncomp_paths
