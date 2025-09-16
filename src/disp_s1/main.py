from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from itertools import repeat
from multiprocessing import get_context
from pathlib import Path
from typing import NamedTuple

import numpy as np
from dolphin import PathOrStr, io, stitching
from dolphin._log import log_runtime, setup_logging
from dolphin.unwrap._utils import create_combined_mask
from dolphin.utils import DummyProcessPoolExecutor, full_suffix, get_max_memory_usage
from dolphin.workflows.config import DisplacementWorkflow
from dolphin.workflows.corrections import run as run_corrections
from dolphin.workflows.displacement import OutputPaths
from dolphin.workflows.displacement import run as run_displacement
from opera_utils import get_dates, group_by_burst, group_by_date
from opera_utils.geometry import get_incidence_angles

from disp_s1 import __version__, product
from disp_s1._masking import create_layover_shadow_masks, create_mask_from_distance
from disp_s1._ps import precompute_ps
from disp_s1.pge_runconfig import AlgorithmParameters, RunConfig, StaticLayersRunConfig

from ._reference import ReferencePoint, read_reference_point
from ._utils import (
    _convert_meters_to_radians,
    _create_correlation_images,
    _update_snaphu_conncomps,
    _update_spurt_conncomps,
)

logger = logging.getLogger(__name__)


@dataclass
class OutputPathsWithCorrections(OutputPaths):
    """Extensions of dolphin's OutputPaths with ionospheric corrections."""

    ionospheric_corrections: list[Path]


@log_runtime
def run(
    cfg: DisplacementWorkflow,
    pge_runconfig: RunConfig | StaticLayersRunConfig,
    debug: bool = False,
) -> None:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    cfg : DisplacementWorkflow
        `DisplacementWorkflow` object for controlling the workflow.
    pge_runconfig : RunConfig
        PGE-specific metadata for the output product.
    debug : bool, optional
        Enable debug logging.
        Default is False.

    """
    cfg.work_directory.mkdir(exist_ok=True, parents=True)
    setup_logging(logger_name="disp_s1", debug=debug, filename=cfg.log_file)
    # Save the start for a metadata field
    processing_start_datetime = datetime.now(timezone.utc)

    # Add a check to fail if passed duplicate dates area passed
    _assert_no_duplicate_dates(cfg.cslc_file_list)

    # Get the dates for any of the burst IDs
    date_to_files = group_by_date(cfg.cslc_file_list, date_idx=0)
    datetimes_present = list(date_to_files.keys())
    # For forward mode, assume that the last processed was the second-to-last date
    *_, (second_to_last_date,), (_last_date,) = datetimes_present
    # Strip timezones defensively
    second_to_last_date = second_to_last_date.replace(tzinfo=None)
    if pge_runconfig.primary_executable.product_type == "DISP_S1_FORWARD":
        if (
            last_processed := pge_runconfig.input_file_group.last_processed
        ) is not None:
            last_processed = last_processed.replace(tzinfo=None)
            if abs(last_processed - second_to_last_date).total_seconds() > 24 * 3600:
                raise ValueError(
                    "last_processed is set, more than 24 hours away from the"
                    " second-to-last.This is not allowed for forward mode."
                )
        else:
            pge_runconfig.input_file_group.last_processed = (
                second_to_last_date + timedelta(days=1)
            )

    # Setup the binary mask as dolphin expects
    if pge_runconfig.dynamic_ancillary_file_group.mask_file:
        water_binary_mask = cfg.work_directory / "water_binary_mask.tif"
        create_mask_from_distance(
            water_distance_file=pge_runconfig.dynamic_ancillary_file_group.mask_file,
            output_file=water_binary_mask,
            # Set a little conservative for the general processing
            land_buffer=1,
            ocean_buffer=1,
        )
        cfg.mask_file = water_binary_mask
    else:
        water_binary_mask = None

    if len(cfg.correction_options.geometry_files) > 0:
        layover_binary_mask_files = create_layover_shadow_masks(
            cslc_static_files=cfg.correction_options.geometry_files,
            output_dir=cfg.work_directory / "layover_shadow_masks",
        )
        cfg.layover_shadow_mask_files = layover_binary_mask_files

    if any("compressed" in f.name.lower() for f in cfg.cslc_file_list):
        # If we are passed Compressed SLCs, combine the old amplitudes with the
        # current real SLCs for a better estimate of amplitude dispersion for PS/SHPs
        logger.info("Combining old amplitudes with current SLCs")
        combined_dispersion_files, combined_mean_files = precompute_ps(cfg=cfg)
        cfg.amplitude_dispersion_files = combined_dispersion_files
        cfg.amplitude_mean_files = combined_mean_files
    else:
        # This is the first ministack: The amplitude estimation will be weak.
        # Drop the PS threshold to a conservative number to avoid false positives
        cfg.ps_options.amp_dispersion_threshold = 0.15

    # Run dolphin's displacement workflow
    out_paths = run_displacement(cfg=cfg, debug=debug)

    assert out_paths.timeseries_paths is not None
    assert out_paths.timeseries_residual_paths is not None
    if pge_runconfig.primary_executable.product_type == "DISP_S1_FORWARD":
        from dolphin.timeseries import _redo_reference

        logger.info(f"Re-referencing time series rasters {out_paths.timeseries_paths}")
        logger.info(f"Setting output reference to {second_to_last_date}")
        final_ts_paths, final_residual_paths = _redo_reference(
            out_paths.timeseries_paths,
            out_paths.timeseries_residual_paths,
            second_to_last_date,
            bad_pixel_mask=np.ma.nomask,
        )
        out_paths.timeseries_paths = final_ts_paths
        out_paths.timeseries_residual_paths = final_residual_paths

    # Filter by last processed date
    if last_processed := pge_runconfig.input_file_group.last_processed:
        logger.info(f"Filtering outputs before {last_processed}")
        logger.info(
            f"Before filtering: {len(out_paths.timeseries_paths)} outputs:"
            f" {out_paths.timeseries_paths}"
        )
        out_paths = _filter_before_last_processed(out_paths, last_processed)
        logger.info(
            f"After filtering: {len(out_paths.timeseries_paths)} outputs:"
            f" {out_paths.timeseries_paths}"
        )

    # Run dolphin's corrections workflow
    out_corrections_paths = run_corrections(
        cfg=cfg,
        correction_options=cfg.correction_options,
        timeseries_paths=out_paths.timeseries_paths,
        out_dir=cfg.work_directory,
        debug=debug,
    )
    assert out_corrections_paths.ionospheric_corrections is not None
    # Update the output paths with the corrections to only pass one object
    out_paths_with_corr = OutputPathsWithCorrections(
        **asdict(out_paths),
        ionospheric_corrections=out_corrections_paths.ionospheric_corrections,
    )

    create_products(
        out_paths=out_paths_with_corr,
        cfg=cfg,
        pge_runconfig=pge_runconfig,
        processing_start_datetime=processing_start_datetime,
    )

    logger.info(f"Product type: {pge_runconfig.primary_executable.product_type}")
    logger.info(f"Product version: {pge_runconfig.product_path_group.product_version}")
    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Config file dolphin version: {cfg._dolphin_version}")
    logger.info(f"Current running disp_s1 version: {__version__}")


def _filter_before_last_processed(
    out_paths: OutputPaths, last_processed: datetime
) -> OutputPaths:
    """Filter `out_paths` to products with secondary datetime > `last_processed`."""
    # For the following attributes, filter based on last_processed
    out_dict = asdict(out_paths)
    # We can add a time buffer of a day, since the minimum repeat is 6 days
    time_buffer = timedelta(days=1)
    for attr in [
        "stitched_ifg_paths",
        "stitched_cor_paths",
        "conncomp_paths",
        "timeseries_paths",
        "timeseries_residual_paths",
    ]:
        cur_files = getattr(out_paths, attr)
        # Overwrite the attribute with the filtered files
        out_dict[attr] = [
            p for p in cur_files if get_dates(p)[1] > (last_processed + time_buffer)
        ]
    return OutputPaths(**out_dict)


def create_products(
    out_paths: OutputPathsWithCorrections,
    cfg: DisplacementWorkflow,
    pge_runconfig: RunConfig,
    processing_start_datetime: datetime | None = None,
):
    """Create NetCDF products from the outputs of dolphin's displacement workflow.

    Parameters
    ----------
    out_paths: [dolphin.workflows.displacement.OutputPaths][]
        Output files of the `dolphin.workflows.DisplacementWorkflow`.
    cfg : DisplacementWorkflow
        `DisplacementWorkflow` object for controlling the workflow.
    pge_runconfig : disp_s1.pge_config.RunConfig
        PGE-specific metadata for the output product.
    processing_start_datetime : datetime.datetime, optional
        The processing start datetime. If not provided, datetime.now() is used.


    """
    if processing_start_datetime is None:
        processing_start_datetime = datetime.now(timezone.utc)
    # Read the reference point
    assert out_paths.timeseries_paths is not None
    ref_point = read_reference_point(out_paths.timeseries_paths[0].parent)

    # Find the geometry files, if created
    los_east_file: Path | None
    los_north_file: Path | None
    try:
        los_east_file = next(cfg.work_directory.rglob("los_east.tif"))
        assert los_east_file is not None
        los_north_file = los_east_file.parent / "los_north.tif"
    except StopIteration:
        los_east_file = los_north_file = None

    # Finalize the output as an HDF5 product
    out_dir = pge_runconfig.product_path_group.output_directory
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Creating {len(out_paths.timeseries_paths)} outputs in {out_dir}")

    # Check for a network unwrapping approach:
    grouped_timeseries_paths = group_by_date(out_paths.timeseries_paths)
    disp_date_keys = set(grouped_timeseries_paths.keys())

    # Check ionospheric corrections
    if out_paths.ionospheric_corrections is not None:
        _assert_dates_match(
            disp_date_keys, out_paths.ionospheric_corrections, "ionosphere"
        )

    combined_mask_file = cfg.work_directory / "combined_water_nodata_mask.tif"
    if pge_runconfig.dynamic_ancillary_file_group.mask_file:
        matching_water_binary_mask = (
            cfg.work_directory / "water_binary_mask_nobuffer.tif"
        )
        tmp_outfile = matching_water_binary_mask.with_suffix(".temp.tif")
        create_mask_from_distance(
            water_distance_file=pge_runconfig.dynamic_ancillary_file_group.mask_file,
            # Make the file in lat/lon
            output_file=tmp_outfile,
            # Give no buffer around the water
            land_buffer=0,
            ocean_buffer=0,
        )
        # Then need to warp to match the output UTM files
        # Warp to match the output UTM files
        stitching.warp_to_match(
            input_file=tmp_outfile,
            match_file=out_paths.timeseries_paths[0],
            output_file=matching_water_binary_mask,
        )
        create_combined_mask(
            mask_filename=matching_water_binary_mask,
            image_filename=out_paths.timeseries_paths[0],
            output_filename=combined_mask_file,
        )
    else:
        matching_water_binary_mask = None
        _create_nodata_mask(
            filename=out_paths.timeseries_paths[0], output_filename=combined_mask_file
        )

    # Check and update correlation paths
    if set(group_by_date(out_paths.stitched_cor_paths).keys()) != disp_date_keys:
        out_paths.stitched_cor_paths = _create_correlation_images(
            out_paths.timeseries_paths,
            window_size=(11, 11),
        )

    # Check and update connected components paths
    assert out_paths.conncomp_paths is not None
    if set(group_by_date(out_paths.conncomp_paths).keys()) != disp_date_keys:
        logger.info("Converting timeseries rasters to radians")
        timeseries_rad_paths = _convert_meters_to_radians(out_paths.timeseries_paths)
        method = cfg.unwrap_options.unwrap_method
        if method in ("snaphu", "phass", "whirlwind"):
            row_looks, col_looks = cfg.phase_linking.half_window.to_looks()
            nlooks = row_looks * col_looks
            out_paths.conncomp_paths = _update_snaphu_conncomps(
                timeseries_paths=timeseries_rad_paths,
                stitched_cor_paths=out_paths.stitched_cor_paths,
                mask_filename=combined_mask_file,
                unwrap_options=cfg.unwrap_options,
                nlooks=nlooks,
            )
        elif method == "spurt":
            out_paths.conncomp_paths = _update_spurt_conncomps(
                # We don't need the scaled-to-radians version here:
                timeseries_paths=out_paths.timeseries_paths,
                template_conncomp_path=out_paths.conncomp_paths[0],
            )
        else:
            raise NotImplementedError(
                f"Regrowing connected components not implemented for {method}"
            )

    # Get the incidence angles for /identification metadata
    if len(cfg.correction_options.geometry_files) > 0:
        near_far_incidence_angles = _get_near_far_incidence_angles(
            cfg.correction_options.geometry_files
        )
    else:
        logger.warning("Using approximate incidence angles")
        near_far_incidence_angles = 30.0, 45.0

    algorithm_parameters = AlgorithmParameters.from_yaml(
        pge_runconfig.dynamic_ancillary_file_group.algorithm_parameters_file
    )

    logger.info(f"Creating {len(out_paths.timeseries_paths)} outputs in {out_dir}")
    # Group all the CSLCs by date to pick out ref/secondaries
    date_to_cslc_files = group_by_date(cfg.cslc_file_list, date_idx=0)
    create_displacement_products(
        out_paths,
        out_dir=out_dir,
        date_to_cslc_files=date_to_cslc_files,
        pge_runconfig=pge_runconfig,
        dolphin_config=cfg,
        processing_start_datetime=processing_start_datetime,
        reference_point=ref_point,
        los_east_file=los_east_file,
        los_north_file=los_north_file,
        near_far_incidence_angles=near_far_incidence_angles,
        water_mask=matching_water_binary_mask,
        max_workers=algorithm_parameters.num_parallel_products,
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


def _assert_dates_match(
    disp_date_keys: set[tuple[datetime, ...]], test_paths: Iterable[Path], name: str
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


def _assert_no_duplicate_dates(input_file_list: Sequence[Path]) -> None:
    """Assert that for each burst ID, there is only one real SLC passed per date."""
    is_compressed = ["compressed" in str(f).lower() for f in input_file_list]

    non_compressed_slcs = [
        f for f, is_comp in zip(input_file_list, is_compressed) if not is_comp
    ]
    for burst_id, file_list in group_by_burst(non_compressed_slcs).items():
        sensing_date_list = [get_dates(f)[0] for f in file_list]
        # Use a set to check for duplicate dates
        if len(sensing_date_list) > len(set(sensing_date_list)):
            msg = f"Duplicate dates passed for {burst_id}:\n"
            file_string = "\n".join(file_list)
            msg += file_string
            raise ValueError(msg)


def _get_near_far_incidence_angles(geometry_files: list[Path]) -> tuple[float, float]:
    import numpy as np

    burst_to_static_layers = group_by_burst(sorted(geometry_files))
    burst_ids = list(burst_to_static_layers.keys())

    # Sort the bursts by IW first, then burst ID number
    def get_iw_key(burst):
        return burst.split("_")[::-1]

    sorted_bursts = sorted(burst_ids, key=get_iw_key)

    near_burst = sorted_bursts[0]  # IW1 (if exists, or lowest IW)
    far_burst = sorted_bursts[-1]  # IW3, or furthest range IW

    # There's only 1 static layers file per burst id
    near_static_layers_file = burst_to_static_layers[near_burst][0]
    far_static_layers_file = burst_to_static_layers[far_burst][0]
    # Get any normal cslc file
    near_incidence = np.nanmin(
        get_incidence_angles(near_static_layers_file, subsample_factor=30)
    ).round(1)
    far_incidence = np.nanmax(
        get_incidence_angles(far_static_layers_file, subsample_factor=30)
    ).round(1)
    return near_incidence, far_incidence


class ProductFiles(NamedTuple):
    """Named tuple to hold the files for each NetCDF product."""

    unwrapped: Path
    conncomp: Path
    temp_coh: Path
    correlation: Path
    shp_counts: Path
    ps_mask: Path
    ionosphere: Path | None
    similarity: Path
    residual: Path
    water_mask: Path | None


def process_product(
    files: ProductFiles,
    out_dir: Path,
    date_to_cslc_files: Mapping[tuple[datetime], list[Path]],
    pge_runconfig: RunConfig,
    dolphin_config: DisplacementWorkflow,
    processing_start_datetime: datetime,
    reference_point: ReferencePoint | None = None,
    los_east_file: Path | None = None,
    los_north_file: Path | None = None,
    near_far_incidence_angles: tuple[float, float] = (30.0, 45.0),
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
    processing_start_datetime : datetime.datetime
        The processing start datetime.
    reference_point : ReferencePoint, optional
        Reference point recorded from dolphin after unwrapping.
        If None, leaves product attributes empty.
    los_east_file : Path, optional
        Path to the east component of line of sight unit vector
    los_north_file : Path, optional
        Path to the north component of line of sight unit vector
    near_far_incidence_angles : tuple[float, float]
        Tuple of near range incidence angle, far range incidence angle.
        If not specified, uses approximate Sentinel-1 values of (30.0, 45.0)

    Returns
    -------
    Path
        Path to the processed output.

    """
    output_name = files.unwrapped.name.replace(full_suffix(files.unwrapped), ".nc")
    # Extra logging for product creation
    product_filename = files.unwrapped.parent / f"log_{output_name}.log"
    setup_logging(logger_name="disp_s1", debug=True, filename=product_filename)
    setup_logging(logger_name="disp_s1", debug=True, filename=pge_runconfig.log_file)

    corrections = {}

    if files.ionosphere is not None:
        corrections["ionosphere"] = io.load_gdal(files.ionosphere)
    else:
        logger.warning(
            "Missing ionospheric correction for %s. Creating empty layer.",
            files.unwrapped,
        )

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
        similarity_filename=files.similarity,
        timeseries_residual_filename=files.residual,
        water_mask_filename=files.water_mask,
        los_east_file=los_east_file,
        los_north_file=los_north_file,
        near_far_incidence_angles=near_far_incidence_angles,
        pge_runconfig=pge_runconfig,
        dolphin_config=dolphin_config,
        reference_cslc_files=ref_slc_files,
        secondary_cslc_files=secondary_slc_files,
        corrections=corrections,
        reference_point=reference_point,
        processing_start_datetime=processing_start_datetime,
    )

    return output_path


def create_displacement_products(
    out_paths: OutputPathsWithCorrections,
    out_dir: Path,
    date_to_cslc_files: Mapping[tuple[datetime], list[Path]],
    pge_runconfig: RunConfig,
    dolphin_config: DisplacementWorkflow,
    processing_start_datetime: datetime,
    reference_point: ReferencePoint | None = None,
    los_east_file: Path | None = None,
    los_north_file: Path | None = None,
    near_far_incidence_angles: tuple[float, float] = (30.0, 45.0),
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
    processing_start_datetime : datetime.datetime
        The processing start datetime.
    reference_point : ReferencePoint, optional
        Named tuple with (row, col, lat, lon) of selected reference pixel
        recorded from dolphin after unwrapping.
        If none, leaves product attributes empty.
    los_east_file : Path, optional
        Path to the east component of line of sight unit vector
    los_north_file : Path, optional
        Path to the north component of line of sight unit vector
    near_far_incidence_angles : tuple[float, float]
        Tuple of near range incidence angle, far range incidence angle.
        If not specified, uses approximate Sentinel-1 values of (30.0, 45.0)
    water_mask : Path, optional
        Binary water mask to use for output product.
        If provided, is used in the `recommended_mask`.
    max_workers : int
        Number of parallel products to process.
        Default is 3.

    """
    assert out_paths.timeseries_paths is not None
    if out_paths.ionospheric_corrections is not None:
        iono_files = out_paths.ionospheric_corrections
    else:
        iono_files = [None] * len(out_paths.timeseries_paths)
    residual_files = out_paths.timeseries_residual_paths or [None] * len(
        out_paths.timeseries_paths
    )

    files = [
        ProductFiles(
            unwrapped=unw,
            conncomp=cc,
            temp_coh=out_paths.stitched_temp_coh_file,
            correlation=cor,
            ps_mask=out_paths.stitched_ps_file,
            shp_counts=out_paths.stitched_shp_count_file,
            ionosphere=iono,
            similarity=out_paths.stitched_similarity_file,
            residual=resid,
            water_mask=water_mask,
        )
        for unw, cc, cor, resid, iono in zip(
            out_paths.timeseries_paths,
            out_paths.conncomp_paths,
            out_paths.stitched_cor_paths,
            residual_files,
            iono_files,
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
                repeat(processing_start_datetime),
                repeat(reference_point),
                repeat(los_east_file),
                repeat(los_north_file),
                repeat(near_far_incidence_angles),
            )
        )


def _create_nodata_mask(filename: PathOrStr, output_filename: PathOrStr) -> None:
    # Mark nodata as 0/False, valid as 1/True
    mask = io.load_gdal(filename, masked=True).filled(0) != 0
    # A valid output has to be valid in the mask, AND not be a `nodata`
    io.write_arr(
        like_filename=filename, arr=mask, nodata=255, output_name=output_filename
    )
