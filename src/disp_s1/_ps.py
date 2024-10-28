from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import dolphin.ps
import numpy as np
from dolphin import io, masking
from dolphin._types import PathOrStr
from dolphin.io import VRTStack
from dolphin.workflows._utils import _create_burst_cfg, _remove_dir_if_empty
from dolphin.workflows.config import DisplacementWorkflow
from dolphin.workflows.wrapped_phase import _get_mask
from opera_utils import group_by_burst

logger = logging.getLogger(__name__)


def precompute_ps(cfg: DisplacementWorkflow):
    try:
        grouped_slc_files = group_by_burst(cfg.cslc_file_list)
    except ValueError as e:
        # Make sure it's not some other ValueError
        if "Could not parse burst id" not in str(e):
            raise
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
        cfg.create_dir_tree()
        b = next(iter(grouped_slc_files.keys()))
        wrapped_phase_cfgs = [(b, cfg)]


def run_burst_ps(cfg: DisplacementWorkflow):
    input_file_list = cfg.cslc_file_list
    if not input_file_list:
        msg = "No input files found"
        raise ValueError(msg)

    subdataset = cfg.input_options.subdataset
    # Mark any files beginning with "compressed" as compressed
    is_compressed = ["compressed" in str(f).lower() for f in input_file_list]

    non_compressed_slcs = [
        f for f, is_comp in zip(input_file_list, is_compressed) if not is_comp
    ]
    vrt_stack = VRTStack(
        non_compressed_slcs,
        subdataset=subdataset,
        outfile=cfg.work_directory / "non_compressed_slc_stack.vrt",
    )

    layover_shadow_mask = (
        cfg.layover_shadow_mask_files[0] if cfg.layover_shadow_mask_files else None
    )
    mask_filename = _get_mask(
        output_dir=cfg.work_directory,
        output_bounds=cfg.output_options.bounds,
        output_bounds_wkt=cfg.output_options.bounds_wkt,
        output_bounds_epsg=cfg.output_options.bounds_epsg,
        like_filename=vrt_stack.outfile,
        layover_shadow_mask=layover_shadow_mask,
        cslc_file_list=non_compressed_slcs,
    )
    nodata_mask = masking.load_mask_as_numpy(mask_filename) if mask_filename else None

    output_file_list = [
        cfg.ps_options._output_file,
        cfg.ps_options._amp_mean_file,
        cfg.ps_options._amp_dispersion_file,
    ]
    ps_output = cfg.ps_options._output_file
    if all(f.exists() for f in output_file_list):
        logger.info(f"Skipping making existing PS files {output_file_list}")
        return output_file_list

    logger.info(f"Creating persistent scatterer file {ps_output}")
    # dispersions: np.ndarray, means: np.ndarray, N: ArrayLike | Sequence
    dolphin.ps.create_ps(
        reader=vrt_stack,
        output_file=output_file_list[0],
        output_amp_mean_file=output_file_list[1],
        output_amp_dispersion_file=output_file_list[2],
        like_filename=vrt_stack.outfile,
        amp_dispersion_threshold=cfg.ps_options.amp_dispersion_threshold,
        nodata_mask=nodata_mask,
        block_shape=cfg.worker_settings.block_shape,
    )

    compressed_slc_files = [
        f for f, is_comp in zip(input_file_list, is_compressed) if is_comp
    ]
    logger.info(f"Combining existing means/dispersions from {compressed_slc_files}")
    run_combine(
        cfg.ps_options._amp_mean_file,
        cfg.ps_options._amp_dispersion_file,
        compressed_slc_files,
        num_slc=len(non_compressed_slcs),
        subdataset=subdataset,
    )


def run_combine(
    cur_mean: Path,
    cur_dispersion: Path,
    compressed_slc_files: list[PathOrStr],
    num_slc: int,
    subdataset: str = "/data/VV",
):
    reader_compslc = io.HDF5StackReader.from_file_list(
        file_list=compressed_slc_files,
        dset_names=subdataset,
        nodata=np.nan,
    )
    reader_compslc_dispersion = io.HDF5StackReader.from_file_list(
        file_list=compressed_slc_files,
        dset_names="/data/amplitude_dispersion",
        nodata=np.nan,
    )
    reader_mean = io.RasterReader(cur_mean, band=1)
    reader_dispersion = io.RasterReader(cur_dispersion, band=1)

    # writer_mean = io.BackgroundRasterWriter(filename="combined_mean.tif")
    # writer_dispersion = io.BackgroundRasterWriter(filename="combined_dispersions.tif")

    def read_and_combine(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[np.ndarray, slice, slice]:
        reader_compslc, reader_compslc_dispersion, reader_mean, reader_dispersion = (
            readers
        )
        compslc_mean = np.abs(reader_compslc[:, rows, cols])
        if compslc_mean.ndim == 2:
            compslc_mean = compslc_mean[np.newaxis]
        compslc_dispersion = reader_compslc_dispersion[:, rows, cols]
        if compslc_dispersion.ndim == 2:
            compslc_dispersion = compslc_dispersion[np.newaxis]

        mean = reader_mean[rows, cols]
        dispersion = reader_dispersion[rows, cols]

        # Fit a line to each pixel with weighted least squares
        dispersions = np.vstack([compslc_dispersion, dispersion])

        means = np.vstack([compslc_mean, mean])
        # Increase the weights from older to newer.
        N = np.linspace(0, 1, num=len(means)) * num_slc
        return (
            dolphin.ps.combine_amplitude_dispersions(
                dispersions=dispersions,
                means=means,
                N=N,
            ),
            rows,
            cols,
        )

    out_paths = ["combined_dispersion.tif", "combined_mean.tif"]
    readers = reader_compslc, reader_compslc_dispersion, reader_mean, reader_dispersion
    writer = io.BackgroundStackWriter(out_paths, like_filename=cur_mean)
    io.process_blocks(
        readers=readers,
        writer=writer,
        func=read_and_combine,
        block_shape=(512, 512),
    )

    writer.notify_finished()
    return out_paths
