from __future__ import annotations

import logging
from collections.abc import Sequence
from math import pi
from multiprocessing import get_context
from pathlib import Path

from dolphin import PathOrStr, io
from dolphin.constants import SENTINEL_1_WAVELENGTH
from dolphin.interferogram import estimate_correlation_from_phase
from dolphin.unwrap import grow_conncomp_snaphu
from dolphin.utils import full_suffix
from dolphin.workflows.config import UnwrapOptions
from tqdm.contrib.concurrent import thread_map

logger = logging.getLogger(__name__)


def _update_snaphu_conncomps(
    timeseries_paths: Sequence[Path],
    stitched_cor_paths: Sequence[Path],
    mask_filename: PathOrStr,
    unwrap_options: UnwrapOptions,
    nlooks: int,
    max_workers: int = 2,
) -> list[Path]:
    """Recompute connected components from SNAPHU after a timeseries inversion.

    `timeseries_paths` contains the post-inversion rasters, one per secondary date.

    Parameters
    ----------
    timeseries_paths : list[Path]
        list of paths to the timeseries files.
    stitched_cor_paths : list[Path]
        list of paths to the pseuedo-correlation rasters.
    mask_filename : PathOrStr
        Path to a binary mask matching shape of `timeseries_paths`.
    unwrap_options : [dolphin.workflows.config.UnwrapOptions][]
        Configuration object containing unwrapping options.
    nlooks : int
        Effective number of looks used to make correlation.
    max_workers : int
        Number of parallel files to process.
        Default is 2.

    Returns
    -------
    list[Path]
        list of updated connected component paths.

    """
    args_list = [
        (unw_f, cor_f, nlooks, mask_filename, unwrap_options)
        for unw_f, cor_f in zip(timeseries_paths, stitched_cor_paths)
    ]

    mp_context = get_context("spawn")
    with mp_context.Pool(max_workers) as pool:
        return list(pool.map(_regrow, args_list))


def _update_spurt_conncomps(
    timeseries_paths: Sequence[Path],
    conncomp_paths: Sequence[Path],
) -> list[Path]:
    """Recompute connected components from spurt after a timeseries inversion.

    Since spurt uses one file computed from `ndimage.label`, we just need to
    rename an example to be the same as the timeseries rasters.

    Parameters
    ----------
    timeseries_paths : list[Path]
        list of paths to the timeseries files.
    conncomp_paths : list[Path]
        list of connected component paths from the spurt unwrapping

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


def _create_correlation_images(
    ts_filenames: Sequence[PathOrStr],
    window_size: tuple[int, int] = (11, 11),
    keep_bits: int = 8,
    num_workers: int = 3,
) -> list[Path]:
    path_tuples: list[tuple[Path, Path]] = []
    output_paths: list[Path] = []
    for fn in ts_filenames:
        ifg_path = Path(fn)
        cor_path = ifg_path.with_suffix(".cor.tif")
        output_paths.append(cor_path)
        if cor_path.exists():
            logger.info(f"Skipping existing interferometric correlation for {ifg_path}")
            continue
        path_tuples.append((ifg_path, cor_path))

    def process_ifg(args):
        ifg_path, cor_path = args
        logger.debug(f"Estimating correlation for {ifg_path}, writing to {cor_path}")
        disp = io.load_gdal(ifg_path)
        disp_rad = disp * (-4 * pi) / SENTINEL_1_WAVELENGTH

        cor = estimate_correlation_from_phase(disp_rad, window_size=window_size)
        if keep_bits:
            io.round_mantissa(cor, keep_bits=keep_bits)

        io.write_arr(
            arr=cor,
            output_name=cor_path,
            like_filename=ifg_path,
            driver="GTiff",
        )

    thread_map(
        process_ifg,
        path_tuples,
        max_workers=num_workers,
        desc="Estimating correlations",
    )

    return output_paths


def _regrow(args: tuple[Path, Path, int, PathOrStr, UnwrapOptions]) -> Path:
    unw_f, cor_f, nlooks, mask_filename, unwrap_options = args
    new_path = grow_conncomp_snaphu(
        unw_filename=unw_f,
        corr_filename=cor_f,
        nlooks=nlooks,
        mask_filename=mask_filename,
        cost=unwrap_options.snaphu_options.cost,
        scratchdir=unwrap_options._directory / "scratch2",
    )
    return new_path
