import logging
from collections.abc import Sequence
from math import pi
from pathlib import Path

from dolphin import PathOrStr, io
from dolphin.constants import SENTINEL_1_WAVELENGTH
from dolphin.interferogram import estimate_correlation_from_phase
from dolphin.unwrap import grow_conncomp_snaphu
from dolphin.workflows.config import UnwrapOptions
from tqdm.contrib.concurrent import thread_map

logger = logging.getLogger(__name__)


def _create_correlation_images(
    ts_filenames: Sequence[PathOrStr],
    window_size: tuple[int, int] = (11, 11),
    keep_bits: int = 8,
    num_workers: int = 3,
):
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
        disp_rad = disp
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
