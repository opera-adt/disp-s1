from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import shapely.ops
from dolphin import PathOrStr, io
from dolphin.constants import SENTINEL_1_WAVELENGTH
from dolphin.interferogram import estimate_correlation_from_phase
from dolphin.unwrap import grow_conncomp_snaphu
from dolphin.utils import full_suffix
from dolphin.workflows.config import UnwrapOptions
from osgeo import gdal
from shapely.geometry import LinearRing, MultiPolygon, Polygon
from tqdm.contrib.concurrent import thread_map

logger = logging.getLogger(__name__)

gdal.UseExceptions()

METERS_TO_RADIANS = (-4 * np.pi) / SENTINEL_1_WAVELENGTH


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
        (idx, unw_f, cor_f, nlooks, mask_filename, unwrap_options)
        for idx, (unw_f, cor_f) in enumerate(zip(timeseries_paths, stitched_cor_paths))
    ]

    mp_context = get_context("spawn")
    with mp_context.Pool(max_workers) as pool:
        return list(pool.map(_regrow, args_list))


def _regrow(args: tuple[int, Path, Path, int, PathOrStr, UnwrapOptions]) -> Path:
    scratch_idx, unw_f, cor_f, nlooks, mask_filename, unwrap_options = args
    new_path = grow_conncomp_snaphu(
        unw_filename=unw_f,
        corr_filename=cor_f,
        nlooks=nlooks,
        mask_filename=mask_filename,
        cost=unwrap_options.snaphu_options.cost,
        scratchdir=unwrap_options._directory / f"scratch{scratch_idx}",
    )
    return new_path


def _update_spurt_conncomps(
    timeseries_paths: Sequence[Path],
    template_conncomp_path: Path,
) -> list[Path]:
    """Recompute connected components from spurt after a timeseries inversion.

    Since spurt uses one file computed from `ndimage.label`, we just need to
    rename an example to be the same as the timeseries rasters.

    Parameters
    ----------
    timeseries_paths : list[Path]
        list of paths to the timeseries files.
    template_conncomp_path : Path
        One connected component paths from the spurt unwrapping.
        Only one is needed while spurt uses only a single mask for pixel selection.

    Returns
    -------
    list[Path]
        list of updated connected component paths.

    """
    new_conncomp_paths: list[Path] = []
    for ts_p in timeseries_paths:
        new_name = template_conncomp_path.parent / str(ts_p.name).replace(
            full_suffix(ts_p), full_suffix(template_conncomp_path)
        )
        try:
            shutil.copy(template_conncomp_path, new_name)
        except shutil.SameFileError:
            pass
        new_conncomp_paths.append(new_name)
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
        disp_rad = disp * METERS_TO_RADIANS

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


def extract_footprint(raster_path: PathOrStr, simplify_tolerance: float = 0.01) -> str:
    """Extract a simplified footprint from a raster file.

    This function opens a raster file, extracts its footprint, simplifies it,
    and returns the a Polygon from the exterior ring as a WKT string.

    Parameters
    ----------
    raster_path : str
        Path to the input raster file.
    simplify_tolerance : float, optional
        Tolerance for simplification of the footprint geometry.
        Default is 0.01.

    Returns
    -------
    str
        WKT string representing the simplified exterior footprint
        in EPSG:4326 (lat/lon) coordinates.

    Notes
    -----
    This function uses GDAL to open the raster and extract the footprint,
    and Shapely to process the geometry.

    """
    from os import fspath

    import shapely
    from osgeo import gdal

    # Extract the footprint as WKT string (don't save)
    wkt = gdal.Footprint(
        None,
        fspath(raster_path),
        format="WKT",
        dstSRS="EPSG:4326",
        simplify=simplify_tolerance,
    )

    # Convert WKT to Shapely geometry, extract exterior, and convert back to Polygon WKT
    in_multi = shapely.from_wkt(wkt)

    # This may have holes; get the exterior
    # Largest polygon should be first in MultiPolygon returned by GDAL
    footprint = shapely.Polygon(in_multi.geoms[0].exterior)
    # Split on antimeridian and return the WKT string
    return split_on_antimeridian(footprint).wkt


def split_on_antimeridian(polygon: Polygon) -> MultiPolygon:
    """Split `polygon` if it crosses the antimeridian (180°).

    Source:
    https://github.com/nasa/opera-sds-pcm/blob/a5a3db25be462e7955e5de06d6f9d1d8236a1ef2/util/geo_util.py#L265
    (where it is `check_dateline`, as it is in isce3)

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Input polygon.

    Returns
    -------
    MultiPolygon
        A MultiPolygon containing 1 or 2 `.geoms`:
        The input polygon if it didn't cross the antimeridian, or
        two polygons otherwise (one on either side of the antimeridian).

    """
    x_min, _, x_max, _ = polygon.bounds

    # Check antimeridian crossing
    if (x_max - x_min > 180.0) or (x_min <= 180.0 <= x_max):
        antimeridian = shapely.wkt.loads("LINESTRING( 180.0 -90.0, 180.0 90.0)")

        # build new polygon with all longitudes between 0 and 360
        x, y = polygon.exterior.coords.xy
        new_x = (k + (k <= 0.0) * 360 for k in x)
        new_ring = LinearRing(zip(new_x, y))

        # Split input polygon
        # (https://gis.stackexchange.com/questions/232771/splitting-polygon-by-linestring-in-geodjango_)
        merged_lines = shapely.ops.linemerge([antimeridian, new_ring])
        border_lines = shapely.ops.unary_union(merged_lines)
        decomp = shapely.ops.polygonize(border_lines)

        polys = list(decomp)

        for polygon_count in range(len(polys)):
            x, y = polys[polygon_count].exterior.coords.xy
            # if there are no longitude values above 180, continue
            if not any(k > 180 for k in x):
                continue

            # otherwise, wrap longitude values down by 360 degrees
            x_wrapped_minus_360 = np.asarray(x) - 360
            polys[polygon_count] = Polygon(zip(x_wrapped_minus_360, y))

    else:
        # If antimeridian is not crossed, treat input polygon as list
        polys = [polygon]

    return MultiPolygon(polys)


def _convert_meters_to_radians(timeseries_paths: Sequence[Path]) -> list[Path]:
    """Copy over .tif, rescaling units from meters to radians."""
    output_files: list[Path] = []
    for in_path in timeseries_paths:
        out_path = in_path.with_suffix(".radians.tif")
        io.write_arr(
            arr=METERS_TO_RADIANS * io.load_gdal(in_path),
            like_filename=in_path,
            output_name=out_path,
        )
        output_files.append(out_path)
    return output_files
