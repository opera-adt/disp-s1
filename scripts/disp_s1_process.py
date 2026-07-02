"""
Standalone DISP-S1 processing pipeline (Sentinel-1), driven from the CLI.

Given a directory of CSLC/GSLC bursts, this runs the full local pipeline for an
arbitrary site/track/frame:

  1. Group CSLCs by burst, create ministacks, run dolphin/disp_s1 per batch,
     carrying compressed SLCs forward between ministacks.
  2. Reformat per-ministack .nc outputs into a single rebased zarr stack.
  3. Estimate linear LOS velocity and append to zarr.
  4. Append extra layers (amplitude, CRLB, stack quality metrics).

All site-specific values (input dir, work dir, frame id, spatial extent,
ministack size, ...) are command-line arguments; nothing is hardcoded to a
particular acquisition. Run ``python disp_s1_process.py --help`` for options.

Example
-------
    python disp_s1_process.py \\
        --cslc-dir /path/to/gslcs/t161_343970_iw2 \\
        --work-dir /path/to/results \\
        --frame-id 42997 \\
        --extent "4.3561,51.0951 : 4.373,51.1031" \\
        --ministack-size 15 --buffer 500
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import re
import shutil
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
import zarr
from pyproj import CRS, Transformer
from rasterio.windows import from_bounds

from dolphin._log import setup_logging
from dolphin.stack import CompressedSlcPlan
from dolphin.timeseries import estimate_velocity, datetime_to_float
from dolphin.utils import get_max_memory_usage
from dolphin.workflows import PreprocessOptions, SnaphuOptions
from dolphin.workflows.config import (
    DisplacementWorkflow,
    HalfWindow,
    InputOptions,
    InterferogramNetwork,
    OutputOptions,
    PhaseLinkingOptions,
    PsOptions,
    Strides,
    UnwrapOptions,
)
from dolphin.workflows.corrections import CorrectionOptions
from dolphin.workflows.displacement import run as run_displacement

from disp_s1 import __version__
from disp_s1 import main as disp_s1_main
from disp_s1._masking import create_mask_from_distance
from disp_s1._ps import precompute_ps
from disp_s1.main import (
    OutputPathsWithCorrections,
    _assert_no_duplicate_dates,
    _filter_before_last_processed,
    create_products,
)
from disp_s1.pge_runconfig import (
    AlgorithmParameters,
    DynamicAncillaryFileGroup,
    InputFileGroup,
    PrimaryExecutable,
    ProcessingMode,
    ProductPathGroup,
    RunConfig,
    StaticAncillaryFileGroup,
)

from opera_utils import OPERA_DATASET_NAME, group_by_burst
from opera_utils.disp._enums import (
    CorrectionDataset,
    DisplacementDataset,
    QualityDataset,
    ReferenceMethod,
)
from opera_utils.disp._netcdf import create_virtual_stack
from opera_utils.disp._rebase import NaNPolicy, create_rebased_displacement
from opera_utils.disp._reference import _get_reference_row_col, get_reference_values
from opera_utils.disp._reformat import (
    _get_zarr_encoding,
    _get_transform,
    _to_shard_dict,
    _write_rebased_stack,
    combine_quality_masks,
)
from opera_utils.disp._utils import _clamp_chunk_dict, _get_netcdf_encoding, round_mantissa
from opera_utils import group_by_date

# ── DEFAULTS ──────────────────────────────────────────────────────────────────
# Site-specific values now come from the CLI (see build_parser / main).
DEFAULT_MS_SIZE = 15        # acquisitions per ministack
DEFAULT_BUFFER = 500.0      # metres of padding around the requested extent
DEFAULT_MAX_COMP = 5        # compressed SLCs carried forward per burst (historical)
DEFAULT_FORWARD_WINDOW = 5  # real SLCs per forward run: n-4, n-3, n-2, n-1, n
DEFAULT_GSLC_GLOB = "t*.h5"  # pattern matching (G)SLC HDF5 files, any burst id

# Keep JAX/XLA memory modest so it plays nice on shared GPUs.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

# ── HELPERS ───────────────────────────────────────────────────────────────────

def make_cfg(
    cslc_files: list[Path],
    work_dir: Path,
    comp_slc_files: list[Path],
    amp_disp_files: list[Path],
    amp_mean_files: list[Path],
    epsg=None,
    bounds: list = None,
    gpu: bool = False,
    compressed_slc_plan: CompressedSlcPlan = CompressedSlcPlan.LAST_PER_MINISTACK,
    output_reference_idx: int = None,
    ministack_size: int = DEFAULT_MS_SIZE,
) -> DisplacementWorkflow:
    return DisplacementWorkflow(
        cslc_file_list=comp_slc_files + cslc_files,
        input_options=InputOptions(subdataset=OPERA_DATASET_NAME),
        work_directory=work_dir,
        amplitude_dispersion_files=amp_disp_files,
        amplitude_mean_files=amp_mean_files,
        phase_linking=PhaseLinkingOptions(
            ministack_size=ministack_size,
            max_num_compressed=5,
            output_reference_idx=output_reference_idx,
            half_window=HalfWindow(y=3, x=7),
            use_evd=False,
            beta=0.0,
            zero_correlation_threshold=0.0,
            shp_method='glrt',
            shp_alpha=0.001,
            mask_input_ps=False,
            baseline_lag=None,
            compressed_slc_plan=compressed_slc_plan,
        ),
        interferogram_network=InterferogramNetwork(max_bandwidth=3),
        output_options=OutputOptions(
            strides=Strides(x=2, y=1),
            bounds=bounds,
            bounds_epsg=epsg,
        ),
        ps_options=PsOptions(amp_dispersion_threshold=0.25),
        unwrap_options=UnwrapOptions(
            run_unwrap=True,
            run_interpolation=True,
            preprocess_options=PreprocessOptions(
                alpha=0.5,
                max_radius=150,
                interpolation_cor_threshold=0.001,
                interpolation_similarity_threshold=0.4,
            ),
            snaphu_options=SnaphuOptions(
                ntiles=(1, 1),
                tile_overlap=(0, 0),
                n_parallel_tiles=1,
                single_tile_reoptimize=False,
            ),
        ),
        worker_settings={"gpu_enabled": gpu, "threads_per_worker": 8, "n_parallel_bursts": 1},
    )


def make_pge_runconfig(
    cfg: DisplacementWorkflow,
    frame_id: int,
    mode: ProcessingMode,
    work_dir: Path,
    output_dir: Path,
    save_compressed_slc: bool = True,
) -> RunConfig:
    alg_params_file = work_dir / "algorithm_parameters.yaml"
    algo_keys = set(AlgorithmParameters.model_fields.keys())
    alg_params = AlgorithmParameters(
        **cfg.model_dump(include=algo_keys),
        spatial_wavelength_cutoff=50.0,
    )
    alg_params.to_yaml(alg_params_file)

    return RunConfig(
        input_file_group=InputFileGroup(
            cslc_file_list=cfg.cslc_file_list,
            frame_id=frame_id,
        ),
        dynamic_ancillary_file_group=DynamicAncillaryFileGroup(
            algorithm_parameters_file=alg_params_file,
            mask_file=cfg.mask_file,
        ),
        static_ancillary_file_group=StaticAncillaryFileGroup(),
        primary_executable=PrimaryExecutable(
            product_type=f"DISP_S1_{mode.value.upper()}",
        ),
        product_path_group=ProductPathGroup(
            product_path=output_dir,
            scratch_path=work_dir,
            sas_output_path=output_dir,
            save_compressed_slc=save_compressed_slc,
        ),
        worker_settings=cfg.worker_settings,
    )


def extent_to_projected_bbox(extent_str: str, gslc_file: Path):
    """Convert 'minlon,minlat : maxlon,maxlat' extent to projected bbox from GSLC CRS."""
    left, right = extent_str.split(" : ")
    minlon, minlat = map(float, left.split(","))
    maxlon, maxlat = map(float, right.split(","))

    ds = xr.open_dataset(gslc_file, group="/data")
    crs_wkt = ds["projection"].attrs["spatial_ref"]
    crs = CRS.from_wkt(crs_wkt)
    epsg = crs.to_epsg()
    ds.close()

    transformer = Transformer.from_crs("EPSG:4326", crs_wkt, always_xy=True)
    minx, miny = transformer.transform(minlon, minlat)
    maxx, maxy = transformer.transform(maxlon, maxlat)

    return minx, miny, maxx, maxy, epsg


@contextlib.contextmanager
def _quiet_logging(log_file):
    """Redirect all dolphin/disp_s1 logging to file; suppress stderr."""
    target_loggers = [
        logging.getLogger(name)
        for name in ("dolphin", "disp_s1", "opera_utils", "root", "")
    ] + [logging.root]

    removed = []
    for lgr in target_loggers:
        for hdlr in list(lgr.handlers):
            if isinstance(hdlr, logging.StreamHandler) and not isinstance(
                hdlr, logging.FileHandler
            ):
                lgr.removeHandler(hdlr)
                removed.append((lgr, hdlr))

    old_stderr = sys.stderr
    sys.stderr = open(log_file, "a")
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr
        for lgr, hdlr in removed:
            lgr.addHandler(hdlr)


def _ensure_correction_options(cfg):
    if not hasattr(cfg, "correction_options"):
        cfg.correction_options = CorrectionOptions()


def run_no_corrections(cfg: DisplacementWorkflow, pge_runconfig: RunConfig, debug: bool = False):
    """Run disp_s1 without the corrections workflow (no geometry files available)."""
    cfg.work_directory.mkdir(exist_ok=True, parents=True)
    if cfg.log_file is None:
        cfg.log_file = cfg.work_directory / "dolphin.log"
    setup_logging(logger_name="disp_s1", debug=debug, filename=cfg.log_file)
    with _quiet_logging(cfg.log_file):
        processing_start_datetime = datetime.now(timezone.utc)

        _assert_no_duplicate_dates(cfg.cslc_file_list)

        is_forward = pge_runconfig.primary_executable.product_type == "DISP_S1_FORWARD"
        if is_forward:
            date_to_files = group_by_date(cfg.cslc_file_list, date_idx=0)
            datetimes_present = list(date_to_files.keys())
            *_, (second_to_last_date,), (_last_date,) = datetimes_present
            second_to_last_date = second_to_last_date.replace(tzinfo=None)
            if pge_runconfig.input_file_group.last_processed is None:
                pge_runconfig.input_file_group.last_processed = (
                    second_to_last_date + timedelta(days=1)
                )

        if pge_runconfig.dynamic_ancillary_file_group.mask_file:
            water_binary_mask = cfg.work_directory / "water_binary_mask.tif"
            create_mask_from_distance(
                water_distance_file=pge_runconfig.dynamic_ancillary_file_group.mask_file,
                output_file=water_binary_mask,
                land_buffer=1,
                ocean_buffer=1,
            )
            cfg.mask_file = water_binary_mask

        if any("compressed" in f.name.lower() for f in cfg.cslc_file_list):
            combined_dispersion_files, combined_mean_files = precompute_ps(cfg=cfg)
            cfg.amplitude_dispersion_files = combined_dispersion_files
            cfg.amplitude_mean_files = combined_mean_files
        else:
            cfg.ps_options.amp_dispersion_threshold = 0.15

        out_paths = run_displacement(cfg=cfg, debug=debug)

        assert out_paths.timeseries_paths is not None
        assert out_paths.timeseries_residual_paths is not None

        if is_forward:
            from dolphin.timeseries import _redo_reference
            final_ts_paths, final_residual_paths = _redo_reference(
                out_paths.timeseries_paths,
                out_paths.timeseries_residual_paths,
                second_to_last_date,
                bad_pixel_mask=np.ma.nomask,
            )
            out_paths.timeseries_paths = final_ts_paths
            out_paths.timeseries_residual_paths = final_residual_paths

        if last_processed := pge_runconfig.input_file_group.last_processed:
            out_paths = _filter_before_last_processed(out_paths, last_processed)

        _ensure_correction_options(cfg)

        out_paths_with_corr = OutputPathsWithCorrections(
            **asdict(out_paths),
            ionospheric_corrections=None,
        )

        create_products(
            out_paths=out_paths_with_corr,
            cfg=cfg,
            pge_runconfig=pge_runconfig,
            processing_start_datetime=processing_start_datetime,
        )

        max_mem = get_max_memory_usage(units="GB")
        print(f"Max memory usage: {max_mem:.2f} GB")
        print(f"disp_s1 version: {__version__}")


# ── REFORMAT ──────────────────────────────────────────────────────────────────

_FNAME_RE = re.compile(r"(\d{8})_(\d{8})\.nc$")


def _parse_dates_from_filename(path: Path) -> tuple[datetime, datetime]:
    m = _FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse dates from filename: {path.name}")
    ref = datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
    sec = datetime.strptime(m.group(2), "%Y%m%d").replace(tzinfo=timezone.utc)
    return ref, sec


def reformat_stack_custom(
    input_files: list[Path],
    output_name: str,
    out_chunks: tuple[int, int, int] = (4, 256, 256),
    shard_factors: tuple[int, int, int] = (1, 4, 4),
    drop_vars: Sequence[str] | None = None,
    apply_solid_earth_corrections: bool = True,
    apply_ionospheric_corrections: bool = False,
    quality_datasets: Sequence[str] | None = ["recommended_mask"],
    quality_thresholds: Sequence[float] | None = [0.5],
    reference_method: ReferenceMethod = ReferenceMethod.HIGH_COHERENCE,
    reference_row: int | None = None,
    reference_col: int | None = None,
    reference_lon: float | None = None,
    reference_lat: float | None = None,
    reference_border_pixels: int = 3,
    reference_coherence_threshold: float = 0.7,
    process_chunk_size: tuple[int, int] = (2048, 2048),
    do_round: bool = True,
) -> None:
    """Reformat per-ministack .nc outputs into a unified zarr/netcdf stack."""
    start_time = time.time()

    if Path(output_name).suffix == ".nc":
        out_format = "h5netcdf"
    elif Path(output_name).suffix == ".zarr":
        out_format = "zarr"
    else:
        raise ValueError("Only .nc and .zarr output formats are supported")

    if drop_vars is None:
        drop_vars = []

    corrections: list[CorrectionDataset] = []
    if apply_solid_earth_corrections:
        corrections.append(CorrectionDataset.SOLID_EARTH_TIDE)
    if apply_ionospheric_corrections:
        corrections.append(CorrectionDataset.IONOSPHERIC_DELAY)

    out_chunk_dict = dict(zip(["time", "y", "x"], out_chunks))
    out_shard_dict = _to_shard_dict(out_chunks, shard_factors)

    input_files = sorted(input_files, key=lambda p: _parse_dates_from_filename(p))

    reference_datetimes = pd.DatetimeIndex(
        [_parse_dates_from_filename(p)[0].replace(tzinfo=None) for p in input_files]
    )

    process_chunk_dict = {"time": 1, "y": process_chunk_size[0], "x": process_chunk_size[1]}

    ds = xr.open_mfdataset(input_files, engine="h5netcdf", chunks=process_chunk_dict)
    ds_corrections = xr.open_mfdataset(
        input_files, engine="h5netcdf", group="corrections", chunks=process_chunk_dict
    )
    ds_bperp = ds_corrections["perpendicular_baseline"].isel(
        y=ds_corrections.y.shape[0] // 2, x=ds_corrections.x.shape[0] // 2
    )

    ds = ds.drop_vars(drop_vars, errors="ignore")

    all_vars = list(ds.data_vars)
    minimal_vars = ["spatial_ref", "reference_time", "water_mask"]
    ds_minimal = ds.drop_vars([v for v in all_vars if v not in minimal_vars])
    ds_minimal["water_mask"] = ds_minimal["water_mask"].isel(time=0)
    ds_minimal["perpendicular_baseline"] = ds_bperp

    if out_format == "zarr":
        encoding = _get_zarr_encoding(ds_minimal, out_chunks, add_coords=True)
        ds_minimal.chunk(out_shard_dict).to_zarr(output_name, encoding=encoding, mode="w", consolidated=False)
    else:
        encoding = _get_netcdf_encoding(ds_minimal, out_chunks)
        ds_minimal.to_netcdf(output_name, engine="h5netcdf", encoding=encoding, mode="w")
    print(f"Wrote minimal dataset in {time.time() - start_time:.1f}s")

    QUALITY_DATASETS = list(QualityDataset)
    remaining_dsets = [
        str(d) for d in QUALITY_DATASETS
        if d != "water_mask" and str(d) not in drop_vars
    ]
    ds_remaining = ds[remaining_dsets].chunk({
        "time": out_shard_dict["time"],
        "y": process_chunk_dict["y"],
        "x": process_chunk_dict["x"],
    })
    da_temp_coh = ds.temporal_coherence[::15]
    avg_coherence = da_temp_coh.shape[0] / (1.0 / da_temp_coh).sum(dim="time", skipna=False, min_count=1)
    ds_remaining["average_temporal_coherence"] = xr.DataArray(
        avg_coherence, dims=("y", "x"), coords={"y": ds.y, "x": ds.x}
    )
    for var in ds_remaining.data_vars:
        d = ds_remaining[var]
        if do_round and np.issubdtype(d.dtype, np.floating):
            d.data = round_mantissa(d.data, keep_bits=7)

    print(f"Writing remaining variables: {list(ds_remaining.data_vars)}")
    if out_format == "zarr":
        encoding = _get_zarr_encoding(ds_remaining, out_chunks)
        ds_remaining.to_zarr(output_name, encoding=encoding, mode="a", consolidated=False)
    else:
        create_virtual_stack(input_files=input_files, output=output_name, dataset_names=remaining_dsets)
        encoding = _get_netcdf_encoding(ds_remaining[["average_temporal_coherence"]], out_chunks)
        ds_remaining[["average_temporal_coherence"]].to_netcdf(output_name, engine="h5netcdf", encoding=encoding, mode="a")
    print(f"Wrote remaining at {time.time() - start_time:.1f}s")

    if reference_method == ReferenceMethod.HIGH_COHERENCE:
        good_pixel_mask = avg_coherence > reference_coherence_threshold
        ref_row = ref_col = None
    elif reference_method == ReferenceMethod.POINT:
        transform = _get_transform(ds)
        crs = CRS.from_wkt(ds.spatial_ref.crs_wkt)
        ref_row, ref_col = _get_reference_row_col(
            row=reference_row, col=reference_col,
            lon=reference_lon, lat=reference_lat,
            crs=crs, transform=transform,
        )
        good_pixel_mask = np.asarray(ds.water_mask) == 1
    elif reference_method in (ReferenceMethod.BORDER, ReferenceMethod.MEDIAN):
        good_pixel_mask = np.asarray(ds.water_mask) == 1
        ref_row = ref_col = None
    else:
        raise ValueError(f"Unknown ReferenceMethod {reference_method}")

    correction_names = [str(c).split("/")[-1] for c in corrections]
    _write_rebased_stack(
        ds, output_name, out_chunks=out_chunks,
        reference_datetimes=reference_datetimes,
        data_var=DisplacementDataset.DISPLACEMENT,
        reference_method=reference_method,
        reference_row=ref_row, reference_col=ref_col,
        border_pixels=reference_border_pixels,
        good_pixel_mask=good_pixel_mask,
        out_format=out_format,
        ds_corrections=ds_corrections[correction_names] if corrections else None,
        quality_datasets=quality_datasets,
        quality_thresholds=quality_thresholds,
        process_chunk_size=process_chunk_size,
        shard_factors=shard_factors,
        do_round=do_round,
    )
    print(f"Wrote displacement at {time.time() - start_time:.1f}s")

    if str(DisplacementDataset.SHORT_WAVELENGTH) in ds.data_vars:
        _write_rebased_stack(
            ds, output_name, out_chunks=out_chunks,
            reference_datetimes=reference_datetimes,
            data_var=DisplacementDataset.SHORT_WAVELENGTH,
            out_format=out_format,
            process_chunk_size=process_chunk_size,
            shard_factors=shard_factors,
            do_round=do_round,
        )
        print(f"Wrote short_wavelength_displacement at {time.time() - start_time:.1f}s")

    if out_format == "zarr":
        zarr.consolidate_metadata(output_name)
        print(f"Consolidated zarr metadata at {time.time() - start_time:.1f}s")


# ── VELOCITY ──────────────────────────────────────────────────────────────────

def add_velocity_to_zarr(zarr_path: str, weight_by_coherence: bool = True) -> None:
    """Estimate linear LOS velocity from displacement and append to zarr."""
    ds = xr.open_zarr(zarr_path)
    da = ds["displacement"]

    dates = da.time.values.astype("datetime64[s]").tolist()
    x_arr = datetime_to_float(dates)

    print("Loading displacement...")
    disp = np.array(da.values, dtype=np.float32)

    if weight_by_coherence and "average_temporal_coherence" in ds:
        coh = np.array(ds["average_temporal_coherence"].values, dtype=np.float32)
        weight_stack = np.broadcast_to(coh[np.newaxis], disp.shape).copy()
        weight_stack[~np.isfinite(disp)] = 0.0
    else:
        weight_stack = None

    disp = np.where(np.isfinite(disp), disp, 0.0)

    print("Estimating velocity with dolphin...")
    vel = np.array(estimate_velocity(x_arr, disp, weight_stack))

    ds_out = xr.Dataset({
        "velocity": xr.DataArray(
            vel.astype(np.float32), dims=("y", "x"),
            coords={"y": ds.y, "x": ds.x},
            attrs={"units": "m/yr", "long_name": "Linear LOS velocity"},
        ),
    })

    encoding = {"velocity": {"compressors": [], "chunks": (256, 256)}}
    ds_out.to_zarr(zarr_path, mode="a", consolidated=False, encoding=encoding)
    zarr.consolidate_metadata(zarr_path)
    print("Done — velocity written to zarr.")


# ── RECHUNK ───────────────────────────────────────────────────────────────────

def rechunk_zarr(zarr_path: str, out_path: str, chunks: dict = None) -> None:
    """Rechunk zarr store for efficient spatial access."""
    if chunks is None:
        chunks = {"time": 1, "y": 512, "x": 512}
    ds = xr.open_zarr(zarr_path)
    ds_rechunked = ds.chunk(chunks)

    encoding = {}
    for var in list(ds_rechunked.data_vars) + list(ds_rechunked.coords):
        da = ds_rechunked[var]
        if hasattr(da, "chunks") and da.chunks:
            encoding[var] = {"chunks": tuple(c[0] for c in da.chunks)}

    ds_rechunked.to_zarr(out_path, mode="w", encoding=encoding)
    print(f"Rechunked zarr written to {out_path}")


# ── EXTRA LAYERS ─────────────────────────────────────────────────────────────

# Sentinel-1 C-band wavelength (m) — used for CRLB phase→displacement conversion
S1_WAVELENGTH_M = 0.05546576


def _crop_tif(path: str | Path, xmin: float, ymin: float, xmax: float, ymax: float) -> np.ndarray:
    """Read and crop a GeoTIFF to the given bounding box, returning a 2-D float32 array.

    nodata (0) is replaced with NaN.
    """
    with rasterio.open(path) as src:
        win = from_bounds(xmin, ymin, xmax, ymax, src.transform)
        arr = src.read(1, window=win).astype(np.float32)
        nd = src.nodata
    if nd is not None:
        arr[arr == nd] = np.nan
    else:
        arr[arr == 0] = np.nan
    return arr


def _zarr_bbox(zarr_path: str) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) half-pixel-expanded bbox of the zarr store."""
    ds = xr.open_zarr(zarr_path)
    dx = abs(float(ds.x[1] - ds.x[0])) / 2
    dy = abs(float(ds.y[0] - ds.y[1])) / 2
    return (
        float(ds.x.min()) - dx,
        float(ds.y.min()) - dy,
        float(ds.x.max()) + dx,
        float(ds.y.max()) + dy,
    )


def _to_db(amp: np.ndarray) -> np.ndarray:
    """Linear amplitude → dB: 20*log10(amp). Zeros and NaN → NaN."""
    return np.where(amp > 0, 20.0 * np.log10(amp), np.nan).astype(np.float32)


def _write_2d(zarr_path: str, name: str, data: np.ndarray, y, x, attrs: dict, chunks=(256, 256)):
    da = xr.DataArray(data.astype(np.float32), dims=("y", "x"), coords={"y": y, "x": x}, attrs=attrs)
    enc = {name: {"compressors": [], "chunks": chunks}}
    da.to_dataset(name=name).to_zarr(zarr_path, mode="a", consolidated=False, encoding=enc)


def _write_3d(zarr_path: str, name: str, data: np.ndarray, time_coords, y, x, attrs: dict, chunks=(4, 256, 256)):
    da = xr.DataArray(
        data.astype(np.float32), dims=("time", "y", "x"),
        coords={"time": time_coords, "y": y, "x": x}, attrs=attrs,
    )
    enc = {name: {"compressors": [], "chunks": chunks}}
    da.to_dataset(name=name).to_zarr(zarr_path, mode="a", consolidated=False, encoding=enc)


def append_extra_layers(
    zarr_path: str,
    work_base: Path,
    gslc_dir: Path,
    gslc_glob: str = DEFAULT_GSLC_GLOB,
    wavelength: float = S1_WAVELENGTH_M,
) -> None:
    """Append amplitude, CRLB, and stack quality metrics to an existing zarr store.

    Layers added
    ------------
    3-D (time, y, x)  — time axis uses the ministack end-date from filename:
      mean_amplitude        — amplitude of each compressed SLC (|VV|, multilooked)
      amplitude_dispersion  — amplitude_dispersion stored inside each compressed SLC

    2-D (y, x)  — derived entirely from the zarr's existing variables:
      median_temporal_coherence    — nanmedian of temporal_coherence over time
      median_phase_similarity      — nanmedian of phase_similarity over time
      sum_inversion_residuals      — sum(|timeseries_inversion_residuals|) over time

    3-D (time, y, x)  — aligned to zarr's full time axis:
      crlb_rad             — Cramér–Rao lower bound, phase std (rad)
      crlb_displacement    — CRLB expressed as displacement uncertainty (m)
      amplitude            — per-date amplitude from original GSLC HDF5 files
    """
    ds = xr.open_zarr(zarr_path)
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    y_coords = ds.y.values
    x_coords = ds.x.values
    zarr_dates = {str(t)[:10].replace("-", ""): t for t in ds.time.values}
    sorted_dates = sorted(zarr_dates.keys())
    time_coords = np.array([zarr_dates[d] for d in sorted_dates])
    xmin, ymin, xmax, ymax = _zarr_bbox(zarr_path)
    print(f"zarr bbox: {xmin:.0f} {ymin:.0f} {xmax:.0f} {ymax:.0f}  ({ny}x{nx})")

    # ── 1. Mean amplitude & amplitude_dispersion from compressed SLC HDF5 ─────
    # Filename: compressed_{burst}_{date_last}_{date_first}_{date_last}.h5
    # Each compressed SLC covers [date_first, date_last].  We expand its 2-D
    # values to every zarr date that falls within that range, so the output
    # shares the same 315-date time axis as all other zarr variables.
    print("\n[1/6] Mean amplitude & amplitude_dispersion from compressed SLCs...")
    comp_slcs = sorted((work_base / "comp_slcs").glob("compressed*.h5"))
    if not comp_slcs:
        print("  WARNING: no compressed SLC HDF5 found, skipping.")
    else:
        def _comp_dates(f: Path) -> tuple[str, str]:
            """Return (date_first, date_last) from compressed SLC filename."""
            m = re.search(r"compressed_\S+?_\d{8}_(\d{8})_(\d{8})\.h5", f.name)
            return (m.group(1), m.group(2)) if m else ("", "")

        comp_info = [(d1, d2, f) for f in comp_slcs if "" not in (d := _comp_dates(f)) for d1, d2 in [d]]
        comp_info.sort(key=lambda t: t[0])

        # Crop indices (same for all files — shared grid)
        with h5py.File(comp_info[0][2]) as h:
            xc = h["data/x_coordinates"][:]   # 5 m spacing
            yc = h["data/y_coordinates"][:]   # 10 m spacing
        xi = np.where((xc >= xmin) & (xc <= xmax))[0]
        yi = np.where((yc >= ymin) & (yc <= ymax))[0]
        xi_s, xi_e = int(xi[0]), int(xi[-1]) + 1
        yi_s, yi_e = int(yi[0]), int(yi[-1]) + 1
        xi_e -= (xi_e - xi_s) % 2   # ensure even length for ×2 multilook

        # Build full-time-axis arrays: each date gets the value of the ministack
        # that contains it; last ministack wins if ranges overlap.
        mean_amp_3d  = np.full((len(sorted_dates), ny, nx), np.nan, dtype=np.float32)
        adisp_3d     = np.full((len(sorted_dates), ny, nx), np.nan, dtype=np.float32)

        for date_first, date_last, f in comp_info:
            with h5py.File(f) as h:
                vv    = h["data/VV"][yi_s:yi_e, xi_s:xi_e]
                adisp = h["data/amplitude_dispersion"][yi_s:yi_e, xi_s:xi_e]
            amp = np.abs(vv).astype(np.float32)
            amp_ml   = (amp[:,0::2]   + amp[:,1::2])   / 2
            adisp_ml = (adisp[:,0::2] + adisp[:,1::2]) / 2
            adisp_ml = adisp_ml.astype(np.float32)
            adisp_ml[adisp_ml <= 0] = np.nan

            for t_idx, d in enumerate(sorted_dates):
                if date_first <= d <= date_last:
                    mean_amp_3d[t_idx]  = _to_db(amp_ml)[:ny, :nx]
                    adisp_3d[t_idx]     = adisp_ml[:ny, :nx]

        _write_3d(zarr_path, "mean_amplitude", mean_amp_3d, time_coords, y_coords, x_coords,
                  {"long_name": "Compressed-SLC mean amplitude (|VV|, multilooked ×2 in x)",
                   "units": "dB  (20*log10(linear amplitude))",
                   "note": "constant within each ministack [date_first, date_last]"})
        _write_3d(zarr_path, "amplitude_dispersion", adisp_3d, time_coords, y_coords, x_coords,
                  {"long_name": "Amplitude dispersion (std/mean) from compressed SLC ministack",
                   "note": "constant within each ministack [date_first, date_last]"})
        print(f"  Written mean_amplitude, amplitude_dispersion  "
              f"({len(comp_info)} ministacks → {len(sorted_dates)} time steps)")

    # ── 2. Stack quality summary metrics (2-D, from zarr variables) ───────────
    print("\n[2/6] Stack quality summary metrics (from zarr)...")

    def _nanmedian_2d(da: xr.DataArray) -> np.ndarray:
        return np.nanmedian(da.values.astype(np.float32), axis=0)

    med_tc = _nanmedian_2d(ds.temporal_coherence)
    _write_2d(zarr_path, "median_temporal_coherence", med_tc, y_coords, x_coords,
              {"long_name": "Median temporal coherence across stack", "units": "0-1"})

    med_ps = _nanmedian_2d(ds.phase_similarity)
    _write_2d(zarr_path, "median_phase_similarity", med_ps, y_coords, x_coords,
              {"long_name": "Median phase similarity across stack", "units": "0-1"})

    sum_res = np.nansum(np.abs(ds.timeseries_inversion_residuals.values.astype(np.float32)), axis=0)
    sum_res[sum_res == 0] = np.nan
    _write_2d(zarr_path, "sum_inversion_residuals", sum_res, y_coords, x_coords,
              {"long_name": "Sum of absolute timeseries inversion residuals", "units": "rad"})

    print("  Written median_temporal_coherence, median_phase_similarity, sum_inversion_residuals")

    # ── 3. CRLB (time, y, x) ─────────────────────────────────────────────────
    print("\n[3/6] CRLB (phase std in rad + displacement uncertainty)...")

    # Collect one crlb file per date, preferring later batches when date repeats
    # (later batches use more data → better estimate)
    # Glob batch_*/*/linked_phase/ to be agnostic about the burst directory name.
    date_to_crlb: dict[str, Path] = {}
    for batch_dir in sorted(work_base.glob("batch_*/*/linked_phase/")):
        for ms_dir in sorted(batch_dir.iterdir()):
            if not ms_dir.is_dir():
                continue
            for cf in sorted((ms_dir / "crlb").glob("crlb_*.tif")):
                m = re.search(r"crlb_(\d{8})\.tif", cf.name)
                if m:
                    date_to_crlb[m.group(1)] = cf   # last batch wins

    # Build time-ordered arrays aligned to zarr time axis
    crlb_rad   = np.full((len(zarr_dates), ny, nx), np.nan, dtype=np.float32)
    for t_idx, (date_str, _) in enumerate(sorted(zarr_dates.items())):
        cf = date_to_crlb.get(date_str)
        if cf is None:
            continue
        # CRLB tif stores std in rad directly (dolphin _crlb_from_x returns sqrt(diag(Sigma)))
        arr = _crop_tif(cf, xmin, ymin, xmax, ymax)
        crlb_rad[t_idx] = arr[:ny, :nx]

    _write_3d(zarr_path, "crlb_rad", crlb_rad, time_coords, y_coords, x_coords,
              {"long_name": "Cramér-Rao lower bound — phase std", "units": "rad"})

    # Convert to displacement uncertainty: sigma_d = (wavelength / 4π) * sigma_phi
    factor = wavelength / (4 * np.pi)
    crlb_disp = crlb_rad * factor
    _write_3d(zarr_path, "crlb_displacement", crlb_disp, time_coords, y_coords, x_coords,
              {"long_name": "Cramér-Rao lower bound — displacement uncertainty", "units": "m",
               "wavelength_m": wavelength, "conversion": "wavelength / (4*pi) * crlb_rad"})
    print(f"  Written crlb_rad, crlb_displacement ({sum(v is not None for v in [date_to_crlb.get(d) for d in zarr_dates])} dates matched)")

    # ── 4. Per-date amplitude (dB) + stack amplitude dispersion from GSLC ──────
    print("\n[4/6] Per-date amplitude (dB) from GSLC HDF5...")

    date_to_gslc: dict[str, Path] = {}
    for f in sorted(gslc_dir.rglob(gslc_glob)):
        m = re.search(r"_(\d{8})\.h5$", f.name)
        if m:
            date_to_gslc[m.group(1)] = f

    ref_gslc = next(iter(date_to_gslc.values()))
    with h5py.File(ref_gslc) as h:
        xg = h["data/x_coordinates"][:]
        yg = h["data/y_coordinates"][:]
    xgi = np.where((xg >= xmin) & (xg <= xmax))[0]
    ygi = np.where((yg >= ymin) & (yg <= ymax))[0]
    xgi_s, xgi_e = int(xgi[0]), int(xgi[-1]) + 1
    ygi_s, ygi_e = int(ygi[0]), int(ygi[-1]) + 1
    xgi_e -= (xgi_e - xgi_s) % 2

    amplitude = np.full((len(sorted_dates), ny, nx), np.nan, dtype=np.float32)

    matched = 0
    for t_idx, date_str in enumerate(sorted_dates):
        gf = date_to_gslc.get(date_str)
        if gf is None:
            continue
        with h5py.File(gf) as h:
            vv = h["data/VV"][ygi_s:ygi_e, xgi_s:xgi_e]
        amp = np.abs(vv).astype(np.float32)
        amp_ml = (amp[:, 0::2] + amp[:, 1::2]) / 2
        amplitude[t_idx] = _to_db(amp_ml)[:ny, :nx]
        matched += 1

    _write_3d(zarr_path, "amplitude", amplitude, time_coords, y_coords, x_coords,
              {"long_name": "Per-date amplitude from GSLC (multilook ×2 in x)",
               "units": "dB  (20*log10(linear amplitude))"})
    print(f"  Written amplitude (dB) ({matched}/{len(sorted_dates)} dates matched)")

    # ── 5. Stack amplitude dispersion + mean amplitude dB from zarr (2-D) ──────
    # Re-open zarr to read the amplitude we just wrote (dB).
    # All stats computed in linear space; mean amplitude also stored in dB.
    print("\n[5/6] Stack amplitude dispersion + mean amplitude dB from zarr...")
    ds2 = xr.open_zarr(zarr_path)
    amp_db = ds2["amplitude"].values.astype(np.float32)        # (time, y, x)
    amp_lin = np.where(np.isfinite(amp_db), 10.0 ** (amp_db / 20.0), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        amp_mean_lin = np.nanmean(amp_lin, axis=0)
        amp_std      = np.nanstd(amp_lin, axis=0, ddof=1)
        stack_amp_disp = np.where(amp_mean_lin > 0, amp_std / amp_mean_lin, np.nan)
        amp_mean_db    = np.where(amp_mean_lin > 0, 20.0 * np.log10(amp_mean_lin), np.nan)

    _write_2d(zarr_path, "mean_amplitude_db", amp_mean_db.astype(np.float32),
              y_coords, x_coords,
              {"long_name": "Time-mean amplitude from full GSLC stack",
               "units": "dB  (20*log10(mean linear amplitude))",
               "note": f"mean computed in linear space, then converted to dB ({matched} dates)"})
    _write_2d(zarr_path, "stack_amplitude_dispersion", stack_amp_disp.astype(np.float32),
              y_coords, x_coords,
              {"long_name": "Amplitude dispersion from full GSLC stack (std/mean, linear)",
               "units": "dimensionless",
               "note": f"computed from zarr amplitude variable ({matched} dates)"})
    print(f"  Written mean_amplitude_db (mean={np.nanmean(amp_mean_db):.1f} dB)")
    print(f"  Written stack_amplitude_dispersion (mean={np.nanmean(stack_amp_disp):.3f})")

    # ── 6. Consolidate ────────────────────────────────────────────────────────
    print("\n[6/6] Consolidating zarr metadata...")
    zarr.consolidate_metadata(zarr_path)
    print("Done — extra layers appended and metadata consolidated.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

STAGES = ("process", "reformat", "velocity", "extra")


def run_processing(
    cslc_dir: Path,
    work_base: Path,
    frame_id: int,
    extent: str,
    *,
    ms_size: int,
    max_comp: int,
    buffer: float,
    gpu: bool,
    gslc_glob: str,
    forward_window: int = DEFAULT_FORWARD_WINDOW,
) -> None:
    """Stage 1: run disp_s1 per batch with compressed carry-forward.

    Full ministacks of ``ms_size`` real SLCs run in HISTORICAL mode (each saving
    one compressed SLC). Every remaining date that cannot fill a full ministack
    then runs in FORWARD mode as its own job, with a fixed input stack of
    ``1 compressed SLC + forward_window real SLCs`` ending at that date
    (e.g. n-4, n-3, n-2, n-1, n for forward_window=5). Each forward run yields
    exactly one product (date n), so no dates are dropped.
    """
    # ── Discover CSLCs ────────────────────────────────────────────────────────
    files = sorted(cslc_dir.rglob(gslc_glob))
    if not files:
        raise SystemExit(f"No files matching {gslc_glob!r} under {cslc_dir}")
    burst_to_cslc = group_by_burst(files)
    burst_ids = sorted(burst_to_cslc)
    n_bursts = len(burst_ids)
    reals_by_burst = {b: sorted(burst_to_cslc[b]) for b in burst_ids}
    # Number of acquisition dates common to all bursts (indices align across bursts).
    n_dates = min(len(v) for v in reals_by_burst.values())

    n_hist = n_dates // ms_size          # full historical ministacks
    hist_count = n_hist * ms_size        # real dates consumed by historical
    n_forward = n_dates - hist_count     # leftover dates → one forward run each
    print(
        f"{n_bursts} burst(s), {n_dates} date(s): {n_hist} historical ministack(s) "
        f"of {ms_size} + {n_forward} forward product(s), {len(files)} total CSLCs"
    )
    if n_hist == 0:
        raise SystemExit(
            f"Only {n_dates} date(s) (< ministack size {ms_size}); need at least one "
            "full historical ministack to bootstrap compressed SLCs for forward mode."
        )

    # ── Spatial subset ────────────────────────────────────────────────────────
    minx, miny, maxx, maxy, epsg = extent_to_projected_bbox(extent, files[0])
    print(f"bbox (EPSG:{epsg}): {minx:.0f} {miny:.0f} {maxx:.0f} {maxy:.0f}")
    bounds = [minx - buffer, miny - buffer, maxx + buffer, maxy + buffer]

    # ── Shared compressed SLC store ───────────────────────────────────────────
    comp_slc_dir = work_base / "comp_slcs"
    comp_slc_dir.mkdir(parents=True, exist_ok=True)

    amp_disp_files: list[Path] = []
    amp_mean_files: list[Path] = []

    def _pick_comp(k: int) -> list[Path]:
        """Latest ``k`` compressed SLCs per burst from the shared store."""
        stored = sorted(comp_slc_dir.glob("*.h5")) or sorted(comp_slc_dir.glob("*.tif"))
        picked: list[Path] = []
        for burst_files in group_by_burst(stored).values():
            picked.extend(sorted(burst_files)[-k:])
        return sorted(picked)

    def _run_batch(batch_cslcs, comp_slc_files, work_dir, output_dir, mode):
        """Build cfg + runconfig and run one disp_s1 batch (historical or forward)."""
        n_real = len(batch_cslcs) // n_bursts
        num_ccslc = len(comp_slc_files) // n_bursts
        work_dir.mkdir(parents=True, exist_ok=True)
        cfg = make_cfg(
            cslc_files=batch_cslcs,
            work_dir=work_dir,
            comp_slc_files=comp_slc_files,
            amp_disp_files=amp_disp_files,
            amp_mean_files=amp_mean_files,
            gpu=gpu,
            bounds=bounds,
            epsg=epsg,
            compressed_slc_plan=CompressedSlcPlan.LAST_PER_MINISTACK,
            output_reference_idx=max(0, num_ccslc - 1),
            ministack_size=num_ccslc + n_real + 1,
        )
        pge_rc = make_pge_runconfig(
            cfg=cfg,
            frame_id=frame_id,
            mode=mode,
            work_dir=work_dir,
            output_dir=output_dir,
            save_compressed_slc=(mode == ProcessingMode.HISTORICAL),
        )
        run_no_corrections(cfg, pge_rc)

    def _harvest_compressed(output_dir, work_dir) -> None:
        """Copy compressed SLCs produced by a historical run into the shared store."""
        comp_out = output_dir / "compressed_slcs"
        new_comp = sorted(comp_out.glob("*.h5")) if comp_out.exists() else []
        if not new_comp:
            new_comp = sorted(work_dir.rglob("linked_phase/compressed_*.tif"))
        for src in new_comp:
            dst = comp_slc_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    run_idx = 0

    # ── Historical ministacks (full ms_size groups) ───────────────────────────
    for ms_i in range(n_hist):
        s, e = ms_i * ms_size, (ms_i + 1) * ms_size
        batch_cslcs = [f for b in burst_ids for f in reals_by_burst[b][s:e]]
        comp_slc_files = _pick_comp(max_comp)
        work_dir = work_base / f"batch_{run_idx:03d}_historical"
        output_dir = work_base / f"output_{run_idx:03d}"

        print(f"\n{'='*60}")
        print(
            f"[historical {ms_i + 1}/{n_hist}] date idx {s}-{e - 1}  "
            f"reals={len(batch_cslcs)}  ccslc_in={len(comp_slc_files)}"
        )
        _run_batch(batch_cslcs, comp_slc_files, work_dir, output_dir,
                   ProcessingMode.HISTORICAL)
        _harvest_compressed(output_dir, work_dir)

        new_amp_disp = sorted(work_dir.rglob("*_amp_dispersion.tif"))
        new_amp_mean = sorted(work_dir.rglob("*_amp_mean.tif"))
        if new_amp_disp:
            amp_disp_files = new_amp_disp
        if new_amp_mean:
            amp_mean_files = new_amp_mean
        print(f"  → compressed store: {len(_pick_comp(max_comp))} (latest {max_comp}/burst)")
        run_idx += 1

    # ── Forward products (one run per leftover date) ──────────────────────────
    # Each forward run: 1 compressed SLC + `forward_window` reals ending at date n.
    # run_no_corrections keeps only the newest date's product, so one product/run.
    for k, n in enumerate(range(hist_count, n_dates)):
        s = max(0, n - (forward_window - 1))
        batch_cslcs = [f for b in burst_ids for f in reals_by_burst[b][s:n + 1]]
        comp_fwd = _pick_comp(1)  # exactly 1 ccslc per burst
        if not comp_fwd:
            raise SystemExit("No compressed SLC available for forward mode.")
        work_dir = work_base / f"batch_{run_idx:03d}_forward"
        output_dir = work_base / f"output_{run_idx:03d}"

        print(f"\n{'='*60}")
        print(
            f"[forward {k + 1}/{n_forward}] product date idx {n}  "
            f"window reals idx {s}-{n} ({len(batch_cslcs)} files)  "
            f"ccslc={len(comp_fwd)}"
        )
        _run_batch(batch_cslcs, comp_fwd, work_dir, output_dir,
                   ProcessingMode.FORWARD)
        run_idx += 1


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the DISP-S1 processing pipeline."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Required, site-specific ──
    p.add_argument(
        "--cslc-dir",
        type=Path,
        required=True,
        help="Directory of input (G)SLC burst HDF5 files (searched recursively).",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Output/scratch directory for batches, products, and the zarr store.",
    )
    p.add_argument("--frame-id", type=int, required=True, help="OPERA frame id.")
    p.add_argument(
        "--extent",
        required=True,
        help="Spatial subset as 'minlon,minlat : maxlon,maxlat' (WGS84).",
    )
    # ── Optional processing knobs ──
    p.add_argument(
        "-ms", "--ministack-size", type=int, default=DEFAULT_MS_SIZE,
        help="Number of acquisitions per ministack.",
    )
    p.add_argument(
        "--num-compressed", type=int, default=DEFAULT_MAX_COMP,
        help="Compressed SLCs carried forward per burst (historical).",
    )
    p.add_argument(
        "--forward-window", type=int, default=DEFAULT_FORWARD_WINDOW,
        help="Real SLCs per forward run (window ending at the product date, e.g. "
             "5 → n-4,n-3,n-2,n-1,n). Each forward run also uses 1 compressed SLC.",
    )
    p.add_argument(
        "--buffer", type=float, default=DEFAULT_BUFFER,
        help="Padding (metres) added around the extent bbox.",
    )
    p.add_argument("--gpu", action="store_true", help="Enable GPU (JAX) processing.")
    p.add_argument(
        "--gslc-glob", default=DEFAULT_GSLC_GLOB,
        help="Glob for (G)SLC HDF5 files under --cslc-dir (any burst id by default).",
    )
    p.add_argument(
        "--zarr-out", type=Path, default=None,
        help="Output zarr path (default: <work-dir>/disp.zarr).",
    )
    p.add_argument(
        "--reference-coherence-threshold", type=float, default=0.7,
        help="Coherence threshold for HIGH_COHERENCE reference selection.",
    )
    p.add_argument(
        "--weight-by-coherence", action="store_true",
        help="Weight the velocity fit by average temporal coherence.",
    )
    p.add_argument(
        "--stages", nargs="+", choices=STAGES, default=list(STAGES),
        help="Which pipeline stages to run (default: all). Later stages reuse "
             "existing outputs, so you can re-run e.g. only 'velocity extra'.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    work_base: Path = args.work_dir
    work_base.mkdir(parents=True, exist_ok=True)
    zarr_out = args.zarr_out or (work_base / "disp.zarr")

    # Stage scratch inside the work dir (JPL /tmp is tiny — keep off it).
    os.environ["TMPDIR"] = str(work_base)

    if "process" in args.stages:
        run_processing(
            cslc_dir=args.cslc_dir,
            work_base=work_base,
            frame_id=args.frame_id,
            extent=args.extent,
            ms_size=args.ministack_size,
            max_comp=args.num_compressed,
            buffer=args.buffer,
            gpu=args.gpu,
            gslc_glob=args.gslc_glob,
            forward_window=args.forward_window,
        )

    if "reformat" in args.stages:
        print("\n" + "=" * 60)
        print("Reformatting outputs to zarr...")
        input_files = sorted(work_base.glob("output_*/20*.nc"))
        if not input_files:
            raise SystemExit(
                f"No per-ministack .nc outputs found under {work_base}/output_*/"
            )
        reformat_stack_custom(
            input_files=input_files,
            output_name=str(zarr_out),
            apply_solid_earth_corrections=False,
            apply_ionospheric_corrections=False,
            quality_datasets=None,
            quality_thresholds=None,
            reference_method=ReferenceMethod.HIGH_COHERENCE,
            reference_coherence_threshold=args.reference_coherence_threshold,
            out_chunks=(4, 256, 256),
        )

    if "velocity" in args.stages:
        print("\nEstimating velocity...")
        add_velocity_to_zarr(str(zarr_out), weight_by_coherence=args.weight_by_coherence)

    if "extra" in args.stages:
        print("\nAppending extra layers...")
        append_extra_layers(
            zarr_path=str(zarr_out),
            work_base=work_base,
            gslc_dir=args.cslc_dir,
            gslc_glob=args.gslc_glob,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
