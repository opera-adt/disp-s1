"""Module for creating the OPERA output product in NetCDF format."""

from __future__ import annotations

import datetime
import logging
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence, Union

import h5netcdf
import h5py
import numpy as np
import pyproj
from dolphin import __version__ as dolphin_version
from dolphin import filtering, io
from dolphin._types import Filename
from dolphin.io import round_mantissa
from dolphin.utils import format_dates
from numpy.typing import ArrayLike, DTypeLike
from opera_utils import (
    OPERA_DATASET_NAME,
    filter_by_burst_id,
    filter_by_date,
    get_dates,
    get_radar_wavelength,
    get_zero_doppler_time,
)
from tqdm.contrib.concurrent import process_map

from . import __version__ as disp_s1_version
from ._common import DATETIME_FORMAT
from .browse_image import make_browse_image_from_arr
from .pge_runconfig import RunConfig
from .product_info import DISPLACEMENT_PRODUCTS, ProductInfo

logger = logging.getLogger(__name__)

CORRECTIONS_GROUP_NAME = "corrections"
IDENTIFICATION_GROUP_NAME = "identification"
METADATA_GROUP_NAME = "metadata"
GLOBAL_ATTRS = {
    "Conventions": "CF-1.8",
    "contact": "operaops@jpl.nasa.gov",
    "institution": "NASA JPL",
    "mission_name": "OPERA",
    "reference_document": "TBD",
    "title": "OPERA L3_DISP-S1 Product",
}

# Use the "paging file space strategy"
# https://docs.h5py.org/en/stable/high/file.html#h5py.File
# Page size should be larger than the largest chunk in the file
FILE_OPTS = {"fs_strategy": "page", "fs_page_size": 2**22}
CHUNK_SHAPE = (128, 128)

# Convert chunks to a tuple or h5py errors
HDF5_OPTS = io.DEFAULT_HDF5_OPTIONS.copy()
HDF5_OPTS["chunks"] = tuple(CHUNK_SHAPE)  # type: ignore
# The GRID_MAPPING_DSET variable is used to store the name of the dataset containing
# the grid mapping information, which includes the coordinate reference system (CRS)
# and the GeoTransform. This is in accordance with the CF 1.8 conventions for adding
# geospatial metadata to NetCDF files.
# http://cfconventions.org/cf-conventions/cf-conventions.html#grid-mappings-and-projections
# Note that the name "spatial_ref" used here is arbitrary, but it follows the default
# used by other libraries, such as rioxarray:
# https://github.com/corteva/rioxarray/blob/5783693895b4b055909c5758a72a5d40a365ef11/rioxarray/rioxarray.py#L34 # noqa
GRID_MAPPING_DSET = "spatial_ref"

COMPRESSED_SLC_TEMPLATE = "compressed_{burst_id}_{date_str}.h5"


def create_output_product(
    output_name: Filename,
    unw_filename: Filename,
    conncomp_filename: Filename,
    temp_coh_filename: Filename,
    ifg_corr_filename: Filename,
    ps_mask_filename: Filename,
    unwrapper_mask_filename: Filename | None,
    pge_runconfig: RunConfig,
    reference_cslc_file: Filename,
    secondary_cslc_file: Filename,
    corrections: Optional[dict[str, ArrayLike]] = None,
    wavelength_cutoff: float = 50_000.0,
):
    """Create the OPERA output product in NetCDF format.

    Parameters
    ----------
    output_name : Filename, optional
        The path to the output NetCDF file.
    unw_filename : Filename
        The path to the input unwrapped phase image.
    conncomp_filename : Filename
        The path to the input connected components image.
    temp_coh_filename : Filename
        The path to the input temporal coherence image.
    ifg_corr_filename : Filename
        The path to the input interferometric correlation image.
    ps_mask_filename : Filename
        The path to the input persistent scatterer mask image.
    unwrapper_mask_filename : Filename, optional
        The path to the boolean mask used during unwrapping to ignore pixels.
    pge_runconfig : Optional[RunConfig], optional
        The PGE run configuration, by default None
        Used to add extra metadata to the output file.
    reference_cslc_file : Filename
        An input CSLC product corresponding to the reference date.
        Used for metadata generation.
    secondary_cslc_file : Filename
        An input CSLC product corresponding to the secondary date.
        Used for metadata generation.
    corrections : dict[str, ArrayLike], optional
        A dictionary of corrections to write to the output file, by default None
    wavelength_cutoff : float, optional
        The wavelength cutoff for filtering long wavelengths.
        Default is 50_000.0


    """
    if corrections is None:
        corrections = {}
    crs = io.get_raster_crs(unw_filename)
    gt = io.get_raster_gt(unw_filename)

    reference_start_time = get_zero_doppler_time(reference_cslc_file, type_="start")
    secondary_start_time = get_zero_doppler_time(secondary_cslc_file, type_="start")
    secondary_end_time = get_zero_doppler_time(secondary_cslc_file, type_="end")

    # TODO: get rid after https://github.com/isce-framework/dolphin/pull/367 merged
    radar_wavelength = get_radar_wavelength(reference_cslc_file)
    phase2disp = -1 * float(radar_wavelength) / (4.0 * np.pi)

    try:
        footprint_wkt = extract_footprint(raster_path=unw_filename)
    except Exception:
        logger.error("Failed to extract raster footprint", exc_info=True)
        footprint_wkt = ""

    # Load and process unwrapped phase data, needs more custom masking
    unw_arr_ma = io.load_gdal(unw_filename, masked=True)
    unw_arr = np.ma.filled(unw_arr_ma, 0)
    mask = unw_arr == 0

    disp_arr = unw_arr * phase2disp
    shape = unw_arr.shape

    _, x_res, _, _, _, y_res = gt
    # Average for the pixel spacing for filtering
    pixel_spacing = (abs(x_res) + abs(y_res)) / 2
    logger.info(
        "Creating short wavelength displacement product with %s meter cutoff",
        wavelength_cutoff,
    )
    bad_corr = io.load_gdal(ifg_corr_filename) < 0.5
    bad_conncomp = io.load_gdal(conncomp_filename, masked=True).filled(0) == 0
    filtered_disp_arr = filtering.filter_long_wavelength(
        unwrapped_phase=disp_arr,
        bad_pixel_mask=bad_corr | bad_conncomp,
        wavelength_cutoff=wavelength_cutoff,
        pixel_spacing=pixel_spacing,
    )
    DISPLACEMENT_PRODUCTS.short_wavelength_displacement.attrs |= {
        "wavelength_cutoff": str(wavelength_cutoff)
    }

    disp_arr[mask] = np.nan
    filtered_disp_arr[mask] = np.nan

    product_infos: list[ProductInfo] = list(DISPLACEMENT_PRODUCTS)

    with h5netcdf.File(output_name, "w", **FILE_OPTS) as f:
        f.attrs.update(GLOBAL_ATTRS)
        _create_grid_mapping(group=f, crs=crs, gt=gt)

        _create_yx_dsets(group=f, gt=gt, shape=shape, include_time=True)
        _create_time_dset(
            group=f,
            time=secondary_start_time,
            long_name="Time corresponding to beginning of Displacement frame",
        )
        for info, data in zip(product_infos[:2], [disp_arr, filtered_disp_arr]):
            round_mantissa(data, keep_bits=info.keep_bits)
            _create_geo_dataset(
                group=f,
                name=info.name,
                data=data,
                description=info.description,
                fillvalue=info.fillvalue,
                attrs=info.attrs,
            )

            make_browse_image_from_arr(
                Path(output_name).with_suffix(f".{info.name}.png"), data
            )
            del data  # Free up memory

        # For the others, load and save each individually
        data_files = [
            conncomp_filename,
            temp_coh_filename,
            ifg_corr_filename,
            ps_mask_filename,
            unwrapper_mask_filename,
        ]

        for info, filename in zip(product_infos[2:], data_files):
            if filename is None:
                data = np.full(shape=shape, fill_value=info.fillvalue, dtype=info.dtype)
            else:
                data = io.load_gdal(filename)

            if info.keep_bits is not None:
                round_mantissa(data, keep_bits=info.keep_bits)

            _create_geo_dataset(
                group=f,
                name=info.name,
                data=data,
                description=info.description,
                fillvalue=info.fillvalue,
                attrs=info.attrs,
            )
            del data  # Free up memory

    _create_corrections_group(
        output_name=output_name,
        corrections=corrections,
        shape=shape,
        gt=gt,
        crs=crs,
        secondary_start_time=secondary_start_time,
    )

    _create_identification_group(
        output_name=output_name,
        pge_runconfig=pge_runconfig,
        radar_wavelength=radar_wavelength,
        reference_start_time=reference_start_time,
        secondary_start_time=secondary_start_time,
        secondary_end_time=secondary_end_time,
        footprint_wkt=footprint_wkt,
    )

    _create_metadata_group(output_name=output_name, pge_runconfig=pge_runconfig)


def _create_corrections_group(
    output_name: Filename,
    corrections: dict[str, ArrayLike],
    shape: tuple[int, int],
    gt: list[float],
    crs: pyproj.CRS,
    secondary_start_time: datetime.datetime,
) -> None:
    keep_bits = 10
    logger.debug("Rounding mantissa in corrections to %s bits", keep_bits)
    for data in corrections.values():
        # Use same amount of truncation for all correction layers
        if np.issubdtype(data.dtype, np.floating):
            round_mantissa(data, keep_bits=keep_bits)
    logger.info("Creating corrections group in %s", output_name)
    with h5netcdf.File(output_name, "a") as f:
        # Create the group holding phase corrections used on the unwrapped phase
        corrections_group = f.create_group(CORRECTIONS_GROUP_NAME)
        corrections_group.attrs["description"] = (
            "Phase corrections applied to the unwrapped_phase"
        )
        empty_arr = np.zeros(shape, dtype="float32")

        # TODO: Are we going to downsample these for space?
        # if so, they need they're own X/Y variables and GeoTransform
        _create_grid_mapping(group=corrections_group, crs=crs, gt=gt)
        _create_yx_dsets(group=corrections_group, gt=gt, shape=shape, include_time=True)
        _create_time_dset(
            group=corrections_group,
            time=secondary_start_time,
            long_name="time corresponding to beginning of Displacement frame",
        )
        troposphere = corrections.get("troposphere", empty_arr)
        _create_geo_dataset(
            group=corrections_group,
            name="tropospheric_delay",
            data=troposphere,
            description="Tropospheric phase delay used to correct the unwrapped phase",
            fillvalue=np.nan,
            attrs={"units": "radians"},
        )
        ionosphere = corrections.get("ionosphere", empty_arr)
        _create_geo_dataset(
            group=corrections_group,
            name="ionospheric_delay",
            data=ionosphere,
            description="Ionospheric phase delay used to correct the unwrapped phase",
            fillvalue=np.nan,
            attrs={"units": "radians"},
        )
        solid_earth = corrections.get("solid_earth", empty_arr)
        _create_geo_dataset(
            group=corrections_group,
            name="solid_earth_tide",
            data=solid_earth,
            description="Solid Earth tide used to correct the unwrapped phase",
            fillvalue=np.nan,
            attrs={"units": "radians"},
        )
        plate_motion = corrections.get("plate_motion", empty_arr)
        _create_geo_dataset(
            group=corrections_group,
            name="plate_motion",
            data=plate_motion,
            description="Phase ramp caused by tectonic plate motion",
            fillvalue=np.nan,
            attrs={"units": "radians"},
        )
        # Make a scalar dataset for the reference point
        reference_point = corrections.get("reference_point", 0.0)
        _create_dataset(
            group=corrections_group,
            name="reference_point",
            dimensions=(),
            data=reference_point,
            fillvalue=0,
            description=(
                "Dummy dataset containing attributes with the locations where the"
                " reference phase was taken."
            ),
            dtype=int,
            # Note: the dataset contains attributes with lists, since the reference
            # could have come from multiple points (e.g. boxcar average of an area).
            attrs={
                "units": "unitless",
                "rows": [],
                "cols": [],
                "latitudes": [],
                "longitudes": [],
            },
        )


def _create_identification_group(
    output_name: Filename,
    pge_runconfig: RunConfig,
    radar_wavelength: float,
    reference_start_time: datetime.datetime,
    secondary_start_time: datetime.datetime,
    secondary_end_time: datetime.datetime,
    footprint_wkt: str,
) -> None:
    """Create the identification group in the output file."""
    with h5netcdf.File(output_name, "a") as f:
        identification_group = f.create_group(IDENTIFICATION_GROUP_NAME)
        _create_dataset(
            group=identification_group,
            name="frame_id",
            dimensions=(),
            data=pge_runconfig.input_file_group.frame_id,
            fillvalue=None,
            description="ID number of the processed frame.",
        )
        _create_dataset(
            group=identification_group,
            name="product_version",
            dimensions=(),
            data=pge_runconfig.product_path_group.product_version,
            fillvalue=None,
            description="Version of the product.",
        )

        _create_dataset(
            group=identification_group,
            name="zero_doppler_start_time",
            dimensions=(),
            data=secondary_start_time.strftime(DATETIME_FORMAT),
            fillvalue=None,
            description=(
                "Zero doppler start time of the first burst contained in the frame for"
                " the secondary acquisition."
            ),
        )
        _create_dataset(
            group=identification_group,
            name="zero_doppler_end_time",
            dimensions=(),
            data=secondary_end_time.strftime(DATETIME_FORMAT),
            fillvalue=None,
            description=(
                "Zero doppler start time of the last burst contained in the frame for"
                " the secondary acquisition."
            ),
        )

        _create_dataset(
            group=identification_group,
            name="bounding_polygon",
            dimensions=(),
            data=footprint_wkt,
            fillvalue=None,
            description="WKT representation of bounding polygon of the image",
            attrs={"units": "degrees"},
        )

        _create_dataset(
            group=identification_group,
            name="radar_wavelength",
            dimensions=(),
            data=radar_wavelength,
            fillvalue=None,
            description="Wavelength of the transmitted signal",
            attrs={"units": "meters"},
        )

        _create_dataset(
            group=identification_group,
            name="reference_datetime",
            dimensions=(),
            data=reference_start_time.strftime(DATETIME_FORMAT),
            fillvalue=None,
            description=(
                "UTC datetime of the acquisition sensing start of the reference epoch"
                " to which the unwrapped phase is referenced."
            ),
        )
        _create_dataset(
            group=identification_group,
            name="secondary_datetime",
            dimensions=(),
            data=secondary_start_time.strftime(DATETIME_FORMAT),
            fillvalue=None,
            description=(
                "UTC datetime of the acquisition sensing start of current acquisition"
                " used to create the unwrapped phase."
            ),
        )


def _create_metadata_group(output_name: Filename, pge_runconfig: RunConfig) -> None:
    """Create the metadata group in the output file."""
    with h5netcdf.File(output_name, "a") as f:
        metadata_group = f.create_group(METADATA_GROUP_NAME)
        _create_dataset(
            group=metadata_group,
            name="disp_s1_software_version",
            dimensions=(),
            data=disp_s1_version,
            fillvalue=None,
            description="Version of the disp-s1 software used to generate the product.",
        )
        _create_dataset(
            group=metadata_group,
            name="dolphin_software_version",
            dimensions=(),
            data=dolphin_version,
            fillvalue=None,
            description="Version of the dolphin software used to generate the product.",
        )

        # TODO: prob should just make a _to_string method?
        ss = StringIO()
        pge_runconfig.to_yaml(ss)
        runconfig_str = ss.getvalue()
        _create_dataset(
            group=metadata_group,
            name="pge_runconfig",
            dimensions=(),
            data=runconfig_str,
            fillvalue=None,
            description=(
                "The full PGE runconfig YAML file used to generate the product."
            ),
        )


def _create_dataset(
    *,
    group: h5netcdf.Group,
    name: str,
    dimensions: Optional[Sequence[str]],
    data: Union[np.ndarray, str],
    description: str,
    fillvalue: Optional[float],
    attrs: Optional[dict[str, Any]] = None,
    dtype: Optional[DTypeLike] = None,
) -> h5netcdf.Variable:
    if attrs is None:
        attrs = {}
    attrs.update(long_name=description)

    options = HDF5_OPTS
    if isinstance(data, str):
        options = {}
        # This is a string, so we need to convert it to bytes or it will fail
        data = np.string_(data)
    elif np.array(data).size <= 1:
        # Scalars don't need chunks/compression
        options = {}
    dset = group.create_variable(
        name,
        dimensions=dimensions,
        data=data,
        dtype=dtype,
        fillvalue=fillvalue,
        **options,
    )
    dset.attrs.update(attrs)
    return dset


def _create_geo_dataset(
    *,
    group: h5netcdf.Group,
    name: str,
    data: np.ndarray,
    description: str,
    fillvalue: float,
    attrs: Optional[dict[str, Any]],
    include_time: bool = False,
) -> h5netcdf.Variable:
    if include_time:
        dimensions = ["time", "y", "x"]
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
    else:
        dimensions = ["y", "x"]
    dset = _create_dataset(
        group=group,
        name=name,
        dimensions=dimensions,
        data=data,
        description=description,
        fillvalue=fillvalue,
        attrs=attrs,
    )
    dset.attrs["grid_mapping"] = GRID_MAPPING_DSET
    return dset


def _create_yx_arrays(
    gt: list[float], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Create the x and y coordinate datasets."""
    ysize, xsize = shape
    # Parse the geotransform
    x_origin, x_res, _, y_origin, _, y_res = gt

    # Make the x/y arrays
    # Note that these are the center of the pixels, whereas the GeoTransform
    # is the upper left corner of the top left pixel.
    y = np.arange(y_origin + y_res / 2, y_origin + y_res * ysize, y_res)
    x = np.arange(x_origin + x_res / 2, x_origin + x_res * xsize, x_res)
    return y, x


def _create_yx_dsets(
    group: h5netcdf.Group,
    gt: list[float],
    shape: tuple[int, int],
    include_time: bool = False,
) -> tuple[h5netcdf.Variable, h5netcdf.Variable]:
    """Create the y, x, and coordinate datasets."""
    y, x = _create_yx_arrays(gt, shape)

    if not group.dimensions:
        dims = {"y": y.size, "x": x.size}
        if include_time:
            dims["time"] = 1
        group.dimensions = dims

    # Create the x/y datasets
    y_ds = group.create_variable("y", ("y",), data=y, dtype=float)
    x_ds = group.create_variable("x", ("x",), data=x, dtype=float)

    for name, ds in zip(["y", "x"], [y_ds, x_ds]):
        ds.attrs["standard_name"] = f"projection_{name}_coordinate"
        ds.attrs["long_name"] = f"{name.replace('_', ' ')} coordinate of projection"
        ds.attrs["units"] = "m"
    return y_ds, x_ds


def _create_time_dset(
    group: h5netcdf.Group, time: datetime.datetime, long_name: str = "time"
) -> tuple[h5netcdf.Variable, h5netcdf.Variable]:
    """Create the time coordinate dataset."""
    times, calendar, units = _create_time_array([time])
    t_ds = group.create_variable("time", ("time",), data=times, dtype=float)
    t_ds.attrs["standard_name"] = "time"
    t_ds.attrs["long_name"] = long_name
    t_ds.attrs["calendar"] = calendar
    t_ds.attrs["units"] = units

    return t_ds


def _create_time_array(times: list[datetime.datetime]):
    """Set up the CF-compliant time array and dimension metadata.

    References
    ----------
    http://cfconventions.org/cf-conventions/cf-conventions.html#time-coordinate

    """
    # 'calendar': 'standard',
    # 'units': 'seconds since 2017-02-03 00:00:00.000000'
    # Create the time array
    since_time = times[0]
    time = np.array([(t - since_time).total_seconds() for t in times])
    calendar = "standard"
    units = f"seconds since {since_time.strftime(DATETIME_FORMAT)}"
    return time, calendar, units


def _create_grid_mapping(group, crs: pyproj.CRS, gt: list[float]) -> h5netcdf.Variable:
    """Set up the grid mapping variable."""
    # https://github.com/corteva/rioxarray/blob/21284f67db536d9c104aa872ab0bbc261259e59e/rioxarray/rioxarray.py#L34
    dset = group.create_variable(GRID_MAPPING_DSET, (), data=0, dtype=int)

    dset.attrs.update(crs.to_cf())
    # Also add the GeoTransform
    gt_string = " ".join([str(x) for x in gt])
    dset.attrs.update(
        {
            "GeoTransform": gt_string,
            "units": "unitless",
            "long_name": "Dummy variable with geo-referencing metadata in attributes",
        }
    )

    return dset


class CompressedSLCInfo(NamedTuple):
    """Data for creating one compressed SLC HDF5."""

    burst_id: str
    comp_slc_file: Path
    output_dir: Path
    opera_cslc_file: Path


def process_compressed_slc(info: CompressedSLCInfo) -> Path:
    """Make one compressed SLC output product."""
    burst_id, comp_slc_file, output_dir, opera_cslc_file = info
    date_str = format_dates(*get_dates(comp_slc_file.stem))
    name = COMPRESSED_SLC_TEMPLATE.format(burst_id=burst_id, date_str=date_str)
    outname = Path(output_dir) / name

    if outname.exists():
        logger.info(f"Skipping existing {outname}")

    crs = io.get_raster_crs(comp_slc_file)
    gt = io.get_raster_gt(comp_slc_file)
    data = io.load_gdal(comp_slc_file, band=1)
    # COMPASS used `truncate_mantissa` default, 10 bits
    round_mantissa(data, keep_bits=10)

    # Input metadata is stored within the GDAL "DOLPHIN" domain
    metadata_dict = io.get_raster_metadata(comp_slc_file, "DOLPHIN")
    attrs = {"units": "unitless"}
    attrs.update(metadata_dict)

    *parts, dset_name = OPERA_DATASET_NAME.split("/")
    dispersion_dset_name = "amplitude_dispersion"
    group_name = "/".join(parts)
    logger.info(f"Writing {outname}")
    with h5py.File(outname, "w") as hf:
        # add type to root for GDAL recognition of complex datasets in NetCDF
        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(hf["/"].id, np.string_("complex64"))

    with h5netcdf.File(outname, mode="a", invalid_netcdf=True) as f:
        f.attrs.update(attrs)

        data_group = f.create_group(group_name)
        _create_grid_mapping(group=data_group, crs=crs, gt=gt)
        _create_yx_dsets(group=data_group, gt=gt, shape=data.shape, include_time=False)
        _create_geo_dataset(
            group=data_group,
            name=dset_name,
            data=data,
            description="Compressed SLC product",
            fillvalue=np.nan + 0j,
            attrs=attrs,
        )
        del data

        # Add the amplitude dispersion
        amp_dispersion_data = io.load_gdal(comp_slc_file, band=2).real.astype("float32")
        round_mantissa(amp_dispersion_data, keep_bits=10)
        _create_geo_dataset(
            group=data_group,
            name=dispersion_dset_name,
            data=amp_dispersion_data,
            description="Amplitude dispersion for the compressed SLC files.",
            fillvalue=np.nan,
            attrs={"units": "unitless"},
        )

    copy_opera_cslc_metadata(opera_cslc_file, outname)

    return outname


def copy_opera_cslc_metadata(
    comp_slc_file: Filename, output_hdf5_file: Filename
) -> None:
    """Copy orbit and metadata datasets from the input CSLC file the compressed SLC.

    Parameters
    ----------
    comp_slc_file : Filename
        Path to the input CSLC file.
    output_hdf5_file : Filename
        Path to the output compressed SLC file.

    """
    dsets_to_copy = [
        "/metadata/processing_information/input_burst_metadata/wavelength",
        "/identification/zero_doppler_end_time",
        "/identification/zero_doppler_start_time",
        "/identification/bounding_polygon",
        "/metadata/orbit",  #          Group
    ]

    with h5py.File(comp_slc_file, "r") as src, h5py.File(output_hdf5_file, "a") as dst:
        for dset_path in dsets_to_copy:
            if dset_path in src:
                # Create parent group if it doesn't exist
                dst.require_group(str(Path(dset_path).parent))

                # Remove existing dataset/group if it exists
                if dset_path in dst:
                    del dst[dset_path]

                # Copy the dataset or group
                src.copy(
                    src[dset_path],
                    dst[str(Path(dset_path).parent)],
                    name=Path(dset_path).name,
                )
            else:
                logger.warning(
                    f"Dataset or group {dset_path} not found in {comp_slc_file}"
                )

    logger.info(f"Copied metadata from {comp_slc_file} to {output_hdf5_file}")


def create_compressed_products(
    comp_slc_dict: Mapping[str, Sequence[Path]],
    output_dir: Filename,
    cslc_file_list: Sequence[Path],
    max_workers: int = 3,
) -> list[Path]:
    """Create all compressed SLC output products.

    Parameters
    ----------
    comp_slc_dict : dict[str, list[Path]]
        A dictionary mapping burst_id to lists of compressed SLC files.
    output_dir : Filename
        The directory to write the compressed SLC products to.
    cslc_file_list : Sequence[Path]
        Full set of input CSLCs used during processing.
        Used to pick out metadata corresponding to each compressed SLC's
        reference date.
    max_workers : int
        Number of parallel threads to use to create products.
        Default is 3.

    Returns
    -------
    list[Path]
        Paths to output compressed SLC files

    """
    compressed_slc_infos = []
    for burst_id, comp_slc_files in comp_slc_dict.items():
        for comp_slc_file in comp_slc_files:
            # Pick out the one that matches the current date/burst_id
            ref_date = get_dates(comp_slc_file)[0]
            valid_date_files = filter_by_date(cslc_file_list, [ref_date])
            matching_files = filter_by_burst_id(valid_date_files, burst_id)
            msg = (
                f"Found {len(matching_files)} matching CSLC files for"
                f" {burst_id} {ref_date}"
            )
            logger.info(msg)
            logger.info(matching_files)

            cur_opera_cslc = matching_files[-1]
            c = CompressedSLCInfo(burst_id, comp_slc_file, output_dir, cur_opera_cslc)
            compressed_slc_infos.append(c)

    results = process_map(
        process_compressed_slc,
        compressed_slc_infos,
        max_workers=max_workers,
        desc="Processing compressed SLCs",
    )

    logger.info("Finished creating all compressed SLC products.")
    return results


def extract_footprint(raster_path: Filename, simplify_tolerance: float = 0.01) -> str:
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
    return footprint.wkt
