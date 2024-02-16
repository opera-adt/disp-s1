"""Module for creating the OPERA output product in NetCDF format."""

from __future__ import annotations

import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import h5netcdf
import h5py
import numpy as np
import pyproj
from dolphin import __version__ as dolphin_version
from dolphin import io
from dolphin._log import get_log
from dolphin._types import Filename
from dolphin.utils import format_dates
from isce3.core.types import truncate_mantissa
from numpy.typing import ArrayLike, DTypeLike
from opera_utils import OPERA_DATASET_NAME, get_dates, get_union_polygon

from . import __version__ as disp_s1_version
from . import _parse_cslc_product
from ._common import DATETIME_FORMAT
from .browse_image import make_browse_image_from_arr
from .pge_runconfig import RunConfig
from .product_info import DISP_PRODUCTS_INFO

logger = get_log(__name__)

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
FILE_OPTS = {"fs_strategy": "page", "fs_page_size": 2**22}

# Convert chunks to a tuple or h5py errors
HDF5_OPTS = io.DEFAULT_HDF5_OPTIONS.copy()
HDF5_OPTS["chunks"] = tuple(HDF5_OPTS["chunks"])  # type: ignore
# The GRID_MAPPING_DSET variable is used to store the name of the dataset containing
# the grid mapping information, which includes the coordinate reference system (CRS)
# and the GeoTransform. This is in accordance with the CF 1.8 conventions for adding
# geospatial metadata to NetCDF files.
# http://cfconventions.org/cf-conventions/cf-conventions.html#grid-mappings-and-projections
# Note that the name "spatial_ref" used here is arbitrary, but it follows the default
# used by other libraries, such as rioxarray:
# https://github.com/corteva/rioxarray/blob/5783693895b4b055909c5758a72a5d40a365ef11/rioxarray/rioxarray.py#L34 # noqa
GRID_MAPPING_DSET = "spatial_ref"

COMPRESSED_SLC_TEMPLATE = "compressed_{burst}_{date_str}.h5"


def create_output_product(
    output_name: Filename,
    unw_filename: Filename,
    conncomp_filename: Filename,
    temp_coh_filename: Filename,
    ifg_corr_filename: Filename,
    ps_mask_filename: Filename,
    pge_runconfig: RunConfig,
    cslc_files: Sequence[Filename],
    corrections: Optional[dict[str, ArrayLike]] = None,
):
    """Create the OPERA output product in NetCDF format.

    Parameters
    ----------
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
    output_name : Filename, optional
        The path to the output NetCDF file, by default "output.nc"
    corrections : dict[str, ArrayLike], optional
        A dictionary of corrections to write to the output file, by default None
    cslc_files : Sequence[Filename]
        The list of input CSLC products used to generate the output product.
    pge_runconfig : Optional[RunConfig], optional
        The PGE run configuration, by default None
        Used to add extra metadata to the output file.

    """
    # Read the Geotiff file and its metadata
    if corrections is None:
        corrections = {}
    crs = io.get_raster_crs(unw_filename)
    gt = io.get_raster_gt(unw_filename)
    unw_arr_ma = io.load_gdal(unw_filename, masked=True)
    unw_arr = np.ma.filled(unw_arr_ma, 0)

    conncomp_arr = io.load_gdal(conncomp_filename)
    temp_coh_arr = io.load_gdal(temp_coh_filename)
    truncate_mantissa(temp_coh_arr)
    ifg_corr_arr = io.load_gdal(ifg_corr_filename)
    truncate_mantissa(ifg_corr_arr)

    # Get the nodata mask (which for snaphu is 0)
    mask = unw_arr == 0
    # Set to NaN for final output
    unw_arr[mask] = np.nan

    assert unw_arr.shape == conncomp_arr.shape == temp_coh_arr.shape

    start_times = [
        _parse_cslc_product.get_zero_doppler_time(f, type_="start") for f in cslc_files
    ]
    start_time = min(start_times)
    end_times = [
        _parse_cslc_product.get_zero_doppler_time(f, type_="end") for f in cslc_files
    ]
    end_time = max(end_times)

    with h5netcdf.File(output_name, "w", **FILE_OPTS) as f:
        # Create the NetCDF file
        f.attrs.update(GLOBAL_ATTRS)

        # Set up the grid mapping variable for each group with rasters
        _create_grid_mapping(group=f, crs=crs, gt=gt)

        # Set up the X/Y variables for each group
        _create_yx_dsets(group=f, gt=gt, shape=unw_arr.shape, include_time=True)
        _create_time_dset(
            group=f,
            time=start_time,
            long_name="Time corresponding to beginning of Displacement frame",
        )

        # ######## Main datasets ###########
        # Write the displacement array / conncomp arrays
        disp_products_info = DISP_PRODUCTS_INFO
        disp_data = [
            unw_arr,
            conncomp_arr,
            temp_coh_arr,
            ifg_corr_arr,
            io.load_gdal(ps_mask_filename),
        ]
        disp_products = list(zip(disp_products_info, disp_data))
        for nfo, data in disp_products:
            _create_geo_dataset(
                group=f,
                name=nfo.name,
                data=data,
                description=nfo.description,
                fillvalue=nfo.fillvalue,
                attrs=nfo.attrs,
            )
            make_browse_image_from_arr(
                Path(output_name).with_suffix(f".{nfo.name}.png"), data
            )

    _create_corrections_group(
        output_name=output_name,
        corrections=corrections,
        shape=unw_arr.shape,
        gt=gt,
        crs=crs,
        start_time=min(start_times),
    )

    _create_identification_group(
        output_name=output_name,
        pge_runconfig=pge_runconfig,
        cslc_files=cslc_files,
        start_time=start_time,
        end_time=end_time,
    )

    _create_metadata_group(output_name=output_name, pge_runconfig=pge_runconfig)


def _create_corrections_group(
    output_name: Filename,
    corrections: dict[str, ArrayLike],
    shape: tuple[int, int],
    gt: list[float],
    crs: pyproj.CRS,
    start_time: datetime.datetime,
) -> None:
    with h5netcdf.File(output_name, "a") as f:
        # Create the group holding phase corrections used on the unwrapped phase
        corrections_group = f.create_group(CORRECTIONS_GROUP_NAME)
        corrections_group.attrs[
            "description"
        ] = "Phase corrections applied to the unwrapped_phase"
        empty_arr = np.zeros(shape, dtype="float32")

        # TODO: Are we going to downsample these for space?
        # if so, they need they're own X/Y variables and GeoTransform
        _create_grid_mapping(group=corrections_group, crs=crs, gt=gt)
        _create_yx_dsets(group=corrections_group, gt=gt, shape=shape, include_time=True)
        _create_time_dset(
            group=corrections_group,
            time=start_time,
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
    cslc_files: Sequence[Filename],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
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
            data=start_time.strftime(DATETIME_FORMAT),
            fillvalue=None,
            description=(
                "Zero doppler start time of the first burst contained in the frame."
            ),
        )
        _create_dataset(
            group=identification_group,
            name="zero_doppler_end_time",
            dimensions=(),
            data=end_time.strftime(DATETIME_FORMAT),
            fillvalue=None,
            description=(
                "Zero doppler end time of the last burst contained in the frame."
            ),
        )

        _create_dataset(
            group=identification_group,
            name="bounding_polygon",
            dimensions=(),
            data=get_union_polygon(cslc_files).wkt,
            fillvalue=None,
            description="WKT representation of bounding polygon of the image",
            attrs={"units": "degrees"},
        )

        wavelength, attrs = _parse_cslc_product.get_radar_wavelength(cslc_files[-1])
        desc = attrs.pop("description")
        _create_dataset(
            group=identification_group,
            name="radar_wavelength",
            dimensions=(),
            data=wavelength,
            fillvalue=None,
            description=desc,
            attrs=attrs,
        )

        reference_date, secondary_date = get_dates(output_name)[:2]
        _create_dataset(
            group=identification_group,
            name="reference_datetime",
            dimensions=(),
            data=reference_date.strftime(DATETIME_FORMAT),
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
            data=secondary_date.strftime(DATETIME_FORMAT),
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
            "long_name": (
                "Dummy variable containing geo-referencing metadata in attributes"
            ),
        }
    )

    return dset


def create_compressed_products(
    comp_slc_dict: dict[str, list[Path]], output_dir: Filename
):
    """Make the compressed SLC output product."""

    def form_name(filename: Path, burst: str):
        # filename: compressed_20180222_20180716.tif
        date_str = format_dates(*get_dates(filename.stem))
        return COMPRESSED_SLC_TEMPLATE.format(burst=burst, date_str=date_str)

    attrs = GLOBAL_ATTRS.copy()
    attrs["title"] = "Compressed SLC"
    *parts, dset_name = OPERA_DATASET_NAME.split("/")
    group_name = "/".join(parts)

    for burst, comp_slc_files in comp_slc_dict.items():
        for comp_slc_file in comp_slc_files:
            outname = Path(output_dir) / form_name(comp_slc_file, burst)
            if outname.exists():
                logger.info(f"Skipping existing {outname}")
                continue

            crs = io.get_raster_crs(comp_slc_file)
            gt = io.get_raster_gt(comp_slc_file)
            data = io.load_gdal(comp_slc_file)
            truncate_mantissa(data)

            # Input metadata is stored within the GDAL "DOLPHIN" domain
            metadata_dict = io.get_raster_metadata(comp_slc_file, "DOLPHIN")
            attrs = {"units": "unitless"}
            attrs.update(metadata_dict)

            logger.info(f"Writing {outname}")
            with h5py.File(outname, "w") as hf:
                # add type to root for GDAL recognition of complex datasets in NetCDF
                ctype = h5py.h5t.py_create(np.complex64)
                ctype.commit(hf["/"].id, np.string_("complex64"))

            with h5netcdf.File(outname, mode="a", invalid_netcdf=True) as f:
                f.attrs.update(attrs)

                data_group = f.create_group(group_name)
                _create_grid_mapping(group=data_group, crs=crs, gt=gt)
                _create_yx_dsets(
                    group=data_group, gt=gt, shape=data.shape, include_time=False
                )
                _create_geo_dataset(
                    group=data_group,
                    name=dset_name,
                    data=data,
                    description="Compressed SLC product",
                    fillvalue=np.nan + 0j,
                    attrs=attrs,
                )
