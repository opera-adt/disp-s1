"""Module for creating the OPERA output product in NetCDF format."""
from __future__ import annotations

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
from dolphin.opera_utils import OPERA_DATASET_NAME
from dolphin.utils import get_dates
from isce3.core.types import truncate_mantissa
from numpy.typing import ArrayLike, DTypeLike
from PIL import Image

from disp_s1.pge_runconfig import RunConfig

logger = get_log(__name__)

CORRECTIONS_GROUP_NAME = "corrections"
IDENTIFICATION_GROUP_NAME = "identification"
GLOBAL_ATTRS = dict(
    Conventions="CF-1.8",
    contact="operaops@jpl.nasa.gov",
    institution="NASA JPL",
    mission_name="OPERA",
    reference_document="TBD",
    title="OPERA L3_DISP-S1 Product",
)

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
    unw_filename: Filename,
    conncomp_filename: Filename,
    tcorr_filename: Filename,
    spatial_corr_filename: Filename,
    output_name: Filename,
    corrections: dict[str, ArrayLike] = {},
    pge_runconfig: Optional[RunConfig] = None,
    create_browse_image: bool = True,
):
    """Create the OPERA output product in NetCDF format.

    Parameters
    ----------
    unw_filename : Filename
        The path to the input unwrapped phase image.
    conncomp_filename : Filename
        The path to the input connected components image.
    tcorr_filename : Filename
        The path to the input temporal correlation image.
    spatial_corr_filename : Filename
        The path to the input spatial correlation image.
    output_name : Filename, optional
        The path to the output NetCDF file, by default "output.nc"
    corrections : dict[str, ArrayLike], optional
        A dictionary of corrections to write to the output file, by default None
    pge_runconfig : Optional[RunConfig], optional
        The PGE run configuration, by default None
        Used to add extra metadata to the output file.
    create_browse_image : bool
        If true, creates a PNG browse image of the unwrapped phase
        with filename as `output_name`.png
    """
    # Read the Geotiff file and its metadata
    crs = io.get_raster_crs(unw_filename)
    gt = io.get_raster_gt(unw_filename)
    unw_arr_ma = io.load_gdal(unw_filename, masked=True)
    unw_arr = np.ma.filled(unw_arr_ma, 0)

    conncomp_arr = io.load_gdal(conncomp_filename)
    tcorr_arr = io.load_gdal(tcorr_filename)
    truncate_mantissa(tcorr_arr)
    # TODO: add spatial correlation, pass through to function
    spatial_corr_arr = io.load_gdal(spatial_corr_filename)
    truncate_mantissa(spatial_corr_arr)

    # Get the nodata mask (which for snaphu is 0)
    mask = unw_arr == 0
    # Set to NaN for final output
    unw_arr[mask] = np.nan

    assert unw_arr.shape == conncomp_arr.shape == tcorr_arr.shape

    if create_browse_image:
        make_browse_image(Path(output_name).with_suffix(".png"), unw_arr)

    with h5netcdf.File(output_name, "w") as f:
        # Create the NetCDF file
        f.attrs.update(GLOBAL_ATTRS)

        # Set up the grid mapping variable for each group with rasters
        _create_grid_mapping(group=f, crs=crs, gt=gt)

        # Set up the X/Y variables for each group
        _create_yx_dsets(group=f, gt=gt, shape=unw_arr.shape)

        # Write the displacement array / conncomp arrays
        _create_geo_dataset(
            group=f,
            name="unwrapped_phase",
            data=unw_arr,
            description="Unwrapped phase",
            fillvalue=np.nan,
            attrs=dict(units="radians"),
        )
        _create_geo_dataset(
            group=f,
            name="connected_component_labels",
            data=conncomp_arr,
            description="Connected component labels of the unwrapped phase",
            fillvalue=0,
            attrs=dict(units="unitless"),
        )
        _create_geo_dataset(
            group=f,
            name="temporal_correlation",
            data=tcorr_arr,
            description="Temporal correlation of phase inversion",
            fillvalue=np.nan,
            attrs=dict(units="unitless"),
        )
        _create_geo_dataset(
            group=f,
            name="spatial_correlation",
            data=spatial_corr_arr,
            description="Multilooked sample interferometric correlation",
            fillvalue=np.nan,
            attrs=dict(units="unitless"),
        )

        # Create the group holding phase corrections that were used on the unwrapped phase
        corrections_group = f.create_group(CORRECTIONS_GROUP_NAME)
        corrections_group.attrs["description"] = (
            "Phase corrections applied to the unwrapped_phase"
        )

        # TODO: Are we going to downsample these for space?
        # if so, they need they're own X/Y variables and GeoTransform
        _create_grid_mapping(group=corrections_group, crs=crs, gt=gt)
        _create_yx_dsets(group=corrections_group, gt=gt, shape=unw_arr.shape)
        troposphere = corrections.get("troposphere", np.zeros_like(unw_arr))
        _create_geo_dataset(
            group=corrections_group,
            name="tropospheric_delay",
            data=troposphere,
            description="Tropospheric phase delay used to correct the unwrapped phase",
            fillvalue=np.nan,
            attrs=dict(units="radians"),
        )
        ionosphere = corrections.get("ionosphere", np.zeros_like(unw_arr))
        _create_geo_dataset(
            group=corrections_group,
            name="ionospheric_delay",
            data=ionosphere,
            description="Ionospheric phase delay used to correct the unwrapped phase",
            fillvalue=np.nan,
            attrs=dict(units="radians"),
        )
        solid_earth = corrections.get("solid_earth", np.zeros_like(unw_arr))
        _create_geo_dataset(
            group=corrections_group,
            name="solid_earth_tide",
            data=solid_earth,
            description="Solid Earth tide used to correct the unwrapped phase",
            fillvalue=np.nan,
            attrs=dict(units="radians"),
        )
        plate_motion = corrections.get("plate_motion", np.zeros_like(unw_arr))
        _create_geo_dataset(
            group=corrections_group,
            name="plate_motion",
            data=plate_motion,
            description="Phase ramp caused by tectonic plate motion",
            fillvalue=np.nan,
            attrs=dict(units="radians"),
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
            # could have come from multiple points (e.g. some boxcar average of an area).
            attrs=dict(units="unitless", rows=[], cols=[], latitudes=[], longitudes=[]),
        )

    # End of the product for non-PGE users
    if pge_runconfig is None:
        return

    # Add the PGE metadata to the file
    with h5netcdf.File(output_name, "a") as f:
        identification_group = f.create_group(IDENTIFICATION_GROUP_NAME)
        _create_dataset(
            group=identification_group,
            name="frame_id",
            dimensions=(),
            data=pge_runconfig.input_file_group.frame_id,
            fillvalue=None,
            description="ID number of the processed frame.",
            attrs=dict(units="unitless"),
        )
        # product_version
        _create_dataset(
            group=identification_group,
            name="product_version",
            dimensions=(),
            data=pge_runconfig.product_path_group.product_version,
            fillvalue=None,
            description="Version of the product.",
            attrs=dict(units="unitless"),
        )
        # software_version
        _create_dataset(
            group=identification_group,
            name="software_version",
            dimensions=(),
            data=dolphin_version,
            fillvalue=None,
            description="Version of the Dolphin software used to generate the product.",
            attrs=dict(units="unitless"),
        )

        # TODO: prob should just make a _to_string method?
        ss = StringIO()
        pge_runconfig.to_yaml(ss)
        runconfig_str = ss.getvalue()
        _create_dataset(
            group=identification_group,
            name="pge_runconfig",
            dimensions=(),
            data=runconfig_str,
            fillvalue=None,
            description=(
                "The full PGE runconfig YAML file used to generate the product."
            ),
            attrs=dict(units="unitless"),
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
) -> h5netcdf.Variable:
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
) -> tuple[h5netcdf.Variable, h5netcdf.Variable]:
    """Create the x and y coordinate datasets."""
    y, x = _create_yx_arrays(gt, shape)

    if not group.dimensions:
        group.dimensions = dict(y=y.size, x=x.size)
    # Create the datasets
    y_ds = group.create_variable("y", ("y",), data=y, dtype=float)
    x_ds = group.create_variable("x", ("x",), data=x, dtype=float)

    for name, ds in zip(["y", "x"], [y_ds, x_ds]):
        ds.attrs["standard_name"] = f"projection_{name}_coordinate"
        ds.attrs["long_name"] = f"{name.replace('_', ' ')} coordinate of projection"
        ds.attrs["units"] = "m"

    return y_ds, x_ds


def _create_grid_mapping(group, crs: pyproj.CRS, gt: list[float]) -> h5netcdf.Variable:
    """Set up the grid mapping variable."""
    # https://github.com/corteva/rioxarray/blob/21284f67db536d9c104aa872ab0bbc261259e59e/rioxarray/rioxarray.py#L34
    dset = group.create_variable(GRID_MAPPING_DSET, (), data=0, dtype=int)

    dset.attrs.update(crs.to_cf())
    # Also add the GeoTransform
    gt_string = " ".join([str(x) for x in gt])
    dset.attrs.update(
        dict(
            GeoTransform=gt_string,
            units="unitless",
            long_name=(
                "Dummy variable containing geo-referencing metadata in attributes"
            ),
        )
    )

    return dset


def create_compressed_products(comp_slc_dict: dict[str, Path], output_dir: Filename):
    """Make the compressed SLC output product."""

    def form_name(filename: Path, burst: str):
        # filename: compressed_20180222_20180716.tif
        date_str = io._format_date_pair(*get_dates(filename.stem))
        return COMPRESSED_SLC_TEMPLATE.format(burst=burst, date_str=date_str)

    attrs = GLOBAL_ATTRS.copy()
    attrs["title"] = "Compressed SLC"
    *parts, dset_name = OPERA_DATASET_NAME.split("/")
    group_name = "/".join(parts)

    for burst, comp_slc_file in comp_slc_dict.items():
        outname = Path(output_dir) / form_name(comp_slc_file, burst)
        if outname.exists():
            logger.info(f"Skipping existing {outname}")
            continue

        crs = io.get_raster_crs(comp_slc_file)
        gt = io.get_raster_gt(comp_slc_file)
        data = io.load_gdal(comp_slc_file)
        truncate_mantissa(data)

        logger.info(f"Writing {outname}")
        with h5py.File(outname, "w") as hf:
            # add type to root for GDAL recognition of complex datasets in NetCDF
            ctype = h5py.h5t.py_create(np.complex64)
            ctype.commit(hf["/"].id, np.string_("complex64"))

        with h5netcdf.File(outname, mode="a", invalid_netcdf=True) as f:
            f.attrs.update(attrs)

            data_group = f.create_group(group_name)
            _create_grid_mapping(group=data_group, crs=crs, gt=gt)
            _create_yx_dsets(group=data_group, gt=gt, shape=data.shape)
            _create_geo_dataset(
                group=data_group,
                name=dset_name,
                data=data,
                description="Compressed SLC product",
                fillvalue=np.nan + 0j,
                attrs=dict(units="unitless"),
            )


def make_browse_image(
    output_filename: Filename,
    arr: ArrayLike,
    max_dim_allowed: int = 2048,
) -> None:
    """Create a PNG browse image for the output product.

    Parameters
    ----------
    output_filename : Filename
        Name of output PNG
    arr : ArrayLike
        input 2D image array
    max_dim_allowed : int, default = 2048
        Size (in pixels) of the maximum allowed dimension of output image.
        Image gets rescaled with same aspect ratio.
    """
    orig_shape = arr.shape
    scaling_ratio = max([s / max_dim_allowed for s in orig_shape])
    # scale original shape by scaling ratio
    scaled_shape = [int(np.ceil(s / scaling_ratio)) for s in orig_shape]

    # TODO: Make actual browse image
    dummy = np.zeros(scaled_shape, dtype="uint8")
    img = Image.fromarray(dummy, mode="L")
    img.save(output_filename, transparency=0)
