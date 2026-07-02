from pathlib import Path

import h5py
import numpy as np
import pytest
import shapely
from dolphin import io
from dolphin.workflows import UnwrapOptions
from shapely import MultiPolygon, from_wkt

from disp_s1._utils import (
    METERS_TO_RADIANS,
    _convert_meters_to_radians,
    _create_correlation_images,
    _update_snaphu_conncomps,
    _update_spurt_conncomps,
    extract_footprint,
    split_on_antimeridian,
)

DATA_DIR = Path(__file__).parent / "data"
UNW_FILE = DATA_DIR / "20160716_20160809.unw.tif"


@pytest.fixture
def ts_filenames(tmp_path) -> list[Path]:
    end_date = 20160809
    filenames: list[Path] = []

    # Make 2d, 512, 512 ramp for the data:
    ramp_rad = (np.arange(0, 512).reshape(512, 1) / 10) * np.ones((1, 512))
    ramp_rad += np.random.randn(*ramp_rad.shape)

    rad_to_meters = 1 / METERS_TO_RADIANS
    ramp_meters = ramp_rad * rad_to_meters

    for i in range(3):
        cur_file = tmp_path / f"20160716_{end_date + i}.tif"
        filenames.append(cur_file)
        io.write_arr(arr=ramp_meters, like_filename=UNW_FILE, output_name=cur_file)

    return filenames


def test_create_correlations(ts_filenames):
    output_paths = _create_correlation_images(
        ts_filenames=ts_filenames,
        num_workers=1,
    )
    assert len(output_paths) == 3
    for path in output_paths:
        assert path.exists()
        assert path.suffix == ".tif"
        data = io.load_gdal(path)
        assert data.shape == (512, 512)
        assert np.all(data >= 0)
        assert np.all(data <= 1)
        # Because we made a ramp with ~5 fringes, and some noise, the sliding
        # window should get mid correlation
        assert 0.2 < data.mean() < 0.8


def test_convert_meters_to_radians(ts_filenames):
    """Test that _convert_meters_to_radians correctly creates scaled files."""
    unw_scaled_paths = _convert_meters_to_radians(ts_filenames)

    assert len(unw_scaled_paths) == 3
    assert unw_scaled_paths[0] == ts_filenames[0].with_suffix(".radians.tif")
    assert unw_scaled_paths[0].exists()

    # Read the data through the VRT to verify the scaling
    expected_scale_factor = METERS_TO_RADIANS
    for unw_p, ts_p in zip(unw_scaled_paths, ts_filenames, strict=True):
        scaled_data = io.load_gdal(unw_p)
        disp_meters = io.load_gdal(ts_p)

        # Check that the data was correctly scaled
        expected_disp_rad = disp_meters * expected_scale_factor
        assert np.allclose(
            scaled_data, expected_disp_rad
        ), "Data read through VRT does not match expected scaled values"


def test_update_snaphu_conncomps(ts_filenames):
    cor_paths = _create_correlation_images(
        ts_filenames=ts_filenames,
        num_workers=1,
    )
    mask_path = str(ts_filenames[0]) + ".mask.tif"
    cols, rows = io.get_raster_xysize(ts_filenames[0])
    io.write_arr(
        arr=np.ones((rows, cols), dtype=bool),
        like_filename=UNW_FILE,
        output_name=mask_path,
    )

    _update_snaphu_conncomps(
        timeseries_paths=ts_filenames,
        stitched_cor_paths=cor_paths,
        mask_filename=mask_path,
        unwrap_options=UnwrapOptions(),
        nlooks=50,
    )


@pytest.fixture
def setup_test_files(tmp_path):
    """Create temporary test files mimicking the timeseries and conncomp structure."""
    # Create directories
    unwrap_dir = tmp_path / "unwrapped"
    ts_dir = tmp_path / "timeseries"
    unwrap_dir.mkdir()
    ts_dir.mkdir()

    # Create timeseries paths (these are the target names we want)
    ts_dates = [
        "20160708_20170603",
        "20160708_20170615",
        "20160708_20170627",
        "20160708_20170709",
        "20170709_20170721",
        "20170709_20170814",
        "20170709_20170826",
        "20170709_20170907",
        "20170709_20170919",
        "20170709_20171001",
        "20170709_20171013",
        "20170709_20171025",
        "20170709_20171106",
        "20170709_20171118",
        "20170709_20171130",
    ]
    ts_paths = []
    # Create empty test files for timeseries outputs
    for date in ts_dates:
        path = ts_dir / f"{date}.tif"
        np.zeros((1, 1)).tofile(path)
        ts_paths.append(path)

    # Create some conncomp files (including the ones that caused issues)
    cc_dates = [
        "20160708_20170603",
        "20160708_20170615",
        "20160708_20170627",
        "20170603_20170615",
        "20170603_20170627",
        "20170603_20170709",
        "20170615_20170627",
        "20170615_20170709",
        "20170615_20170721",
        "20170627_20170709",
    ]
    # Create empty test files for conncomps created during unwrapping
    cc_paths = []
    for date in cc_dates:
        path = unwrap_dir / f"{date}.unw.conncomp.tif"
        np.zeros((1, 1)).tofile(path)
        cc_paths.append(path)

    return ts_paths, cc_paths


def test_update_spurt_conncomps(setup_test_files):
    ts_paths, cc_paths = setup_test_files

    updated_paths = _update_spurt_conncomps(ts_paths, cc_paths[0])

    assert len(updated_paths) == len(
        ts_paths
    ), "Output length should match timeseries length"

    # Test 2: Verify all output paths exist
    exist_status = [p.exists() for p in updated_paths]
    assert all(
        exist_status
    ), f"Missing files at indices: {[i for i, x in enumerate(exist_status) if not x]}"

    # Check that output paths match timeseries date patterns
    for ts_p, cc_p in zip(ts_paths, updated_paths):
        ts_stem = ts_p.stem  # e.g. "20160708_20170603"
        assert cc_p.stem.startswith(ts_stem)
        assert cc_p.name.endswith(".unw.conncomp.tif"), f"Wrong extension for {cc_p}"


def test_extract_footprint_gdal_canary(tmp_path):
    """GDAL-version canary for ``extract_footprint``.

    Writes a synthetic raster with a *diagonal* band of valid pixels (an
    axis-aligned block would hide orientation/rotation bugs) and asserts the
    footprint is a single rotated-rectangle MULTIPOLYGON that contains every
    valid cell. If a future GDAL changes ``gdal.Footprint`` -- re-introducing
    simplification, flipping orientation, or altering the convex hull so the box
    no longer covers the data -- this test will flag it.
    """
    n = 30
    arr = np.zeros((n, n), dtype="uint8")
    rr, cc = np.mgrid[0:n, 0:n]
    arr[np.abs(rr - cc) < 4] = 1  # diagonal stripe

    # Simple north-up EPSG:4326 georeferencing (1 px = 0.01 deg).
    gt = (-120.0, 0.01, 0.0, 40.0, 0.0, -0.01)
    tif = tmp_path / "synthetic.tif"
    io.write_arr(
        arr=arr,
        output_name=tif,
        geotransform=gt,
        projection="EPSG:4326",
        dtype="uint8",
        nodata=0,
    )

    geom = from_wkt(extract_footprint(tif))
    assert geom.geom_type == "MultiPolygon"
    assert len(geom.geoms[0].exterior.coords) == 5  # rotated rectangle

    vy, vx = np.where(arr)
    lon = gt[0] + (vx + 0.5) * gt[1]
    lat = gt[3] + (vy + 0.5) * gt[5]
    assert shapely.contains(geom, shapely.points(lon, lat)).all()


def test_recompute_matches_forward_footprint(tmp_path):
    """The recompute script reproduces the forward bounding polygon.

    Wraps one synthetic array into a GeoTIFF (the forward ``unw`` input that
    ``product.py`` footprints) and a CF NetCDF (the product the recompute script
    reads back), then asserts the two bounding polygons are identical. This locks
    in forward/recompute parity and fails if a future GDAL's NetCDF y-flip ever
    leaks through the h5py read path. No external data needed.
    """
    import sys

    from pyproj import CRS

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from recompute_product_bounds import compute_bounding_polygon

    # Small diagonal valid band (value 1, else 0) -> non-trivial rotated rect.
    n = 30
    arr = np.zeros((n, n), dtype="float32")
    rr, cc = np.mgrid[0:n, 0:n]
    arr[np.abs(rr - cc) < 4] = 1.0

    gt = (302070.0, 10.0, 0.0, 4058360.0, 0.0, -10.0)
    crs = CRS.from_epsg(32638)

    # Forward "unw" GeoTIFF (footprinted directly by product.py).
    geotiff = tmp_path / "unw.tif"
    io.write_arr(
        arr=arr,
        output_name=geotiff,
        geotransform=gt,
        projection=crs.to_wkt(),
        dtype="float32",
        nodata=0,
    )

    # Product NetCDF wrapping the same data in CF (y, x) layout.
    x = gt[0] + (np.arange(n) + 0.5) * gt[1]
    y = gt[3] + (np.arange(n) + 0.5) * gt[5]
    nc = tmp_path / "product.nc"
    with h5py.File(nc, "w") as f:
        f.attrs["Conventions"] = "CF-1.8"
        xds = f.create_dataset("x", data=x)
        xds.make_scale()
        yds = f.create_dataset("y", data=y)
        yds.make_scale()
        dds = f.create_dataset("displacement", data=arr)
        dds.dims[0].attach_scale(yds)
        dds.dims[1].attach_scale(xds)
        dds.attrs["grid_mapping"] = "spatial_ref"
        sr = f.create_dataset("spatial_ref", data=0)
        sr.attrs["crs_wkt"] = crs.to_wkt()

    forward = from_wkt(extract_footprint(geotiff))
    recompute = from_wkt(compute_bounding_polygon(nc))
    assert forward.equals(recompute)


def test_check_split_on_antimeridian():
    """Test a polygon not on the antimeridian."""
    polygon = from_wkt(
        "POLYGON ((-111.343848786559 33.167961010325, -111.418794476835"
        " 32.8232467872268, -110.528205605868 32.6834192684192, -110.510122005303"
        " 32.8302954024662, -110.475359683262 32.9094439302636, -110.455061760006"
        " 33.0281538694422, -111.343848786559 33.167961010325))"
    )
    multipoly = split_on_antimeridian(polygon)

    assert len(multipoly.geoms) == 1
    assert multipoly.geoms[0] == polygon
    assert multipoly.area == polygon.area


def test_split_on_antimeridian_input_multipolygon():
    polygon = from_wkt(
        "POLYGON ((-111.343848786559 33.167961010325, -111.418794476835"
        " 32.8232467872268, -110.528205605868 32.6834192684192, -110.510122005303"
        " 32.8302954024662, -110.475359683262 32.9094439302636, -110.455061760006"
        " 33.0281538694422, -111.343848786559 33.167961010325))"
    )
    multipoly = MultiPolygon([polygon])

    multipoly = split_on_antimeridian(multipoly)
    assert len(multipoly.geoms) == 1
    assert multipoly.geoms[0] == polygon
    assert multipoly.area == polygon.area


def test_dateline_crossing():
    # Polygon on Ross Ice Shelf (Antarctica)
    # crossing the dateline
    # Example taken from isce3 unit test
    polygon_wkt = (
        "POLYGON((-160.9795 -76.9215,163.3981 -77.0962,"
        "152.885 -81.8908,-149.3722 -81.6129,-160.9795 -76.9215))"
    )
    polygon = from_wkt(polygon_wkt)
    in_multipoly = MultiPolygon([polygon])

    # Check if crossing dateline
    multipoly = split_on_antimeridian(in_multipoly)

    assert len(multipoly.geoms) == 2

    # Test failing frame over alaska
    polygon = from_wkt(
        "POLYGON ((-179.695183609472 52.2562453232354,"
        " 179.99794861982 52.2563010695419,"
        " 179.158193448683 52.173216336064, 179.172068656626 52.1157703763663,"
        " 177.933195522318 51.9815738118871, 177.848811722813 52.2543703653295,"
        " 176.622541622451 52.1084712235005, 176.744861163101 51.7385106813533,"
        " -179.99953210322 51.7384557700677, -179.57485943579 51.7791857416867,"
        " -179.695183609472 52.2562453232354))"
    )
    multipoly = split_on_antimeridian(polygon)
    assert len(multipoly.geoms) == 2
    assert (
        multipoly.area < 0.1 * polygon.area
    )  # Original failing case had area of hundreds
