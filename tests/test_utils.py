from math import pi
from pathlib import Path

import numpy as np
import pytest
from dolphin import io
from dolphin.constants import SENTINEL_1_WAVELENGTH
from dolphin.workflows import UnwrapOptions

from disp_s1._utils import (
    _create_correlation_images,
    _update_snaphu_conncomps,
    _update_spurt_conncomps,
)

DATA_DIR = Path(__file__).parent / "data"
UNW_FILE = DATA_DIR / "20160716_20160809.unw.tif"


@pytest.fixture
def ts_filenames(tmp_path):
    end_date = 20160809
    filenames = []

    # Make 2d, 512, 512 ramp for the data:
    ramp_rad = (np.arange(0, 512).reshape(512, 1) / 10) * np.ones((1, 512))
    ramp_rad += np.random.randn(*ramp_rad.shape)

    rad_to_meters = SENTINEL_1_WAVELENGTH / (-4 * pi)
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
    for date in ts_dates:
        path = ts_dir / f"{date}.tif"
        # Create empty test files
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
    cc_paths = []
    for date in cc_dates:
        path = unwrap_dir / f"{date}.unw.conncomp.tif"
        # Create empty test files
        np.zeros((1, 1)).tofile(path)
        cc_paths.append(path)

    return ts_paths, cc_paths


def test_update_spurt_conncomps(setup_test_files):
    """Test the _update_spurt_conncomps function."""
    ts_paths, cc_paths = setup_test_files

    # Run the function
    updated_paths = _update_spurt_conncomps(ts_paths, cc_paths[0])

    # Test 1: Check that output length matches input timeseries length
    assert len(updated_paths) == len(
        ts_paths
    ), "Output length should match timeseries length"

    # Test 2: Verify all output paths exist
    exist_status = [p.exists() for p in updated_paths]
    assert all(
        exist_status
    ), f"Missing files at indices: {[i for i, x in enumerate(exist_status) if not x]}"

    # Test 3: Check that output paths match timeseries date patterns
    for ts_p, cc_p in zip(ts_paths, updated_paths):
        ts_stem = ts_p.stem  # e.g. "20160708_20170603"
        assert cc_p.stem.startswith(ts_stem)

    # Test 4: Check correct extensions on output files
    for p in updated_paths:
        assert p.name.endswith(".unw.conncomp.tif"), f"Wrong extension for {p}"
