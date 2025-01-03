from math import pi
from pathlib import Path

import numpy as np
import pytest
from dolphin import io
from dolphin.constants import SENTINEL_1_WAVELENGTH
from dolphin.workflows import UnwrapOptions

from disp_s1._utils import _create_correlation_images, _update_snaphu_conncomps

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
