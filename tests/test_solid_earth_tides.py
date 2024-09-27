from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from dolphin import io

from disp_s1.solid_earth_tides import calculate_solid_earth_tides_correction

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize("orbit_direction", ["ascending", "descending"])
def test_calculate_solid_earth_tides_correction(orbit_direction):
    ifgram_filename = TEST_DATA_DIR / "20160716_20160809.unw.tif"
    los_east_file = TEST_DATA_DIR / "los_east.tif"
    los_north_file = TEST_DATA_DIR / "los_north.tif"
    reference_start_time = datetime(2016, 7, 16, 13, 27, 39, 698599)
    reference_stop_time = datetime(2016, 7, 16, 13, 27, 42, 145748)
    secondary_start_time = datetime(2016, 8, 9, 10, 45, 20, 562106)
    secondary_stop_time = datetime(2016, 8, 9, 10, 45, 23, 9255)

    solid_earth_t = calculate_solid_earth_tides_correction(
        ifgram_filename,
        reference_start_time,
        reference_stop_time,
        secondary_start_time,
        secondary_stop_time,
        los_east_file,
        los_north_file,
        orbit_direction=orbit_direction,
    )

    assert solid_earth_t.shape == io.get_raster_xysize(ifgram_filename)[::-1]
    assert np.nanmax(solid_earth_t) < 0.1
