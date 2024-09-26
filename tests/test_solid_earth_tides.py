from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dolphin import io

from disp_s1.solid_earth_tides import calculate_solid_earth_tides_correction

TEST_DATA_DIR = Path(__file__).parent / "data"


def test_calculate_solid_earth_tides_correction():
    ifgram_filename = TEST_DATA_DIR / "20160716_20160809.unw.tif"
    los_east_file = TEST_DATA_DIR / "los_east.tif"
    los_north_file = TEST_DATA_DIR / "los_north.tif"
    reference_start_time = pd.to_datetime(datetime(2016, 7, 16, 13, 27, 39, 698599))
    reference_stop_time = pd.to_datetime(datetime(2016, 7, 16, 13, 27, 42, 145748))
    secondary_start_time = pd.to_datetime(datetime(2016, 8, 9, 10, 45, 20, 562106))
    secondary_stop_time = pd.to_datetime(datetime(2016, 8, 9, 10, 45, 23, 9255))

    arr0 = io.load_gdal(ifgram_filename)

    solid_earth_t = calculate_solid_earth_tides_correction(
        ifgram_filename,
        reference_start_time,
        reference_stop_time,
        secondary_start_time,
        secondary_stop_time,
        los_east_file,
        los_north_file,
    )

    assert solid_earth_t.shape == arr0.shape
    assert np.max(solid_earth_t) < 0.1
