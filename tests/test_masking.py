import zipfile
from pathlib import Path

import numpy.testing as npt
import pytest
from dolphin import io

from disp_s1._masking import create_mask_from_distance


@pytest.fixture
def mask_file_zipped():
    return Path(__file__).parent / "data/water_distance.tif.zip"


@pytest.fixture
def expected_file_zipped():
    return


def test_bounds(tmp_path, mask_file_zipped):
    # Unzip to tmp_path
    with zipfile.ZipFile(mask_file_zipped, "r") as zip_ref:
        zip_ref.extractall(tmp_path)

    # Get the path of the extracted TIF file
    input_tif = tmp_path / "water_distance.tif"
    outpath = tmp_path / "mask.tif"
    create_mask_from_distance(
        input_tif, output_file=outpath, land_buffer=0, ocean_buffer=0
    )
    expected0 = io.load_gdal(Path(__file__).parent / "data/expected_mask0.tif")
    npt.assert_array_equal(expected0, io.load_gdal(outpath))

    outpath = tmp_path / "mask2.tif"
    create_mask_from_distance(
        input_tif, output_file=outpath, land_buffer=2, ocean_buffer=2
    )
    expected2 = io.load_gdal(Path(__file__).parent / "data/expected_mask2.tif")
    npt.assert_array_equal(expected2, io.load_gdal(outpath))
