import h5py
import numpy as np
import pytest
from dolphin import io

from disp_s1.product import create_compressed_products, create_output_product

# random place in hawaii
GEOTRANSFORM = [204500.0, 5.0, 0.0, 2151300.0, 0.0, -10.0]
SRS = "EPSG:32605"
SHAPE = (256, 256)


@pytest.fixture
def unw_filename(tmp_path) -> str:
    data = np.random.randn(*SHAPE).astype(np.float32)
    filename = tmp_path / "unw.tif"
    io.write_arr(
        arr=data, output_name=filename, geotransform=GEOTRANSFORM, projection=SRS
    )
    return filename


@pytest.fixture
def conncomp_filename(tmp_path) -> str:
    data = np.random.randn(*SHAPE).astype(np.uint32)
    filename = tmp_path / "conncomp.tif"
    io.write_arr(
        arr=data, output_name=filename, geotransform=GEOTRANSFORM, projection=SRS
    )
    return filename


@pytest.fixture
def tcorr_filename(tmp_path) -> str:
    data = np.random.randn(*SHAPE).astype(np.float32)
    filename = tmp_path / "tcorr.tif"
    io.write_arr(
        arr=data, output_name=filename, geotransform=GEOTRANSFORM, projection=SRS
    )
    return filename


@pytest.fixture
def ifg_corr_filename(tmp_path) -> str:
    data = np.random.randn(*SHAPE).astype(np.float32)
    filename = tmp_path / "ifg_corr.tif"
    io.write_arr(
        arr=data, output_name=filename, geotransform=GEOTRANSFORM, projection=SRS
    )
    return filename


def test_create_output_product(
    tmp_path,
    unw_filename,
    conncomp_filename,
    tcorr_filename,
    ifg_corr_filename,
):
    output_name = tmp_path / "output_product.nc"

    create_output_product(
        unw_filename=unw_filename,
        conncomp_filename=conncomp_filename,
        tcorr_filename=tcorr_filename,
        ifg_corr_filename=ifg_corr_filename,
        output_name=output_name,
        corrections={},
    )


@pytest.fixture
def comp_slc(tmp_path) -> str:
    data = np.random.randn(*SHAPE).astype(np.complex64)
    date_pair = "20220101_20220102"
    filename = tmp_path / f"compressed_{date_pair}.tif"
    io.write_arr(
        arr=data, output_name=filename, geotransform=GEOTRANSFORM, projection=SRS
    )
    return filename


def test_create_compressed_slc(
    tmp_path,
    comp_slc,
):
    burst = "t123_123456_iw1"
    date_pair = "20220101_20220102"
    comp_slc_dict = {burst: comp_slc}

    create_compressed_products(comp_slc_dict, output_dir=tmp_path)

    expected_name = tmp_path / f"compressed_{burst}_{date_pair}.h5"
    assert expected_name.exists()
    # Check product structure
    with h5py.File(expected_name) as hf:
        assert hf["/data/VV"].size > 0
