from pathlib import Path

import h5py
import pytest

from disp_s1.pge_runconfig import RunConfig
from disp_s1.product import create_compressed_products, create_output_product

TEST_FILE = (
    Path(__file__).parent
    / "data"
    / "OPERA_L2_CSLC-S1_T087-185683-IW2_20221228T161651Z_20240504T181714Z_S1A_VV_v1.1.h5"  # noqa: E501
)


# Shapely runtime warning
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_create_output_product(
    tmp_path,
    test_data_dir,
    scratch_dir,
):
    # Get the test run scratch directory
    # $ ls scratch/forward/
    # interferograms  log_sas.log  t042_088905_iw1  t042_088906_iw1  unwrapped
    # $ ls scratch/forward/unwrapped/
    # 20221119_20221213.unw.conncomp  20221119_20221213.unw.conncomp.hdr  ...
    unw_dir = scratch_dir / "forward/unwrapped"
    unw_filename = unw_dir / "20221119_20221213.unw.tif"
    conncomp_filename = unw_dir / "20221119_20221213.unw.conncomp"

    ifg_dir = scratch_dir / "forward/interferograms/stitched"
    tcorr_filename = ifg_dir / "tcorr.tif"
    ps_mask_filename = ifg_dir / "ps_mask_looked.tif"
    ifg_corr_filename = ifg_dir / "20221119_20221213.cor"

    cslc_files = sorted(
        (test_data_dir / "delivery_data_small/input_slcs").glob("t*.h5")
    )
    assert len(cslc_files) > 0

    output_name = tmp_path / "20221119_20221213.unw.nc"

    pge_runconfig = RunConfig.from_yaml(
        test_data_dir / "delivery_data_small/config_files/runconfig_forward.yaml"
    )

    create_output_product(
        unw_filename=unw_filename,
        conncomp_filename=conncomp_filename,
        tcorr_filename=tcorr_filename,
        ifg_corr_filename=ifg_corr_filename,
        output_name=output_name,
        corrections={},
        ps_mask_filename=ps_mask_filename,
        pge_runconfig=pge_runconfig,
        cslc_files=cslc_files,
    )


def test_create_compressed_slc(tmp_path):
    # date_str = "20220101_20220102_20220103"
    # T087-185683-IW2
    burst = "t087_185683_iw2"
    comp_slc = TEST_FILE
    comp_slc_dict = {burst: [comp_slc]}

    create_compressed_products(comp_slc_dict, output_dir=tmp_path)

    expected_name = tmp_path / (comp_slc.with_suffix(".h5").name)
    assert expected_name.exists()
    # Check product structure
    with h5py.File(expected_name) as hf:
        assert hf["/data/VV"].size > 0
        assert "/metadata/orbit" in hf
        assert "/identification/zero_doppler_start_time" in hf
        assert "/metadata/processing_information/input_burst_metadata/wavelength" in hf
