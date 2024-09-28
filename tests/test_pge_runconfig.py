import datetime
import sys
import warnings
from pathlib import Path

import opera_utils
import pytest

from disp_s1.pge_runconfig import (
    AlgorithmParameters,
    DynamicAncillaryFileGroup,
    InputFileGroup,
    PrimaryExecutable,
    ProductPathGroup,
    RunConfig,
    StaticAncillaryFileGroup,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*utcnow.*:DeprecationWarning",
)


def test_algorithm_parameters_schema():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        AlgorithmParameters.print_yaml_schema()


def test_run_config_schema():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        RunConfig.print_yaml_schema()


@pytest.fixture
def input_file_group(slc_file_list_nc_with_sds):
    file_list, subdataset = slc_file_list_nc_with_sds
    return InputFileGroup(cslc_file_list=file_list, frame_id=11114)


@pytest.fixture
def algorithm_parameters_file(tmp_path):
    f = tmp_path / "test.yaml"
    AlgorithmParameters().to_yaml(f)
    return f


@pytest.fixture
def dynamic_ancillary_file_group(algorithm_parameters_file):
    return DynamicAncillaryFileGroup(
        algorithm_parameters_file=algorithm_parameters_file
    )


@pytest.fixture
def frame_to_burst_json_file():
    return opera_utils.datasets.fetch_frame_to_burst_mapping_file()


@pytest.fixture
def reference_date_json_file():
    return Path(__file__).parent / "data/reference-dates.json"


@pytest.fixture
def static_ancillary_file_group(frame_to_burst_json_file, reference_date_json_file):
    return StaticAncillaryFileGroup(
        frame_to_burst_json=frame_to_burst_json_file,
        reference_date_database_json=reference_date_json_file,
    )


@pytest.fixture
def product_path_group(tmp_path):
    product_path = tmp_path / "product_path"
    product_path.mkdir()
    return ProductPathGroup(product_path=product_path)


@pytest.fixture
def runconfig_minimum(
    input_file_group,
    dynamic_ancillary_file_group,
    static_ancillary_file_group,
    product_path_group,
):
    c = RunConfig(
        input_file_group=input_file_group,
        primary_executable=PrimaryExecutable(),
        dynamic_ancillary_file_group=dynamic_ancillary_file_group,
        static_ancillary_file_group=static_ancillary_file_group,
        product_path_group=product_path_group,
    )
    return c


def test_runconfig_to_yaml(runconfig_minimum):
    print(runconfig_minimum.to_yaml(sys.stdout))


def test_runconfig_to_workflow(runconfig_minimum):
    print(runconfig_minimum.to_workflow())


def test_runconfig_from_workflow(
    tmp_path, frame_to_burst_json_file, reference_date_json_file, runconfig_minimum
):
    w = runconfig_minimum.to_workflow()
    frame_id = runconfig_minimum.input_file_group.frame_id
    algo_file = tmp_path / "algo_params.yaml"
    proc_mode = "forward"
    w2 = RunConfig.from_workflow(
        w,
        frame_id=frame_id,
        frame_to_burst_json=frame_to_burst_json_file,
        reference_date_json=reference_date_json_file,
        processing_mode=proc_mode,
        algorithm_parameters_file=algo_file,
    ).to_workflow()

    # these will be slightly different
    w2.creation_time_utc = w.creation_time_utc
    assert w == w2


def test_runconfig_yaml_roundtrip(tmp_path, runconfig_minimum):
    f = tmp_path / "test.yaml"
    runconfig_minimum.to_yaml(f)
    c = RunConfig.from_yaml(f)
    assert c == runconfig_minimum


@pytest.fixture
def hawaii_slc_list():
    # "23210": [
    #   "2016-07-08T16:15:44",
    #   "2017-07-09T16:15:07",
    #   "2018-07-16T16:15:14",
    #   "2019-07-11T16:15:20",
    #   "2020-07-17T16:15:27",
    #   "2021-07-12T16:15:32",
    #   "2022-07-13T16:16:21",
    #   "2023-07-08T16:16:25",
    #   "2024-07-14T16:16:24"
    # ],
    return [
        "COMPRESSED_OPERA_L2_CSLC-S1_T087-185680-IW1_20170610T000000Z_20170610T000000Z_20171001T000000Z_20240429T000000Z_S1B_VV_v1.1.h5",
        # This is the compressed SLC where the "base phase" date within a later
        # reference date. So we expected output index of 1:
        "COMPRESSED_OPERA_L2_CSLC-S1_T087-185680-IW1_20170709T000000Z_20170709T000000Z_20171201T000000Z_20240429T000000Z_S1B_VV_v1.1.h5",
        "COMPRESSED_OPERA_L2_CSLC-S1_T087-185680-IW1_20170709T000000Z_20171210T000000Z_20180604T000000Z_20240429T000000Z_S1B_VV_v1.1.h5",
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180610T161531Z_20240429T233903Z_S1B_VV_v1.1.h5",
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180622T161532Z_20240430T025857Z_S1B_VV_v1.1.h5",
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180628T161614Z_20240430T043443Z_S1A_VV_v1.1.h5",
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180710T161615Z_20240428T043529Z_S1A_VV_v1.1.h5",
        # This one should become the "extra reference":
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180716T161534Z_20240428T062045Z_S1B_VV_v1.1.h5",
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180722T161616Z_20240428T075037Z_S1A_VV_v1.1.h5",
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180728T161534Z_20240428T093746Z_S1B_VV_v1.1.h5",
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180803T161616Z_20240428T110636Z_S1A_VV_v1.1.h5",
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180809T161535Z_20240428T125546Z_S1B_VV_v1.1.h5",
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180815T161617Z_20240428T143339Z_S1A_VV_v1.1.h5",
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180827T161618Z_20240428T175537Z_S1A_VV_v1.1.h5",
    ]


def test_reference_changeover(
    dynamic_ancillary_file_group,
    static_ancillary_file_group,
    product_path_group,
    hawaii_slc_list,
):
    rc = RunConfig(
        input_file_group=InputFileGroup(cslc_file_list=hawaii_slc_list, frame_id=23210),
        primary_executable=PrimaryExecutable(),
        dynamic_ancillary_file_group=dynamic_ancillary_file_group,
        static_ancillary_file_group=static_ancillary_file_group,
        product_path_group=product_path_group,
    )
    cfg = rc.to_workflow()
    assert cfg.output_options.extra_reference_date == datetime.datetime(2018, 7, 16)
    assert cfg.phase_linking.output_reference_idx == 1

    # Check a that non-exact match to the reference still works,
    # AFTER the reference request
    hawaii_slc_list[7] = (
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180717T161534Z_20240428T062045Z_S1B_VV_v1.1.h5"
    )
    rc = RunConfig(
        input_file_group=InputFileGroup(cslc_file_list=hawaii_slc_list, frame_id=23210),
        primary_executable=PrimaryExecutable(),
        dynamic_ancillary_file_group=dynamic_ancillary_file_group,
        static_ancillary_file_group=static_ancillary_file_group,
        product_path_group=product_path_group,
    )
    cfg = rc.to_workflow()
    assert cfg.output_options.extra_reference_date == datetime.datetime(2018, 7, 17)

    # ...but not BEFORE the reference request
    hawaii_slc_list[7] = (
        "OPERA_L2_CSLC-S1_T087-185680-IW1_20180715T161534Z_20240428T062045Z_S1B_VV_v1.1.h5"
    )
    rc = RunConfig(
        input_file_group=InputFileGroup(cslc_file_list=hawaii_slc_list, frame_id=23210),
        primary_executable=PrimaryExecutable(),
        dynamic_ancillary_file_group=dynamic_ancillary_file_group,
        static_ancillary_file_group=static_ancillary_file_group,
        product_path_group=product_path_group,
    )
    cfg = rc.to_workflow()
    assert cfg.output_options.extra_reference_date == datetime.datetime(2018, 7, 22)
