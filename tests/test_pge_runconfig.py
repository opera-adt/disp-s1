import sys
import warnings

import pytest

from disp_s1 import utils
from disp_s1.pge_runconfig import (
    AlgorithmParameters,
    DynamicAncillaryFileGroup,
    InputFileGroup,
    PrimaryExecutable,
    ProductPathGroup,
    RunConfig,
    StaticAncillaryFileGroup,
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
    return InputFileGroup(cslc_file_list=file_list, frame_id=10)


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
def static_ancillary_file_group():
    return StaticAncillaryFileGroup(frame_to_burst_json=utils.FRAME_TO_BURST_JSON_FILE)


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


def test_runconfig_from_workflow(tmp_path, runconfig_minimum):
    w = runconfig_minimum.to_workflow()
    frame_id = runconfig_minimum.input_file_group.frame_id
    algo_file = tmp_path / "algo_params.yaml"
    frame_to_burst_json = utils.FRAME_TO_BURST_JSON_FILE
    proc_mode = "forward"
    w2 = RunConfig.from_workflow(
        w,
        frame_id=frame_id,
        frame_to_burst_json=frame_to_burst_json,
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
