import os
import tarfile
from pathlib import Path

import numpy as np
import pytest
from make_netcdf import create_test_nc

# https://numba.readthedocs.io/en/stable/user/threading-layer.html#example-of-limiting-the-number-of-threads
if not os.environ.get("NUMBA_NUM_THREADS"):
    os.environ["NUMBA_NUM_THREADS"] = str(min(os.cpu_count(), 16))  # type: ignore

from dolphin.phase_link import simulate

NUM_ACQ = 30


# https://github.com/pytest-dev/pytest/issues/667#issuecomment-112206152
@pytest.fixture
def random():
    np.random.seed(1234)
    simulate._seed(1234)


@pytest.fixture(scope="session")
def slc_stack():
    shape = (NUM_ACQ, 5, 10)
    sigma = 0.5
    data = np.random.normal(0, sigma, size=shape) + 1j * np.random.normal(
        0, sigma, size=shape
    )
    data = data.astype(np.complex64)
    return data


@pytest.fixture()
def slc_file_list_nc_with_sds(tmp_path, slc_stack):
    """Save NetCDF files with multiple valid datsets."""
    start_date = 20220101
    d = tmp_path / "nc_with_sds"
    name_template = d / "{date}.nc"
    d.mkdir()
    file_list = []
    subdirs = ["/data", "/data2"]
    ds_name = "VV"
    for i in range(len(slc_stack)):
        fname = str(name_template).format(date=str(start_date + i))
        create_test_nc(
            fname, epsg=32615, subdir=subdirs, data_ds_name=ds_name, data=slc_stack[i]
        )
        # just point to one of them
        file_list.append(fname)

    subdataset = "/data/VV"
    return file_list, subdataset


# @pytest.fixture
# def delivery_data_tar_file():
DATA_DIR = Path(__file__).parent / "data"
DELIVERY_DATA_TAR_FILE = Path(f"{DATA_DIR}/delivery_data_small.tar")
WORKFLOW_SCRATCH_FILE = DATA_DIR / "delivery_data_small_scratch.tar"


def _untar_dir(tmp_path, delivery_data_tar_file: Path, target_dir=None):
    """Untar a all, or a specific directory, from the data file.

    Returns
    -------
    pathlib.Path
        The directory where the specific directory was untarred.
    """
    mode = "r:gz" if delivery_data_tar_file.suffix == ".gz" else "r"
    with tarfile.open(delivery_data_tar_file, mode) as tar:
        if target_dir is None:
            tar.extractall(path=tmp_path)
        else:
            for member in tar.getmembers():
                if member.name.startswith(target_dir):
                    tar.extract(member, path=tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Untar the data file and return the directory where it was untarred.

    Returns
    -------
    pathlib.Path
        The directory where the data was untarred.
    """
    tmpdir = tmp_path_factory.mktemp("test_data")
    return _untar_dir(tmpdir, DELIVERY_DATA_TAR_FILE)


@pytest.fixture(scope="session")
def golden_outputs_dir(tmp_path_factory):
    """Untar the golden outputs/scratch directory from the data file.

    Returns
    -------
    pathlib.Path
        The directory where the specific directory was untarred.
    """
    tmpdir = tmp_path_factory.mktemp("test_data")
    target_dir = "delivery_data_small/golden_output/"
    return _untar_dir(tmpdir, DELIVERY_DATA_TAR_FILE, target_dir)


@pytest.fixture(scope="session")
def scratch_dir(tmp_path_factory):
    """Untar the golden outputs/scratch directory from the data file.

    Returns
    -------
    pathlib.Path
        The directory where the specific directory was untarred.
    """
    tmpdir = tmp_path_factory.mktemp("test_data")
    return _untar_dir(tmpdir, WORKFLOW_SCRATCH_FILE) / "scratch"
