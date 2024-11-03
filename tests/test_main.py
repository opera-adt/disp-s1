from contextlib import chdir
from pathlib import Path

import pytest

from disp_s1.cli.run import run_main

TEST_DATA_DIR = Path(__file__).parent / "data/delivery_data_small"


def test_run_main():
    config_file = TEST_DATA_DIR / "config_files/runconfig_historical.yaml"
    if not config_file.exists():
        pytest.skip(f"Test data does not exist at {TEST_DATA_DIR}")

    with chdir(TEST_DATA_DIR):
        run_main(config_file="config_files/runconfig_historical.yaml")
