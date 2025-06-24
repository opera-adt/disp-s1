from contextlib import chdir
from datetime import datetime
from pathlib import Path

import pytest
from dolphin.workflows.displacement import OutputPaths

from disp_s1._reference import ReferencePoint
from disp_s1.cli.run import run_main
from disp_s1.main import _filter_before_last_processed

TEST_DATA_DIR = Path(__file__).parent / "data/delivery_data_small"


def test_run_main():
    config_file = TEST_DATA_DIR / "config_files/runconfig_historical.yaml"
    if not config_file.exists():
        pytest.skip(f"Test data does not exist at {TEST_DATA_DIR}")

    with chdir(TEST_DATA_DIR):
        run_main(config_file="config_files/runconfig_historical.yaml")


@pytest.fixture
def out_paths():
    """Test results from delivery_data_small run."""
    return OutputPaths(
        comp_slc_dict={
            "t027_056725_iw1": [
                Path(
                    "t027_056725_iw1/linked_phase/compressed_20170430_20170217_20170430.tif"
                )
            ],
            "t027_056726_iw1": [
                Path(
                    "t027_056726_iw1/linked_phase/compressed_20170430_20170217_20170430.tif"
                )
            ],
        },
        stitched_ifg_paths=[
            Path("interferograms/20170217_20170301.int.tif"),
            Path("interferograms/20170217_20170313.int.tif"),
            Path("interferograms/20170217_20170325.int.tif"),
            Path("interferograms/20170301_20170313.int.tif"),
            Path("interferograms/20170301_20170325.int.tif"),
            Path("interferograms/20170301_20170406.int.tif"),
            Path("interferograms/20170313_20170325.int.tif"),
            Path("interferograms/20170313_20170406.int.tif"),
            Path("interferograms/20170313_20170418.int.tif"),
            Path("interferograms/20170325_20170406.int.tif"),
            Path("interferograms/20170325_20170418.int.tif"),
            Path("interferograms/20170325_20170430.int.tif"),
            Path("interferograms/20170406_20170418.int.tif"),
            Path("interferograms/20170406_20170430.int.tif"),
            Path("interferograms/20170418_20170430.int.tif"),
        ],
        stitched_cor_paths=[
            Path("interferograms/20170217_20170301.int.cor.tif"),
            Path("interferograms/20170217_20170313.int.cor.tif"),
            Path("interferograms/20170217_20170325.int.cor.tif"),
            Path("interferograms/20170301_20170313.int.cor.tif"),
            Path("interferograms/20170301_20170325.int.cor.tif"),
            Path("interferograms/20170301_20170406.int.cor.tif"),
            Path("interferograms/20170313_20170325.int.cor.tif"),
            Path("interferograms/20170313_20170406.int.cor.tif"),
            Path("interferograms/20170313_20170418.int.cor.tif"),
            Path("interferograms/20170325_20170406.int.cor.tif"),
            Path("interferograms/20170325_20170418.int.cor.tif"),
            Path("interferograms/20170325_20170430.int.cor.tif"),
            Path("interferograms/20170406_20170418.int.cor.tif"),
            Path("interferograms/20170406_20170430.int.cor.tif"),
            Path("interferograms/20170418_20170430.int.cor.tif"),
        ],
        stitched_temp_coh_file=Path("interferograms/temporal_coherence.tif"),
        stitched_ps_file=Path("interferograms/ps_mask_looked.tif"),
        stitched_amp_dispersion_file=Path("interferograms/amp_dispersion_looked.tif"),
        stitched_shp_count_file=Path("interferograms/shp_counts.tif"),
        stitched_similarity_file=Path("interferograms/similarity.tif"),
        unwrapped_paths=[
            Path("unwrapped/20170217_20170301.unw.tif"),
            Path("unwrapped/20170217_20170313.unw.tif"),
            Path("unwrapped/20170217_20170325.unw.tif"),
            Path("unwrapped/20170301_20170313.unw.tif"),
            Path("unwrapped/20170301_20170325.unw.tif"),
            Path("unwrapped/20170301_20170406.unw.tif"),
            Path("unwrapped/20170313_20170325.unw.tif"),
            Path("unwrapped/20170313_20170406.unw.tif"),
            Path("unwrapped/20170313_20170418.unw.tif"),
            Path("unwrapped/20170325_20170406.unw.tif"),
            Path("unwrapped/20170325_20170418.unw.tif"),
            Path("unwrapped/20170325_20170430.unw.tif"),
            Path("unwrapped/20170406_20170418.unw.tif"),
            Path("unwrapped/20170406_20170430.unw.tif"),
            Path("unwrapped/20170418_20170430.unw.tif"),
        ],
        conncomp_paths=[
            Path("unwrapped/20170217_20170301.unw.conncomp.tif"),
            Path("unwrapped/20170217_20170313.unw.conncomp.tif"),
            Path("unwrapped/20170217_20170325.unw.conncomp.tif"),
            Path("unwrapped/20170301_20170313.unw.conncomp.tif"),
            Path("unwrapped/20170301_20170325.unw.conncomp.tif"),
            Path("unwrapped/20170301_20170406.unw.conncomp.tif"),
            Path("unwrapped/20170313_20170325.unw.conncomp.tif"),
            Path("unwrapped/20170313_20170406.unw.conncomp.tif"),
            Path("unwrapped/20170313_20170418.unw.conncomp.tif"),
            Path("unwrapped/20170325_20170406.unw.conncomp.tif"),
            Path("unwrapped/20170325_20170418.unw.conncomp.tif"),
            Path("unwrapped/20170325_20170430.unw.conncomp.tif"),
            Path("unwrapped/20170406_20170418.unw.conncomp.tif"),
            Path("unwrapped/20170406_20170430.unw.conncomp.tif"),
            Path("unwrapped/20170418_20170430.unw.conncomp.tif"),
        ],
        timeseries_paths=[
            Path("timeseries/20170217_20170301.tif"),
            Path("timeseries/20170217_20170313.tif"),
            Path("timeseries/20170217_20170325.tif"),
            Path("timeseries/20170217_20170406.tif"),
            Path("timeseries/20170217_20170418.tif"),
            Path("timeseries/20170217_20170430.tif"),
        ],
        timeseries_residual_paths=[
            Path("timeseries/residuals_20170217_20170301.tif"),
            Path("timeseries/residuals_20170217_20170313.tif"),
            Path("timeseries/residuals_20170217_20170325.tif"),
            Path("timeseries/residuals_20170217_20170406.tif"),
            Path("timeseries/residuals_20170217_20170418.tif"),
            Path("timeseries/residuals_20170217_20170430.tif"),
        ],
        reference_point=ReferencePoint(row=4163, col=7527, lat=37.5, lon=-122.3),
    )


def test_filter_before_last_processed(out_paths: OutputPaths):
    """Test the _filter_before_last_processed function filters OutputPaths correctly."""

    # Test filtering with last_processed = 2017-04-06 (middle of the date range)
    # This should keep files with secondary date >= 2017-04-06
    last_processed = datetime(2017, 4, 6)
    filtered_paths = _filter_before_last_processed(out_paths, last_processed)

    # Expected files to keep (secondary date >= 2017-04-06):
    # - 20170301_20170406 (Apr 6)
    # - 20170313_20170406 (Apr 6)
    # - 20170313_20170418 (Apr 18)
    # - 20170325_20170406 (Apr 6)
    # - 20170325_20170418 (Apr 18)
    # - 20170325_20170430 (Apr 30)
    # - 20170406_20170418 (Apr 18)
    # - 20170406_20170430 (Apr 30)
    # - 20170418_20170430 (Apr 30)

    expected_ifg_paths = [
        Path("interferograms/20170301_20170406.int.tif"),
        Path("interferograms/20170313_20170406.int.tif"),
        Path("interferograms/20170313_20170418.int.tif"),
        Path("interferograms/20170325_20170406.int.tif"),
        Path("interferograms/20170325_20170418.int.tif"),
        Path("interferograms/20170325_20170430.int.tif"),
        Path("interferograms/20170406_20170418.int.tif"),
        Path("interferograms/20170406_20170430.int.tif"),
        Path("interferograms/20170418_20170430.int.tif"),
    ]

    assert filtered_paths.stitched_ifg_paths == expected_ifg_paths
    assert len(filtered_paths.stitched_cor_paths) == len(expected_ifg_paths)
    assert len(filtered_paths.conncomp_paths) == len(expected_ifg_paths)
    # unwrapped_paths is not filtered, so it should remain unchanged
    assert filtered_paths.unwrapped_paths == out_paths.unwrapped_paths

    # Timeseries paths should keep those with secondary date >= 2017-04-06
    expected_timeseries = [
        Path("timeseries/20170217_20170406.tif"),
        Path("timeseries/20170217_20170418.tif"),
        Path("timeseries/20170217_20170430.tif"),
    ]
    assert filtered_paths.timeseries_paths == expected_timeseries
    assert len(filtered_paths.timeseries_residual_paths) == len(expected_timeseries)


def test_filter_before_last_processed_all_dates(out_paths: OutputPaths):
    """Test the _filter_before_last_processed function filters OutputPaths correctly."""
    # Test with last_processed before all dates - should keep everything
    last_processed_early = datetime(2017, 2, 1)
    filtered_all = _filter_before_last_processed(out_paths, last_processed_early)

    assert filtered_all.stitched_ifg_paths == out_paths.stitched_ifg_paths
    assert filtered_all.stitched_cor_paths == out_paths.stitched_cor_paths
    assert filtered_all.conncomp_paths == out_paths.conncomp_paths
    assert filtered_all.unwrapped_paths == out_paths.unwrapped_paths
    assert filtered_all.timeseries_paths == out_paths.timeseries_paths
    assert filtered_all.timeseries_residual_paths == out_paths.timeseries_residual_paths
