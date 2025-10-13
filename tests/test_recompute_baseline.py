"""Tests for recomputing perpendicular baseline from saved orbit data."""

# Import the function from the script
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from recompute_perpendicular_baseline import recompute_perpendicular_baseline


@pytest.fixture
def test_file():
    """Path to the test DISP-S1 file."""
    return Path(
        "./OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"
    )


def test_recompute_baseline_matches_original(test_file, tmp_path):
    """Test that recomputed baseline matches the original within tolerance."""
    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    # Read the original baseline
    with h5py.File(test_file) as f:
        original_baseline = f["/corrections/perpendicular_baseline"][:]

    # Recompute the baseline
    output_file = tmp_path / "test_output.nc"
    recompute_perpendicular_baseline(test_file, output_file, subsample=50)

    # Read the recomputed baseline
    with h5py.File(output_file) as f:
        recomputed_baseline = f["/corrections/perpendicular_baseline"][:]

    # Check that they match within tolerance
    # Use a fairly generous tolerance since we're using interpolation
    # and the computation might have some numerical differences
    mask = ~np.isnan(original_baseline) & ~np.isnan(recomputed_baseline)
    diff = np.abs(original_baseline[mask] - recomputed_baseline[mask])

    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    print(f"Max difference: {max_diff} m")
    print(f"Mean difference: {mean_diff} m")
    print(f"Std difference: {std_diff} m")

    # The baselines should be very close (within 1 meter)
    assert max_diff < 1.0, f"Max difference {max_diff} m exceeds 1.0 m"
    assert mean_diff < 0.1, f"Mean difference {mean_diff} m exceeds 0.1 m"


def test_recompute_baseline_preserves_metadata(test_file, tmp_path):
    """Test that other metadata is preserved when recomputing baseline."""
    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    output_file = tmp_path / "test_output.nc"
    recompute_perpendicular_baseline(test_file, output_file, subsample=50)

    # Check that key metadata is preserved
    with h5py.File(test_file) as f_orig, h5py.File(output_file) as f_new:
        # Check that the grid dimensions are the same
        assert f_orig["/x"][:].shape == f_new["/x"][:].shape
        assert f_orig["/y"][:].shape == f_new["/y"][:].shape

        # Check that other correction layers are preserved
        assert "/corrections/ionospheric_delay" in f_new
        assert "/corrections/solid_earth_tide" in f_new

        # Check that displacement data is preserved
        assert "/displacement" in f_new
        np.testing.assert_array_equal(
            f_orig["/displacement"][:], f_new["/displacement"][:]
        )


def test_recompute_baseline_attributes(test_file, tmp_path):
    """Test that the recomputed baseline has the correct attributes."""
    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    output_file = tmp_path / "test_output.nc"
    recompute_perpendicular_baseline(test_file, output_file, subsample=50)

    with h5py.File(output_file) as f:
        baseline_dset = f["/corrections/perpendicular_baseline"]

        # Check attributes
        assert "long_name" in baseline_dset.attrs
        assert "units" in baseline_dset.attrs
        units = baseline_dset.attrs["units"]
        if isinstance(units, bytes):
            units = units.decode("utf-8")
        assert units == "meters"
        assert "grid_mapping" in baseline_dset.attrs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
