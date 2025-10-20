"""Tests for recomputing perpendicular baseline from saved orbit data."""

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
    url = "https://datapool.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20250408T163512Z.nc"
    file = Path() / Path(url).name
    if not file.exists():
        import subprocess

        subprocess.run(["wget", url], check=True)
    return file


@pytest.fixture(scope="module")
def recomputed_file(tmp_path_factory):
    """Recompute the baseline once and reuse across all tests."""
    test_file = Path(
        "./OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20250318T222753Z.nc"
    )
    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    tmp_dir = tmp_path_factory.mktemp("baseline_test")
    output_file = tmp_dir / "test_output.nc"
    # Don't update metadata in tests to keep tests deterministic
    recompute_perpendicular_baseline(
        test_file, output_file, subsample=50, update_metadata=False
    )
    return output_file


def test_recompute_baseline_matches_original(test_file, recomputed_file):
    """Test that recomputed baseline matches the original within tolerance."""
    # Read the original baseline
    with h5py.File(test_file) as f:
        original_baseline = f["/corrections/perpendicular_baseline"][:]

    # Read the recomputed baseline
    with h5py.File(recomputed_file) as f:
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


def test_recompute_baseline_preserves_metadata(test_file, recomputed_file):
    """Test that other metadata is preserved when recomputing baseline."""
    # Check that key metadata is preserved
    with h5py.File(test_file) as f_orig, h5py.File(recomputed_file) as f_new:
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


def test_recompute_baseline_attributes(recomputed_file):
    """Test that the recomputed baseline has the correct attributes."""
    with h5py.File(recomputed_file) as f:
        baseline_dset = f["/corrections/perpendicular_baseline"]

        # Check attributes
        assert "long_name" in baseline_dset.attrs
        assert "units" in baseline_dset.attrs
        units = baseline_dset.attrs["units"]
        if isinstance(units, bytes):
            units = units.decode("utf-8")
        assert units == "meters"
        assert "grid_mapping" in baseline_dset.attrs


def test_metadata_update(test_file, tmp_path):
    """Test that metadata timestamps are updated correctly."""
    output_file = tmp_path / "test_metadata.nc"

    # Get original metadata
    with h5py.File(test_file) as f:
        original_datetime = f["/identification/processing_start_datetime"][()].decode(
            "utf-8"
        )
        original_version = f["/identification/product_version"][()].decode("utf-8")

    # Recompute with metadata update
    recompute_perpendicular_baseline(
        test_file, output_file, subsample=50, update_metadata=True, update_version=False
    )

    # Check that processing_start_datetime was updated
    with h5py.File(output_file) as f:
        new_datetime = f["/identification/processing_start_datetime"][()].decode(
            "utf-8"
        )
        new_version = f["/identification/product_version"][()].decode("utf-8")

    assert new_datetime != original_datetime
    assert new_version == original_version


def test_version_update(test_file, tmp_path):
    """Test that product version is updated correctly."""
    output_file = tmp_path / "test_version.nc"

    # Get original version
    with h5py.File(test_file) as f:
        original_version = f["/identification/product_version"][()].decode("utf-8")

    # Recompute with version update
    recompute_perpendicular_baseline(
        test_file,
        output_file,
        subsample=50,
        update_metadata=True,
        update_version=True,
        new_version="1.1",
    )

    # Check that version was updated
    with h5py.File(output_file) as f:
        new_version = f["/identification/product_version"][()].decode("utf-8")

    assert new_version == "1.1"
    assert new_version != original_version


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
