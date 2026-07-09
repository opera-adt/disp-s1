"""Tests for the combined baseline + bounding-polygon recompute script."""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import shapely

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from recompute_bperp_bbounds import recompute_bperp_bbounds
from recompute_product_bounds import BOUNDING_POLYGON_PATH

# Default test product (downloaded from ASF if not present locally). Override
# with a local file via the DISP_S1_TEST_FILE environment variable.
_DEFAULT_URL = "https://datapool.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20250408T163512Z.nc"
BASELINE_PATH = "/corrections/perpendicular_baseline"
# Coarse subsample keeps the baseline computation fast in tests.
_SUBSAMPLE = 200


@pytest.fixture
def test_file():
    """Path to the test DISP-S1 file."""
    override = os.environ.get("DISP_S1_TEST_FILE")
    if override:
        path = Path(override)
        if not path.exists():
            pytest.skip(f"DISP_S1_TEST_FILE not found: {path}")
        return path

    file = Path() / Path(_DEFAULT_URL).name
    if not file.exists():
        import subprocess

        subprocess.run(["wget", _DEFAULT_URL], check=True)
    return file


@pytest.fixture(scope="module")
def recomputed_file(tmp_path_factory):
    """Run the combined recompute once and reuse across tests."""
    override = os.environ.get("DISP_S1_TEST_FILE")
    test_file = Path(override) if override else Path() / Path(_DEFAULT_URL).name
    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    tmp_dir = tmp_path_factory.mktemp("bperp_bbounds_test")
    output_file = tmp_dir / "test_output.nc"
    # update_metadata=False keeps the run deterministic.
    recompute_bperp_bbounds(
        test_file, output_file, subsample=_SUBSAMPLE, update_metadata=False
    )
    return output_file


def test_baseline_recomputed(test_file, recomputed_file):
    """The perpendicular baseline is rewritten at full resolution (and changed)."""
    with h5py.File(test_file) as f:
        original = f[BASELINE_PATH][:]
    with h5py.File(recomputed_file) as f:
        recomputed = f[BASELINE_PATH][:]

    assert recomputed.shape == original.shape
    assert np.isfinite(recomputed).any()
    # A fresh computation (here on a coarser grid) is not bit-identical to the
    # stored layer, confirming it was actually recomputed rather than copied.
    assert not np.array_equal(np.nan_to_num(original), np.nan_to_num(recomputed))


def test_bounding_polygon_is_rotated_rectangle(recomputed_file):
    """The bounding polygon is updated to a 5-point MULTIPOLYGON rectangle."""
    with h5py.File(recomputed_file) as f:
        geom = shapely.from_wkt(f[BOUNDING_POLYGON_PATH][()].decode("utf-8"))

    assert geom.geom_type == "MultiPolygon"
    assert len(geom.geoms[0].exterior.coords) == 5


def test_displacement_preserved(test_file, recomputed_file):
    """The displacement layer is untouched by the combined recompute."""
    with h5py.File(test_file) as o, h5py.File(recomputed_file) as c:
        np.testing.assert_array_equal(o["/displacement"][:], c["/displacement"][:])


def test_both_layers_updated_in_one_pass(test_file, recomputed_file):
    """A single output carries both the new baseline and the new polygon."""
    with h5py.File(test_file) as o, h5py.File(recomputed_file) as c:
        baseline_changed = not np.array_equal(
            np.nan_to_num(o[BASELINE_PATH][:]), np.nan_to_num(c[BASELINE_PATH][:])
        )
        polygon_changed = o[BOUNDING_POLYGON_PATH][()] != c[BOUNDING_POLYGON_PATH][()]
    assert baseline_changed and polygon_changed


def test_metadata_update(test_file, tmp_path):
    """processing_start_datetime is bumped once when update_metadata=True."""
    output_file = tmp_path / "meta.nc"
    with h5py.File(test_file) as f:
        original_dt = f["/identification/processing_start_datetime"][()].decode("utf-8")

    recompute_bperp_bbounds(
        test_file, output_file, subsample=_SUBSAMPLE, update_metadata=True
    )

    with h5py.File(output_file) as f:
        new_dt = f["/identification/processing_start_datetime"][()].decode("utf-8")
    assert new_dt != original_dt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
