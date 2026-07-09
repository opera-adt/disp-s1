"""Tests for recomputing the bounding polygon from the displacement layer."""

import os
import sys
from pathlib import Path

import h5py
import pytest
import shapely

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from recompute_product_bounds import (
    BOUNDING_POLYGON_PATH,
    recompute_product_bounds,
)

# Default test product (downloaded from ASF if not present locally). Override
# with a local file via the DISP_S1_TEST_FILE environment variable.
_DEFAULT_URL = "https://datapool.asf.alaska.edu/DISP/OPERA-S1/OPERA_L3_DISP-S1_IW_F11116_VV_20160705T140755Z_20160729T140756Z_v1.0_20250408T163512Z.nc"


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
    """Recompute the bounding polygon once and reuse across all tests."""
    override = os.environ.get("DISP_S1_TEST_FILE")
    test_file = Path(override) if override else Path() / Path(_DEFAULT_URL).name
    if not test_file.exists():
        pytest.skip(f"Test file not found: {test_file}")

    tmp_dir = tmp_path_factory.mktemp("bounds_test")
    output_file = tmp_dir / "test_output.nc"
    # Don't update metadata in tests to keep tests deterministic
    recompute_product_bounds(test_file, output_file, update_metadata=False)
    return output_file


def test_recomputed_polygon_is_valid(recomputed_file):
    """The recomputed bounding polygon is a valid lon/lat geometry."""
    with h5py.File(recomputed_file) as f:
        wkt = f[BOUNDING_POLYGON_PATH][()].decode("utf-8")

    geom = shapely.from_wkt(wkt)
    assert geom.is_valid
    assert not geom.is_empty
    assert geom.geom_type in ("Polygon", "MultiPolygon")

    # Coordinates should be in EPSG:4326 (lon/lat) range.
    minx, miny, maxx, maxy = geom.bounds
    assert -180.0 <= minx <= maxx <= 180.0
    assert -90.0 <= miny <= maxy <= 90.0


def test_minimum_rotated_rectangle_fix(recomputed_file):
    """The default fix returns a 4-corner (rotated-rectangle) polygon."""
    with h5py.File(recomputed_file) as f:
        wkt = f[BOUNDING_POLYGON_PATH][()].decode("utf-8")

    geom = shapely.from_wkt(wkt)
    polygon = geom.geoms[0] if geom.geom_type == "MultiPolygon" else geom
    # A minimum rotated rectangle has 5 exterior coords (4 unique + closing).
    assert len(polygon.exterior.coords) == 5


def test_polygon_covers_all_valid_pixels(test_file, recomputed_file):
    """Every valid displacement pixel must fall inside the recomputed polygon.

    This guards the orientation and simplification regressions: a y-flipped read
    or pre-simplified hull leaves a few percent of the data outside the box.
    """
    import numpy as np
    from pyproj import CRS, Transformer

    with h5py.File(recomputed_file) as f:
        polygon = shapely.from_wkt(f[BOUNDING_POLYGON_PATH][()].decode("utf-8"))
    with h5py.File(test_file) as f:
        disp = f["/displacement"][:]
        x = f["/x"][:]
        y = f["/y"][:]
        crs = CRS.from_wkt(f["/spatial_ref"].attrs["crs_wkt"])

    rows, cols = np.where(np.isfinite(disp) & (disp != 0))
    # Subsample for speed; boundary coverage is what matters.
    sub = slice(None, None, 25)
    transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(x[cols[sub]], y[rows[sub]])
    inside = shapely.contains(polygon, shapely.points(lon, lat))
    assert inside.all()


def test_recompute_preserves_data(test_file, recomputed_file):
    """Other datasets are preserved when recomputing the bounding polygon."""
    import numpy as np

    with h5py.File(test_file) as f_orig, h5py.File(recomputed_file) as f_new:
        assert f_orig["/x"][:].shape == f_new["/x"][:].shape
        assert f_orig["/y"][:].shape == f_new["/y"][:].shape
        assert "/displacement" in f_new
        np.testing.assert_array_equal(
            f_orig["/displacement"][:], f_new["/displacement"][:]
        )


def test_metadata_update(test_file, tmp_path):
    """Test that metadata timestamps are updated correctly."""
    output_file = tmp_path / "test_metadata.nc"

    with h5py.File(test_file) as f:
        original_datetime = f["/identification/processing_start_datetime"][()].decode(
            "utf-8"
        )
        original_version = f["/identification/product_version"][()].decode("utf-8")

    recompute_product_bounds(
        test_file, output_file, update_metadata=True, update_version=False
    )

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

    with h5py.File(test_file) as f:
        original_version = f["/identification/product_version"][()].decode("utf-8")

    recompute_product_bounds(
        test_file,
        output_file,
        update_metadata=True,
        update_version=True,
        new_version="1.1",
    )

    with h5py.File(output_file) as f:
        new_version = f["/identification/product_version"][()].decode("utf-8")

    assert new_version == "1.1"
    assert new_version != original_version


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
