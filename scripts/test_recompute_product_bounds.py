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
    compute_bounding_polygon,
    recompute_product_bounds,
    repack_h5_file,
    write_h5_string,
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


def test_write_h5_string_grows_dataset_without_truncating(tmp_path):
    """write_h5_string must not truncate when the new value is longer.

    Regression test for a bug where the forward-produced bounding_polygon
    dataset is a fixed-length HDF5 string sized to the *original* WKT, and a
    longer recomputed antimeridian-split MULTIPOLYGON got silently cut short
    by a plain in-place `dataset[()] = ...` assignment.
    """
    import numpy as np

    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(
            BOUNDING_POLYGON_PATH,
            data=np.bytes_("POLYGON ((1 1, 2 2, 3 3, 1 1))"),
        )
        f[BOUNDING_POLYGON_PATH].attrs["units"] = "degrees"

    long_wkt = "MULTIPOLYGON (((1 1, 2 2, 3 3, 4 4, 1 1)), ((5 5, 6 6, 7 7, 8 8, 5 5)))"
    assert len(long_wkt) > len("POLYGON ((1 1, 2 2, 3 3, 1 1))")

    with h5py.File(path, "a") as f:
        write_h5_string(f, BOUNDING_POLYGON_PATH, long_wkt)

    with h5py.File(path, "r") as f:
        dset = f[BOUNDING_POLYGON_PATH]
        readback = dset[()].decode("utf-8")
        assert dset.attrs["units"] == "degrees"

    assert readback == long_wkt


def test_repack_h5_file_preserves_data_and_filters(tmp_path):
    """repack_h5_file must keep the dataset's data, chunking, and compression."""
    import numpy as np

    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "/data",
            data=np.zeros((64, 64), dtype="float32"),
            chunks=(32, 32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

    new_data = np.random.default_rng(0).random((64, 64)).astype("float32")
    with h5py.File(path, "a") as f:
        f["/data"][:] = new_data

    repack_h5_file(path)

    with h5py.File(path, "r") as f:
        dset = f["/data"]
        assert dset.compression == "gzip"
        assert dset.compression_opts == 4
        assert dset.shuffle is True
        assert dset.chunks == (32, 32)
        np.testing.assert_array_equal(dset[:], new_data)


def test_recomputed_polygon_matches_computed_wkt(test_file, recomputed_file):
    """The stored WKT must exactly match what compute_bounding_polygon produced.

    Guards against the fixed-length-string truncation bug: a loose
    `shapely.from_wkt(...).is_valid` check (as in the tests below) can still
    pass on a truncated-but-coincidentally-valid WKT string.
    """
    expected_wkt = compute_bounding_polygon(test_file)
    with h5py.File(recomputed_file) as f:
        stored_wkt = f[BOUNDING_POLYGON_PATH][()].decode("utf-8")

    assert stored_wkt == expected_wkt


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
