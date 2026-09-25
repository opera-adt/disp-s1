"""The unwrapped raster's nodata must come from its mask, not from its values.

`dolphin.timeseries` subtracts the reference pixel's value from the whole
unwrapped array, so that pixel is exactly 0.0 by construction -- it is the one
pixel whose value is known exactly. Inferring nodata as `arr == 0` therefore
discarded it from every product ever written, along with any other pixel that
legitimately unwrapped to zero.
"""

import numpy as np

from disp_s1.product import nodata_mask


def _raster():
    """A 3x3 unwrapped patch: one true nodata pixel, one exact-zero pixel."""
    data = np.array([[1.5, -2.0, 0.5],
                     [0.3, 0.0, -1.1],      # centre is the reference pixel
                     [9.9, 0.7, 2.2]])
    mask = np.zeros_like(data, dtype=bool)
    mask[0, 0] = True                        # the only pixel with no data
    return np.ma.MaskedArray(data, mask=mask)


def test_only_the_masked_pixel_is_nodata():
    got = nodata_mask(_raster())
    assert got[0, 0], "the masked pixel must be nodata"
    assert got.sum() == 1, "nothing else is nodata"


def test_the_reference_pixel_survives():
    """The regression: an exact zero is a value, not a gap."""
    arr = _raster()
    assert arr[1, 1] == 0.0
    assert not nodata_mask(arr)[1, 1]


def test_inferring_from_values_is_what_went_wrong():
    """Kept as the contrast: the old rule loses the reference pixel."""
    arr = _raster()
    old = np.ma.filled(arr, 0) == 0
    assert old[1, 1], "the old rule masked the reference pixel"
    assert not nodata_mask(arr)[1, 1]
    assert old.sum() == 2 and nodata_mask(arr).sum() == 1


def test_an_unmasked_array_has_no_nodata():
    arr = np.ma.MaskedArray(np.zeros((2, 2)), mask=False)
    assert nodata_mask(arr).shape == (2, 2)
    assert not nodata_mask(arr).any()
