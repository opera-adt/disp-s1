import numpy as np

from disp_s1._ps import WeightScheme, _get_weighting


def test_get_weighting():
    expected_ns = [
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 7],
        [1, 2, 3, 4, 7, 12],
    ]
    for num_comp, expected_N in enumerate(expected_ns):
        N = _get_weighting(
            num_compressed_slc_files=num_comp,
            weight_scheme=WeightScheme.EXPONENTIAL,
            num_slc=15,
        )
        assert np.allclose(N, expected_N)
