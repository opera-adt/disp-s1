"""Module for extracting relevant metadata from the CSLC products."""
from datetime import datetime
from typing import Iterable

import h5py
from dolphin._types import Filename


# Get the full acquisition times
def get_zero_doppler_time(
    filenames: Iterable[Filename], type_: str = "start"
) -> list[datetime]:
    """Get the full acquisition time from the CSLC product.

    Uses `/identification/zero_doppler_{type_}_time` from the CSLC product.

    Parameters
    ----------
    filenames : Filename
        Path to the CSLC product.
    type_ : str, optional
        Either "start" or "stop", by default "start".

    Returns
    -------
    str
        Full acquisition time.
    """
    datetimes = []
    for filename in filenames:
        with h5py.File(filename, "r") as hf:
            # Example:
            # "2022-11-19 14:07:51.997436"
            date_str = hf[f"/identification/zero_doppler_{type_}_time"][()]

        datetimes.append(
            datetime.strptime(date_str.decode("utf-8"), "%Y-%m-%d %H:%M:%S.%f")
        )
    return datetimes
