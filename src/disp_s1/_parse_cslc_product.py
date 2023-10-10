"""Module for extracting relevant metadata from the CSLC products."""
from datetime import datetime
from typing import Iterable

import h5py
from dolphin._types import Filename


def get_dset_and_attrs(filename: Filename, dset_name: str):
    """Get the dataset and attributes from the CSLC product.

    Parameters
    ----------
    filename : Filename
        Path to the CSLC product.
    dset_name : str
        Name of the dataset.

    Returns
    -------
    dset : h5py.Dataset
        Dataset.
    attrs : dict
        Attributes.
    """
    with h5py.File(filename, "r") as hf:
        dset = hf[dset_name]
        attrs = dict(dset.attrs)
        return dset[()], attrs


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
