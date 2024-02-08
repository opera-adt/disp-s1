#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import logging
import re
from configparser import ConfigParser
from pathlib import Path
from typing import Sequence

import numpy as np
from opera_utils import get_zero_doppler_time, group_by_date
from opera_utils._types import Bbox
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from disp_s1.ionosphere import download_ionex_for_slcs
from disp_s1.utils import get_frame_bbox

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_cli_args():
    parser = argparse.ArgumentParser(
        description="download weather models and TEC files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--slc-files",
        nargs=argparse.ONE_OR_MORE,
        help="List the paths of corresponding SLC files.",
        required=True,
    )

    parser.add_argument(
        "--frame-id", type=int, help="frame id of the data", required=True
    )

    parser.add_argument(
        "--working-dir", default="./", help="working directory for outputs"
    )

    return parser


def check_exist_grib_file(grib_files: list[Path]) -> list[Path]:
    """Check for grib files that are already downloaded.

    Parameters
    ----------
    grib_files: list[Path]
        list of the path to grib files

    Returns
    -------
    list[Path]
        list of all the grib files that already exist
    """
    grib_files_exist = []
    for file in grib_files:
        if file.exists():
            grib_files_exist.append(file)
    return grib_files_exist


def download_grib_files(
    grib_files: list[Path], tropo_model: str = "ERA5", snwe: Bbox = None
) -> list[Path]:
    """Download weather re-analysis grib files using PyAPS.

    Parameters
    ----------
    grib_files : list[Path]
        list of path to the grib files
    tropo_model : str
        the tropospheric model
    snwe : Bbox
        bounding box of the data required in the format of (south north west east)

    Returns
    -------
    list of path
    """
    import pyaps3 as pa

    logger.info("downloading weather model data using PyAPS ...")

    parent_dir = Path(pa.__file__).parent

    # Get date list to download (skip already downloaded files)
    grib_files_exist = check_exist_grib_file(grib_files)
    grib_files2dload = sorted(set(grib_files) - set(grib_files_exist))
    date_list2dload = [str(re.findall(r"\d{8}", i.name)[0]) for i in grib_files2dload]
    logger.info("number of grib files to download: %d" % len(date_list2dload))

    # Download grib file using PyAPS
    if len(date_list2dload) > 0:
        hour = (
            re.findall(r"\d{8}[-_]\d{2}", grib_files2dload[0].name)[0]
            .replace("-", "_")
            .split("_")[1]
        )
        grib_dir = grib_files2dload[0].parent

        # Check for non-empty account info in PyAPS config file
        check_pyaps_account_config(tropo_model, parent_dir)

        # try 3 times to download, then use whatever downloaded to calculate delay
        i = 0
        while i < 3:
            i += 1
            try:
                if tropo_model in ["ERA5", "ERAINT"]:
                    pa.ECMWFdload(
                        date_list2dload,
                        hour,
                        grib_dir,
                        model=tropo_model,
                        snwe=snwe,
                        flist=grib_files2dload,
                    )

                elif tropo_model == "MERRA":
                    pa.MERRAdload(date_list2dload, hour, grib_dir)

                elif tropo_model == "NARR":
                    pa.NARRdload(date_list2dload, hour, grib_dir)
            except Exception:
                if i < 3:
                    logger.warning(f"The {i} attempt to download failed, retrying it.")
                else:
                    logger.error("Downloading failed for 3 times. Stopping.")

    # check potentially corrupted files
    grib_files = check_exist_grib_file(grib_files)
    return grib_files


def check_pyaps_account_config(tropo_model: str, parent_dir: Path):
    """Check for input in PyAPS config file.

    Parameters
    ----------
    tropo_model :str
        tropo model being used to calculate tropospheric delay
    parent_dir : Path
        Path to the PyAPS package where it is installed

    Raises
    ------
    ValueError
        If account info is not set in PyAPS config file.
    """
    # Convert MintPy tropo model name to data archive center name
    # NARR model included for completeness but no key required
    MODEL2ARCHIVE_NAME = {
        "ERA5": "CDS",
        "ERAI": "ECMWF",
        "MERRA": "MERRA",
        "NARR": "NARR",
    }
    SECTION_OPTS = {
        "CDS": ["key"],
        "ECMWF": ["email", "key"],
        "MERRA": ["user", "password"],
    }

    # Default values in cfg file
    default_values = [
        "the-email-address-used-as-login@ecmwf-website.org",
        "the-user-name-used-as-login@earthdata.nasa.gov",
        "the-password-used-as-login@earthdata.nasa.gov",
        "the-email-adress-used-as-login@ucar-website.org",
        "your-uid:your-api-key",
    ]

    # account file for pyaps3 < and >= 0.3.0
    cfg_file = parent_dir / "model.cfg"
    rc_file = Path("~").home() / ".cdsapirc"

    # for ERA5: ~/.cdsapirc
    if tropo_model == "ERA5" and rc_file.is_file():
        pass

    # check account info for the following models
    elif tropo_model in ["ERA5", "ERAI", "MERRA"]:
        section = MODEL2ARCHIVE_NAME[tropo_model]

        # Read model.cfg file
        cfg_file = parent_dir / "model.cfg"
        cfg = ConfigParser()
        cfg.read(cfg_file)

        # check all required option values
        for opt in SECTION_OPTS[section]:
            val = cfg.get(section, opt)
            if not val or val in default_values:
                msg = "PYAPS: No account info found "
                msg += f"for {tropo_model} in {section} section in file: {cfg_file}"
                raise ValueError(msg)

    return


def get_grib_file_names(
    slc_files: Sequence[Path], grib_dir: Path, frame_id: int
) -> tuple[list[Path], Bbox]:
    """Generate the grib file names for each SLC to download.

    Parameters
    ----------
    slc_files : Sequence[Path]
        list of paths to the slc files
    grib_dir : Path
        Path to the grib directory
    frame_id : int
        DISP-S1 frame id

    Returns
    -------
    tuple[list[Path], Bbox]
        The list of weather model files to be downloaded
        and the bounding box of the frame
    """
    slc_file_list = group_by_date(slc_files)
    first_date = next(iter(slc_file_list))
    acquisition_time = get_zero_doppler_time(slc_file_list[first_date][0])
    hour = closest_weather_model_hour(acquisition_time)
    date_list = [
        datetime.datetime.strftime(key[0], "%Y%m%d") for key in slc_file_list.keys()
    ]

    epsg, bounds = get_frame_bbox(frame_id)

    if epsg != 4326:
        bounds = transform_bounds(CRS.from_epsg(epsg), CRS.from_epsg(4326), *bounds)

    snwe = (bounds[1], bounds[3], bounds[0], bounds[2])
    area = snwe2str(snwe)

    # grib file list
    grib_files = []
    for d in date_list:
        grib_file = grib_dir / f"ERA5{area}_{d}_{hour}.grb"
        grib_files.append(grib_file)

    return grib_files, snwe


def _add_buffer(snwe: tuple[float, float, float, float]):
    s, n, w, e = snwe

    min_buffer = 1
    # lat/lon0/1 --> SNWE
    S = np.floor(min(s, n) - min_buffer).astype(int)
    N = np.ceil(max(s, n) + min_buffer).astype(int)
    W = np.floor(min(w, e) - min_buffer).astype(int)
    E = np.ceil(max(w, e) + min_buffer).astype(int)
    return S, N, W, E


def snwe2str(snwe: tuple[float, float, float, float]) -> str:
    """Get area extent in string."""
    S, N, W, E = _add_buffer(snwe)

    area = ""
    area += f"_S{abs(S)}" if S < 0 else f"_N{abs(S)}"
    area += f"_S{abs(N)}" if N < 0 else f"_N{abs(N)}"
    area += f"_W{abs(W)}" if W < 0 else f"_E{abs(W)}"
    area += f"_W{abs(E)}" if E < 0 else f"_E{abs(E)}"

    return area


def closest_weather_model_hour(sar_acquisition_time: datetime.datetime) -> str:
    """Find closest available time of weather product from SAR acquisition time.

    Parameters
    ----------
    sar_acquisition_time: datetime.datetime
        SAR data acquisition time

    Returns
    -------
    grib_hr: str
        time of closest available weather product in hour
    """
    # get hour/min of SAR acquisition time
    # sar_time = datetime.strptime(sar_acquisition_time, "%Y-%m-%d %H:%M:%S.%f").hour
    sar_time = sar_acquisition_time.hour

    # find closest time in available weather products
    grib_hr_list = np.arange(0, 24)
    grib_hr = int(min(grib_hr_list, key=lambda x: abs(x - sar_time)))

    # add zero padding
    return f"{grib_hr:02d}"


def main():
    """Download ERA5 weather model files and TEC maps for all the given SLCs."""
    parser = _get_cli_args()
    args = parser.parse_args()

    grib_dir = Path(args.working_dir) / "troposphere_files"
    grib_dir.mkdir(exist_ok=True)
    TEC_dir = Path(args.working_dir) / "ionosphere_files"
    TEC_dir.mkdir(exist_ok=True)

    grib_files, snwe = get_grib_file_names(args.slc_files, grib_dir, args.frame_id)

    download_grib_files(grib_files, snwe=snwe)

    download_ionex_for_slcs(args.slc_files, TEC_dir)

    return


if __name__ == "__main__":
    main()
