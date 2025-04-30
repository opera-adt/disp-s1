"""Download Ionosphere correction files from NASA CDDIS archive.

Code adapted from OPERA PCM:
https://github.com/nasa/opera-sds-pcm/blob/1e86f6980edc42e6f326439d9166b62d622a8984/tools/stage_ionosphere_file.py

This script downloads ionosphere correction files corresponding to the acquisition
dates of input Sentinel-1 SLC or CSLC files. It supports both legacy and new
CDDIS archive naming conventions and handles authentication through EarthData Login.
"""

from __future__ import annotations

import datetime
import logging
import netrc
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import requests

# Constants
DEFAULT_EDL_ENDPOINT = "urs.earthdata.nasa.gov"
DEFAULT_DOWNLOAD_ENDPOINT = "https://cddis.nasa.gov/archive/gnss/products/ionex"


# Type aliases
class IonosphereType(str, Enum):
    """Type of ionosphere file to download."""

    JPLG = "jplg"
    JPRG = "jprg"


# Configure logging
logger = logging.getLogger(__name__)


class IonosphereFileNotFoundError(Exception):
    """Raised when a requested ionosphere file cannot be found in the archive."""

    pass


class SessionWithHeaderRedirection(requests.Session):
    """Class to maintain headers after EarthData Login redirect.

    This code was adapted from the examples available here:
    https://urs.earthdata.nasa.gov/documentation/for_users/data_access/python
    """

    def __init__(self, username, password, auth_host=DEFAULT_EDL_ENDPOINT):
        """Initialize the SessionWithHeaderRedirection class."""
        super().__init__()
        self.auth = (username, password)
        self.auth_host = auth_host

    # Overrides from the library to keep headers when redirected to or from
    # the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        """Maintain authentication through redirects.

        Parameters
        ----------
        prepared_request : requests.PreparedRequest
            The new request to be sent
        response : requests.Response
            The response that triggered this rebuild

        """
        headers = prepared_request.headers
        url = prepared_request.url

        if "Authorization" in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            if (
                (original_parsed.hostname != redirect_parsed.hostname)
                and redirect_parsed.hostname != self.auth_host
                and original_parsed.hostname != self.auth_host
            ):
                del headers["Authorization"]


@dataclass
class ArchiveDate:
    """Container for archive date information."""

    year: str
    doy: str

    @classmethod
    def from_date_str(cls, date_str: str) -> "ArchiveDate":
        """Create ArchiveDate from YYYYMMDD formatted string.

        Parameters
        ----------
        date_str : str
            Date string in YYYYMMDD format

        Returns
        -------
        ArchiveDate
            Container with parsed year and day of year

        """
        dt = datetime.datetime.strptime(date_str, "%Y%m%d")
        time_tuple = dt.timetuple()
        return cls(year=str(time_tuple.tm_year), doy=f"{time_tuple.tm_yday:03d}")


def parse_safe_date(filename: Path | str) -> str:
    """Extract acquisition date from Sentinel-1 SAFE filename.

    Parameters
    ----------
    filename : str or Path
        Path to SAFE archive or its filename

    Returns
    -------
    str
        Acquisition date in YYYYMMDD format

    Raises
    ------
    ValueError
        If filename does not match expected SAFE naming convention

    """
    safe_name = Path(filename).stem

    pattern = (
        r"(?P<mission_id>S1A|S1B)_(?P<beam_mode>IW)_(?P<product_type>SLC)(?P<resolution>_)_"
        r"(?P<level>1)(?P<class>S)(?P<pol>SH|SV|DH|DV)_(?P<start_ts>\d{8}T\d{6})_"
        r"(?P<stop_ts>\d{8}T\d{6})_(?P<orbit_num>\d{6})_(?P<data_take_id>[0-9A-F]{6})_"
        r"(?P<product_id>[0-9A-F]{4})"
    )

    match = re.match(pattern, safe_name)
    if not match:
        raise ValueError(f"Invalid SAFE filename format: {filename}")

    return match.group("start_ts").split("T")[0]


def parse_cslc_date(filename: Path | str) -> str:
    """Extract acquisition date from OPERA CSLC filename.

    Parameters
    ----------
    filename : str or Path
        Path to CSLC archive or its filename

    Returns
    -------
    str
        Acquisition date in YYYYMMDD format

    Raises
    ------
    ValueError
        If filename does not match expected CSLC naming convention

    """
    cslc_name = Path(filename).stem

    pattern = (
        r"OPERA_L2_(COMPRESSED-)?CSLC-S1_"
        r"(?P<burst_id>[Tt]\d{3}[-_]\d{6}[-_][Ii][Ww]\d)[-_]"
        r"(?P<acquisition_ts>\d{8}(T\d{6}Z)?)_.*"
    )

    match = re.match(pattern, cslc_name)
    if not match:
        raise ValueError(f"Invalid CSLC filename format: {filename}")

    return match.group("acquisition_ts").split("T")[0]


def get_archive_name(
    ionosphere_type: IonosphereType, archive_date: ArchiveDate, legacy: bool = False
) -> str:
    """Generate ionosphere archive filename.

    Parameters
    ----------
    ionosphere_type : {'jplg', 'jprg'}
        Type of ionosphere file to download
    archive_date : ArchiveDate
        Container with year and day of year
    legacy : bool, optional
        If True, use legacy naming convention, by default False

    Returns
    -------
    str
        Generated archive filename

    """
    if legacy:
        return f"{ionosphere_type.value}{archive_date.doy}0.{archive_date.year[2:]}i.Z"

    product_type = "RAP" if ionosphere_type == "jprg" else "FIN"
    return f"JPL0OPS{product_type}_{archive_date.year}{archive_date.doy}0000_01D_02H_GIM.INX.gz"  # noqa: E501


def download_ionosphere_file(
    url: str, output_dir: Path, session: requests.Session
) -> Path:
    """Download and extract ionosphere file.

    Parameters
    ----------
    url : str
        URL of the ionosphere file to download
    output_dir : Path
        Directory to save the downloaded file
    session : requests.Session
        Authenticated session for downloading

    Returns
    -------
    Path
        Path to the extracted ionosphere file

    Raises
    ------
    requests.exceptions.HTTPError
        If download fails

    """
    response = session.get(url, stream=True)
    response.raise_for_status()

    archive_path = output_dir / Path(url).name
    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    # Extract the downloaded file
    result = subprocess.run(
        ["gunzip", "-c", str(archive_path)], check=True, capture_output=True
    )

    output_path = archive_path.with_suffix("")
    output_path.write_bytes(result.stdout)

    # Clean up compressed file
    archive_path.unlink()

    return output_path


@dataclass
class DownloadConfig:
    """Configuration for ionosphere file download."""

    input_files: list[Path]
    output_dir: Path
    ionosphere_type: IonosphereType = IonosphereType("jplg")
    username: str | None = None
    password: str | None = None
    download_endpoint: str = DEFAULT_DOWNLOAD_ENDPOINT


def download_ionosphere_files(config: DownloadConfig) -> list[Path]:
    """Download ionosphere files for multiple input files.

    Parameters
    ----------
    config : DownloadConfig
        Download configuration parameters

    Returns
    -------
    list[Path]
        Paths to downloaded ionosphere files

    Raises
    ------
    ValueError
        If authentication credentials are missing
    IonosphereFileNotFoundError
        If a required ionosphere file cannot be found

    """
    # Get credentials if not provided
    username = config.username
    password = config.password

    if not (username and password):
        try:
            parsed = netrc.netrc().authenticators(DEFAULT_EDL_ENDPOINT)
            assert parsed is not None
            username, _, password = parsed
        except (FileNotFoundError, TypeError):
            raise ValueError(
                "No authentication credentials provided and none found in .netrc"
            )

    session = SessionWithHeaderRedirection(username, password)
    downloaded_files = []

    for input_file in config.input_files:
        # Try parsing as SAFE first, then CSLC
        try:
            date_str = parse_safe_date(input_file)
        except ValueError:
            try:
                date_str = parse_cslc_date(input_file)
            except ValueError:
                raise ValueError(
                    f"File {input_file} does not match SAFE or CSLC naming convention"
                )

        archive_date = ArchiveDate.from_date_str(date_str)

        # Try both naming conventions
        found = False
        for legacy in (True, False):
            archive_name = get_archive_name(
                config.ionosphere_type, archive_date, legacy
            )
            url = f"{config.download_endpoint}/{archive_date.year}/{archive_date.doy}/{archive_name}"  # noqa: E501

            response = session.head(url)
            if response.status_code == 200:
                output_file = download_ionosphere_file(url, config.output_dir, session)
                downloaded_files.append(output_file)
                found = True
                break

        if not found:
            raise IonosphereFileNotFoundError(
                f"No ionosphere file found for date {date_str}"
            )

    return downloaded_files
