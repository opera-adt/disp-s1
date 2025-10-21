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

    product_type = "RAP" if ionosphere_type == IonosphereType.JPRG else "FIN"
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
    subprocess.CalledProcessError
        If `gunzip` command fails

    """
    logger.debug(f"Attempting to download from: {url}")
    response = session.get(url, stream=True)
    response.raise_for_status()

    archive_path = output_dir / Path(url).name
    logger.debug(f"Saving compressed file to: {archive_path}")
    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    # Extract the downloaded file
    logger.debug(f"Uncompressing {archive_path}")
    try:
        result = subprocess.run(
            ["gunzip", "-c", str(archive_path)], check=True, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to uncompress {archive_path}: {e}")
        # Clean up the failed download
        if archive_path.exists():
            archive_path.unlink()
        raise

    output_path = archive_path.with_suffix("")
    output_path.write_bytes(result.stdout)
    logger.info(f"Successfully created uncompressed file: {output_path}")

    # Clean up compressed file
    logger.debug(f"Removing compressed file: {archive_path}")
    archive_path.unlink()

    return output_path


def download_ionosphere_files(
    input_files: list[Path],
    output_dir: Path,
    ionosphere_type: IonosphereType = IonosphereType("jplg"),
    username: str | None = None,
    password: str | None = None,
    download_endpoint: str = DEFAULT_DOWNLOAD_ENDPOINT,
) -> list[Path]:
    """Download ionosphere files for multiple input files.

    This function processes a list of Sentinel-1 SAFE or OPERA CSLC files,
    determines the unique acquisition dates, and downloads the corresponding
    ionosphere correction files from the NASA CDDIS archive.

    It avoids re-downloading files that already exist in the output directory
    and processes each unique date only once.

    Parameters
    ----------
    input_files : list[Path]
        A list of paths to input SAFE or CSLC files.
    output_dir : Path
        Directory to save downloaded and uncompressed ionosphere files.
    ionosphere_type : IonosphereType, optional
        Type of ionosphere file to download (e.g., 'jplg', 'jprg').
        Defaults to IonosphereType("jplg").
    username : str, optional
        EarthData Login username. If not provided, it will be sought in
        the user's .netrc file.
    password : str, optional
        EarthData Login password. If not provided, it will be sought in
        the user's .netrc file.
    download_endpoint : str, optional
        The base URL for the CDDIS ionosphere archive.
        Defaults to DEFAULT_DOWNLOAD_ENDPOINT.

    Returns
    -------
    list[Path]
        A list of paths to the downloaded (or existing) uncompressed
        ionosphere files, corresponding one-to-one with the `input_files`.

    Raises
    ------
    ValueError
        If authentication credentials are missing (and not in .netrc) or
        if an input file does not match SAFE or CSLC naming conventions.
    IonosphereFileNotFoundError
        If a required ionosphere file for a specific date cannot be
        found in the archive.

    """
    # Get credentials if not provided
    username_val = username
    password_val = password

    if not (username_val and password_val):
        try:
            parsed = netrc.netrc().authenticators(DEFAULT_EDL_ENDPOINT)
            if parsed is None:
                raise TypeError
            username_val, _, password_val = parsed
        except (FileNotFoundError, TypeError):
            raise ValueError(
                "No authentication credentials provided and none found in .netrc"
            )

    session = SessionWithHeaderRedirection(username_val, password_val)

    # 1. Find all unique acquisition dates from the input files
    dates_to_process = set()
    # Map ArchiveDate back to a YYYYMMDD string for errors/logging
    date_str_map = {}
    for input_file in input_files:
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
        dates_to_process.add(archive_date)
        if archive_date not in date_str_map:
            date_str_map[archive_date] = date_str

    # Loop over unique dates and download files if they don't exist
    # Map the processed date to its final local path
    date_to_path_map: dict[ArchiveDate, Path] = {}
    for archive_date in dates_to_process:
        found_or_exists = False
        date_str = date_str_map[archive_date]

        # Try both naming conventions (legacy and new)
        for legacy in (True, False):
            archive_name = get_archive_name(ionosphere_type, archive_date, legacy)

            # This is the path to the final uncompressed file
            expected_output_path = output_dir / Path(archive_name).with_suffix("")

            # Check if the uncompressed file already exists
            if expected_output_path.exists():
                logger.info(
                    f"Using existing ionosphere file for date {date_str}: "
                    f"{expected_output_path}"
                )
                date_to_path_map[archive_date] = expected_output_path
                found_or_exists = True
                break  # Found it, no need to check other format or download

            # If not, check the remote archive
            url = f"{download_endpoint}/{archive_date.year}/{archive_date.doy}/{archive_name}"  # noqa: E501

            try:
                response = session.head(url)
                response.raise_for_status()

                # HEAD was successful
                logger.info(
                    f"Downloading ionosphere file for date {date_str} from {url}"
                )
                output_file = download_ionosphere_file(url, output_dir, session)
                date_to_path_map[archive_date] = output_file
                found_or_exists = True
                break  # Downloaded, stop checking formats

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.debug(f"File not found at {url}, trying next format.")
                    continue  # Try the next format (legacy=False)
                else:
                    logger.error(f"HTTP error checking {url}: {e}")
                    raise  # Re-raise other HTTP errors

        if not found_or_exists:
            raise IonosphereFileNotFoundError(
                f"No ionosphere file found for date {date_str} (Year: "
                f"{archive_date.year}, DoY: {archive_date.doy})"
            )

    # Map the original input files to their corresponding (now local) iono paths
    output_paths = []
    for input_file in input_files:
        # We know these parse because they did in the first loop
        try:
            date_str = parse_safe_date(input_file)
        except ValueError:
            date_str = parse_cslc_date(input_file)

        archive_date = ArchiveDate.from_date_str(date_str)
        output_paths.append(date_to_path_map[archive_date])

    return output_paths
