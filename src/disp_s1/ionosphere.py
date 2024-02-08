import logging
import subprocess
from datetime import date
from pathlib import Path
from typing import Sequence

from opera_utils import group_by_date
from opera_utils._types import Filename

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

IONOSPHERE_TYPE_JPLG = "jplg"
IONOSPHERE_TYPE_JPRG = "jprg"


def download_ionex_for_slcs(
    input_files: Sequence[Filename],
    dest_dir: Filename,
    verbose: bool = False,
) -> list[Path]:
    """Download IONEX files for a list of SLC files.

    Parameters
    ----------
    input_files : Sequence[Path]
        List of SLC files.
    dest_dir : Path
        Directory to save the downloaded files.
    verbose : bool, optional
        Print messages, by default False.

    Returns
    -------
    list[Path]
        List of downloaded IONEX files.

    """
    date_to_file_list = group_by_date(input_files)
    logger.info(f"Found {len(date_to_file_list)} dates in the input files.")

    output_files = []
    for input_date_tuple, _file_list in date_to_file_list.items():
        input_date = input_date_tuple[0]
        logger.info("Downloading for %s", input_date)
        f = download_ionex_for_date(
            input_date, dest_dir=dest_dir.absolute(), verbose=verbose
        )
        output_files.append(f)

    return output_files


def download_ionex_for_date(
    input_date: date,
    dest_dir: Filename,
    solution_code: str = "jplg",
    verbose: bool = False,
) -> Path:
    """Download one IONEX file for a given date.

    Parameters
    ----------
    input_date: date
        The date to download.
    dest_dir : Path
        Directory to save the downloaded files.
    solution_code : str, optional
        Analysis center code, by default "jplg".
    verbose : bool, optional
        Print messages, by default False.

    Returns
    -------
    Path
        Path to the local IONEX text file.

    """
    source_urls = _generate_ionex_filename(input_date, solution_code=solution_code)

    for source_url in source_urls:
        dest_file = Path(dest_dir) / Path(source_url).name

        wget_cmd = ["wget", "--continue", "--auth-no-challenge", source_url]

        if not verbose:
            wget_cmd.append("--quiet")

        unzipped_file = Path(dest_dir) / Path(source_url).stem
        if unzipped_file.exists():
            logger.info(f"{Path(source_url).name} exists")
            continue

        logger.info('Running command: "%s"', " ".join(wget_cmd))
        subprocess.run(wget_cmd, cwd=dest_dir)

        if dest_file.exists():
            unzip_command = f"gzip --force --decompress {dest_file}".split()
            subprocess.run(unzip_command, cwd=dest_dir)

    logger.info('Running command: "%s"', " ".join(wget_cmd))
    subprocess.run(wget_cmd, cwd=dest_dir, check=False)
    return dest_file


def _generate_ionex_filename(
    input_date: date, solution_code: str = "jplg"
) -> list[str]:
    """Generate the IONEX file name.

    Parameters
    ----------
    input_date : datetime
        Date to download
    solution_code : str, optional
        GIM analysis center code, by default "jplg".
    date_format : str, optional
        Date format code, by default "%Y%m%d".

    Returns
    -------
    str
        Complete URL to the IONEX file.

    """
    day_of_year = f"{input_date.timetuple().tm_yday:03d}"
    year_short = str(input_date.year)[2:4]
    file_name = f"{solution_code.lower()}{day_of_year}0.{year_short}i.Z"

    url_directory = f"https://cddis.nasa.gov/archive/gnss/products/ionex/{input_date.year}/{day_of_year}"

    # old naming convention
    old_archive_name = f"{url_directory}/{file_name}"
    # new naming convention
    product_type = "RAP" if solution_code.lower() == IONOSPHERE_TYPE_JPRG else "FIN"
    filename = (
        f"JPL0OPS{product_type}_{input_date.year}{day_of_year}0000_01D_02H_GIM.INX.gz"
    )
    new_archive_name = f"{url_directory}/{filename}"

    return [old_archive_name, new_archive_name]
