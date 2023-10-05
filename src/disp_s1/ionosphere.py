import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence


def download_ionex(
    date_list: Sequence[datetime],
    dest_dir: Path,
    solution_code: str = "jpl",
    verbose: bool = False,
) -> list[Path]:
    """Download IONEX file.

    Parameters
    ----------
    date_list : list[datetime]
        The dates to download.
    dest_dir : Path
        Directory to save the downloaded files.
    solution_code : str, optional
        Analysis center code, by default "jpl".
    verbose : bool, optional
        Print messages, by default False.

    Returns
    -------
    list[Path]
        Paths to the local uncompressed IONEX files.
    """
    out_paths = []
    for date in date_list:
        source_url = _generate_ionex_filename(date, solution_code=solution_code)
        dest_file = dest_dir / Path(source_url).name
        out_paths.append(dest_file)

        wget_cmd = ["wget", "--continue", "--auth-no-challenge", source_url]

        if not verbose:
            wget_cmd.append("--quiet")

        print(wget_cmd)
        subprocess.run(wget_cmd, cwd=dest_dir)
    return out_paths


def _generate_ionex_filename(input_date: datetime, solution_code: str = "jpl") -> str:
    """Generate the IONEX file name.

    Parameters
    ----------
    input_date : datetime
        Date to download
    solution_code : str, optional
        GIM analysis center code, by default "jpl".
    date_format : str, optional
        Date format code, by default "%Y%m%d".

    Returns
    -------
    str
        Complete URL to the IONEX file.
    """
    day_of_year = f"{input_date.timetuple().tm_yday:03d}"
    year_short = str(input_date.year)[2:4]
    file_name = f"{solution_code.lower()}g{day_of_year}0.{year_short}i.Z"

    url_directory = "https://cddis.nasa.gov/archive/gnss/products/ionex"
    return f"{url_directory}/{input_date.year}/{day_of_year}/{file_name}"
