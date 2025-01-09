from collections.abc import Sequence
from pathlib import Path

import click

from disp_s1.ionosphere import (
    DEFAULT_DOWNLOAD_ENDPOINT,
    DownloadConfig,
    IonosphereType,
    download_ionosphere_files,
)


@click.command()
@click.argument("input_files", type=str, nargs=-1, required=True)
@click.option(
    "--output-dir",
    "-o",
    type=Path,
    default=Path.cwd(),
    help="Directory to save downloaded files",
)
@click.option(
    "--type",
    "-t",
    "ionosphere_type",
    type=click.Choice([v.value for v in IonosphereType]),
    default=IonosphereType.JPLG.value,
    help="Type of ionosphere file to download",
)
@click.option("--username", "-u", help="EarthData Login username (if no ~/.netrc)")
@click.option("--password", "-p", help="EarthData Login password (if no ~/.netrc)")
@click.option(
    "--download-endpoint",
    default=DEFAULT_DOWNLOAD_ENDPOINT,
    help="CDDIS download endpoint",
)
def download_ionosphere(
    input_files: Sequence[Path],
    output_dir: Path,
    ionosphere_type: IonosphereType,
    username: str | None,
    password: str | None,
    download_endpoint: str,
):
    """Download ionosphere correction files for Sentinel-1 data.

    INPUT_FILES: One or more SAFE or CSLC files to download ionosphere data for.
    Multiple files can be specified as space-separated arguments.
    """
    config = DownloadConfig(
        input_files=list(input_files),
        output_dir=output_dir,
        ionosphere_type=IonosphereType(ionosphere_type),
        username=username,
        password=password,
        download_endpoint=download_endpoint,
    )
    downloaded_files = download_ionosphere_files(config)

    for file in downloaded_files:
        click.echo(f"Downloaded: {file}")
