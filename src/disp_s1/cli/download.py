from collections.abc import Sequence
from pathlib import Path

import click
import opera_utils.geometry
from opera_utils import get_burst_ids_for_frame

from disp_s1.ionosphere import (
    DEFAULT_DOWNLOAD_ENDPOINT,
    DownloadConfig,
    IonosphereType,
    download_ionosphere_files,
)


@click.group(name="download")
def download_group():
    """Sub-commands for downloading prerequisite data."""


@download_group.command()
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
def ionosphere(
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


@download_group.command()
@click.option("--frame-id", type=int, help="Sentinel-1 OPERA Frame ID")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to save downloaded static layers",
)
@click.option(
    "--max-jobs",
    "-j",
    type=int,
    default=4,
    help="Maximum number of concurrent downloads",
)
@click.option(
    "--burst-ids",
    "-b",
    type=str,
    multiple=True,
    help=(  # noqa: E501
        "Optional specific burst IDs to download. If not provided, gets all bursts for"
        " frame."
    ),
)
def static_layers(
    frame_id: int,
    output_dir: Path,
    max_jobs: int,
    burst_ids: Sequence[str] | None = None,
) -> None:
    """Download CSLC static layers for a given frame ID.

    If specific burst IDs are not provided, downloads static layers for all bursts
    in the frame. The output directory must exist.

    """
    # Use provided burst IDs or get all bursts for frame
    bursts_to_download = (
        list(burst_ids) if burst_ids else get_burst_ids_for_frame(frame_id)
    )

    click.echo(
        f"Downloading static layers for {len(bursts_to_download)} bursts:"
        f" {bursts_to_download}"
    )

    opera_utils.geometry.download_cslc_static_layers(
        burst_ids=bursts_to_download, output_dir=output_dir, max_jobs=max_jobs
    )

    click.echo("Download complete")
