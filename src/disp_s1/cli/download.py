import functools
from collections.abc import Sequence
from pathlib import Path

import click
import opera_utils
from dolphin import Bbox
from opera_utils import get_burst_ids_for_frame

from disp_s1._dem import S3_DEM_BUCKET, S3_LONLAT_VRT_KEY
from disp_s1._dem import stage_dem as stage_dem
from disp_s1._water import create_water_mask
from disp_s1.ionosphere import (
    DEFAULT_DOWNLOAD_ENDPOINT,
    DownloadConfig,
    IonosphereType,
    download_ionosphere_files,
)

click.option = functools.partial(click.option, show_default=True)


@click.group(name="download")
def download_group():
    """Sub-commands for downloading prerequisite data."""
    from dolphin._log import setup_logging

    setup_logging(logger_name="disp_s1")


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
    help=(
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
    import opera_utils.geometry

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


@download_group.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default="dem.vrt",
    help="Output DEM filepath (VRT format)",
)
@click.option("--frame-id", type=int, help="Sentinel-1 OPERA Frame ID")
@click.option(
    "-b", "--bbox", type=float, nargs=4, help="Bounding box (WSEN, decimal degrees)"
)
@click.option(
    "-m", "--margin", type=int, default=5, help="Margin for bounding box in km"
)
@click.option(
    "-s", "--s3-bucket", default=S3_DEM_BUCKET, help="S3 bucket containing global DEM"
)
@click.option(
    "-k", "--s3-key", default=S3_LONLAT_VRT_KEY, help="S3 key path within bucket"
)
@click.option("--debug/--no-debug", default=False, help="Enable debug logging")
def dem(
    output: Path,
    frame_id: int | None,
    bbox: Bbox | None,
    margin: int,
    s3_bucket: str,
    s3_key: str,
    debug: bool,
) -> None:
    """Stage a DEM for local processing."""
    if frame_id is None and bbox is None:
        raise ValueError("Must provide --frame-id or --bbox")
    if frame_id is not None:
        if bbox is not None:
            raise ValueError("Only may provide --frame-id or --bbox")
        # bbox = opera_utils.get_frame_bbox(frame_id=frame_id)
        utm_epsg, utm_bounds = opera_utils.get_frame_bbox(frame_id=frame_id)
        bbox = opera_utils.reproject_bounds(
            utm_bounds, src_epsg=utm_epsg, dst_epsg=4326
        )

    return stage_dem(
        output=output,
        bbox=bbox,
        margin=margin,
        s3_bucket=s3_bucket,
        s3_key=s3_key,
        debug=debug,
    )

@download_group.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default="water_binary_mask.tif",
    help="Output water mask filepath (TIF format)",
)
@click.option("--frame-id", type=int, help="Sentinel-1 OPERA Frame ID")
@click.option(
    "-b", "--bbox", type=float, nargs=4, help="Bounding box (WSEN, decimal degrees)"
)
@click.option(
    "-m", "--margin", type=int, default=5, help="Margin for bounding box in km"
)
@click.option(
    "--land-buffer",
    type=int,
    default=0,
    help="Buffer in km for land water regions (reduces masking).",
    show_default=True,
)
@click.option(
    "--ocean-buffer",
    type=int,
    default=0,
    help="Buffer in km for ocean water regions (reduces masking).",
    show_default=True,
)
@click.option(
    "--aws-profile",
    type=str,
    default="saml-pub",
    help="AWS profile",
    show_default=True,
)
@click.option(
    "--aws-region",
    type=str,
    default="us-west-2",
    help="AWS region",
    show_default=True,
)
@click.option("--debug/--no-debug", default=False, help="Enable debug logging")
def water(
    output: Path,
    frame_id: int | None,
    bbox: Bbox | None,
    margin: int,
    land_buffer: int,
    ocean_buffer: int,
    aws_profile: str,
    aws_region: str,
    debug:bool,
) -> None:
    """Stage a water mask for local processing."""
    if frame_id is None and bbox is None:
        raise ValueError("Must provide --frame-id or --bbox")
    if frame_id is not None:
        if bbox is not None:
            raise ValueError("Only may provide --frame-id or --bbox")
        # bbox = opera_utils.get_frame_bbox(frame_id=frame_id)
        utm_epsg, utm_bounds = opera_utils.get_frame_bbox(frame_id=frame_id)
        bbox = opera_utils.reproject_bounds(
            utm_bounds, src_epsg=utm_epsg, dst_epsg=4326
        )

    return create_water_mask(
        output=output,
        bbox=bbox,
        margin=margin,
        land_buffer=land_buffer,
        ocean_buffer=ocean_buffer,
        aws_profile=aws_profile,
        aws_region=aws_region,
        debug=debug
    )