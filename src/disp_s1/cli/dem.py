from pathlib import Path

import click
from dolphin import Bbox

from disp_s1._dem import S3_DEM_BUCKET
from disp_s1._dem import stage_dem as run_stage_dem


@click.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default="dem.vrt",
    help="Output DEM filepath (VRT format)",
)
@click.option(
    "-b", "--bbox", type=float, nargs=4, help="Bounding box (WSEN, decimal degrees)"
)
@click.option(
    "-m", "--margin", type=int, default=5, help="Margin for bounding box in km"
)
@click.option(
    "-s", "--s3-bucket", default=S3_DEM_BUCKET, help="S3 bucket containing global DEM"
)
@click.option("-k", "--s3-key", default="", help="S3 key path within bucket")
@click.option("--debug/--no-debug", default=False, help="Enable debug logging")
def stage_dem(
    output: Path,
    bbox: Bbox | None,
    margin: int,
    s3_bucket: str,
    s3_key: str,
    debug: bool,
) -> None:
    """Stage a DEM for local processing."""
    return run_stage_dem(
        output=output,
        bbox=bbox,
        margin=margin,
        s3_bucket=s3_bucket,
        s3_key=s3_key,
        debug=debug,
    )
