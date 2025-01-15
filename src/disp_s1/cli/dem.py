from pathlib import Path

import click
import opera_utils
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
@click.option("-k", "--s3-key", default="", help="S3 key path within bucket")
@click.option("--debug/--no-debug", default=False, help="Enable debug logging")
def stage_dem(
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

    return run_stage_dem(
        output=output,
        bbox=bbox,
        margin=margin,
        s3_bucket=s3_bucket,
        s3_key=s3_key,
        debug=debug,
    )
