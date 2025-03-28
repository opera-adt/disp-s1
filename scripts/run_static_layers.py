#!/usr/bin/env python3
import glob
import logging
from pathlib import Path

import opera_utils
import tyro
from dolphin._log import setup_logging
from opera_utils import get_burst_ids_for_frame

from disp_s1._dem import stage_dem
from disp_s1.main_static_layers import run_static_layers
from disp_s1.pge_runconfig import StaticLayersRunConfig

logger = logging.getLogger("disp_s1")


def download_static_layers(
    frame_id: int,
    output_dir: Path = Path("static_layers"),
) -> None:
    """Download CSLC static layers for a given frame ID.

    Parameters
    ----------
    frame_id : int
        Sentinel-1 OPERA Frame ID
    output_dir : Path, optional
        Directory to save downloaded static layers, by default Path("static_layers")

    """
    import opera_utils.geometry

    bursts_to_download = get_burst_ids_for_frame(frame_id)

    logging.info(
        f"Downloading static layers for {len(bursts_to_download)} bursts:"
        f" {bursts_to_download}"
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    opera_utils.geometry.download_cslc_static_layers(
        burst_ids=bursts_to_download, output_dir=output_dir
    )

    logging.info("Download complete")


def download_dem(frame_id: int, output: Path) -> None:
    """Stage a DEM for local processing.

    Parameters
    ----------
    frame_id : int
        Sentinel-1 OPERA Frame ID
    output : Path
        Output path for the DEM (VRT format)

    """
    # Get frame bbox
    utm_epsg, utm_bounds = opera_utils.get_frame_bbox(frame_id=frame_id)
    bbox = opera_utils.reproject_bounds(utm_bounds, src_epsg=utm_epsg, dst_epsg=4326)

    output.parent.mkdir(parents=True, exist_ok=True)
    return stage_dem(
        output=output,
        bbox=bbox,
    )


def download_rtc_static_layers(frame_id: int, output_dir: Path) -> None:
    """Download RTC static layers for a given frame ID.

    Parameters
    ----------
    frame_id : int
        Sentinel-1 OPERA Frame ID
    output_dir : Path
        Directory to save downloaded RTC static layers

    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    burst_ids = get_burst_ids_for_frame(frame_id)
    logging.info(f"Downloading RTC static layers for {len(burst_ids)} bursts")

    opera_utils.download.download_rtc_static_layers(
        burst_ids, output_dir=output_dir, layers=["mask"]
    )

    logging.info("RTC download complete")


def create_runconfig(
    frame_id: int,
    output_dir: Path,
    scratch_dir: Path,
    cslc_static_dir: Path,
    rtc_static_dir: Path,
    dem_file: Path,
) -> StaticLayersRunConfig:
    """Create a YAML runconfig file for static layers processing.

    Parameters
    ----------
    frame_id : int
        Sentinel-1 OPERA Frame ID
    output_dir : Path
        Directory to save output products
    scratch_dir : Path
        Directory for scratch files
    cslc_static_dir : Path
        Directory containing CSLC static layers
    rtc_static_dir : Path
        Directory containing RTC static layers
    dem_file : Path
        Path to the DEM file

    Returns
    -------
    StaticLayersRunConfig
        Object for Static Layers workflow configuration

    """
    cslc_static_files = sorted(glob.glob(str(cslc_static_dir / "*.h5")))
    rtc_static_files = sorted(glob.glob(str(rtc_static_dir / "*_mask.tif")))

    # populate only the required parameters for the static layers workflow
    runconfig_params = {
        "input_file_group": {"frame_id": frame_id},
        "dynamic_ancillary_file_group": {
            "static_layers_files": cslc_static_files,
            "dem_file": str(dem_file),
            "rtc_static_layers_files": rtc_static_files,
        },
        "static_ancillary_file_group": {"frame_to_burst_json": None},
        "primary_executable": {"product_type": "DISP_S1_STATIC"},
        "product_path_group": {
            "product_path": str(output_dir),
            "scratch_path": str(scratch_dir),
            "sas_output_path": str(output_dir),
            "product_version": "1.0",
        },
        "log_file": str(scratch_dir / "log_sas.log"),
    }
    rc = StaticLayersRunConfig(**runconfig_params)
    runconfig_path = Path("runconfig_static.yaml")
    rc.to_yaml(runconfig_path)
    logging.info(f"Created runconfig at {runconfig_path}")
    return rc


def main(frame_id: int, /, output_dir: Path | None = None) -> None:
    """Run the full static layers setup and processing workflow.

    Parameters
    ----------
    frame_id : int
        Sentinel-1 OPERA Frame ID
    output_dir : Path, optional
        Directory to save output products.
        By default, saves to Path(f"F{frame_id:05d}") for the given frame id.

    """
    setup_logging(logger_name="disp_s1")
    if output_dir is None:
        output_dir = Path(f"F{frame_id:05d}")

    scratch_dir = output_dir / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    # Create sub-paths for downloads
    cslc_static_dir = scratch_dir / "cslc_static"
    rtc_static_dir = scratch_dir / "rtc_static"

    # Step 1: Download DEM
    dem_path = scratch_dir / "dem.vrt"
    logging.info(f"Step 1: Downloading DEM for frame {frame_id}")
    download_dem(frame_id=frame_id, output=dem_path)

    # Step 2: Download CSLC static layers
    logging.info(f"Step 2: Downloading CSLC static layers for frame {frame_id}")
    download_static_layers(frame_id=frame_id, output_dir=cslc_static_dir)

    # Step 3: Download RTC static layers
    logging.info(f"Step 3: Downloading RTC static layers for frame {frame_id}")
    download_rtc_static_layers(frame_id=frame_id, output_dir=rtc_static_dir)

    # Step 4: Create runconfig and run
    logging.info("Step 4: Creating runconfig YAML")
    runconfig = create_runconfig(
        frame_id=frame_id,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        cslc_static_dir=cslc_static_dir,
        rtc_static_dir=rtc_static_dir,
        dem_file=dem_path,
    )

    logging.info("\nSetup complete, running static layers processing:")
    run_static_layers(runconfig)


if __name__ == "__main__":
    tyro.cli(main)
