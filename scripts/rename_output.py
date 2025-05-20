#!/usr/bin/env python3
"""Rename a DISP-S1 NetCDF file to the official OPERA DISP-S1 naming convention.

Uses the datasets under the "/identification" group:
  - frame_id
  - source_data_polarization
  - processing_start_datetime
  - reference_zero_doppler_start_time
  - secondary_zero_doppler_start_time

Example:
-------
    $ python scripts/rename_output.py outputs/2017*.nc --dry-run

"""

import datetime
from pathlib import Path
from typing import Optional

import click
import h5py
from opera_utils import get_burst_id, get_dates


def _read_disp_s1_metadata(nc_file: Path) -> dict:
    """Read the DISP-S1 metadata from the NetCDF file.

    Parameters
    ----------
    nc_file : Path
        Path to the NetCDF file.

    Returns
    -------
    metadata : dict
        A dictionary with the following keys:
        ['frame_id', 'polarization', 'generation_dt',
         'reference_dt', 'secondary_dt']

    """
    with h5py.File(nc_file, mode="r") as ds:
        frame_id = ds["/identification/frame_id"][()]
        polarization = ds["/identification/source_data_polarization"][()].decode(
            "utf-8"
        )
        generation_str = ds["/identification/processing_start_datetime"][()].decode(
            "utf-8"
        )
        reference_str = ds["/identification/reference_zero_doppler_start_time"][
            ()
        ].decode("utf-8")
        secondary_str = ds["/identification/secondary_zero_doppler_start_time"][
            ()
        ].decode("utf-8")

    generation_dt = datetime.datetime.fromisoformat(generation_str)
    reference_dt = datetime.datetime.fromisoformat(reference_str)
    secondary_dt = datetime.datetime.fromisoformat(secondary_str)

    return {
        "frame_id": int(frame_id),
        "polarization": polarization,
        "generation_dt": generation_dt,
        "reference_dt": reference_dt,
        "secondary_dt": secondary_dt,
    }


def _format_dt(dt_obj: datetime.datetime) -> str:
    """Format a datetime object as YYYYMMDDTHHMMSSZ (UTC).

    Parameters
    ----------
    dt_obj : datetime.datetime
        Input datetime

    Returns
    -------
    str
        Datetime in format 'YYYYMMDDTHHMMSSZ'

    """
    # Assume UTC always
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc)
    return dt_obj.astimezone(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_filename(
    frame_id: int,
    polarization: str,
    reference_dt: datetime.datetime,
    secondary_dt: datetime.datetime,
    generation_dt: datetime.datetime,
    project: str = "OPERA",
    level: str = "L3",
    name: str = "DISP-S1",
    mode: str = "IW",
    version: str = "0.10",
) -> str:
    """Build the filename similar to DISP-S1 production naming convention.

    Parameters
    ----------
    frame_id : int
        Unique frame identification number.
    polarization : str
        The polarization (e.g., 'VV', 'VH', 'HH', 'HV', etc.).
    reference_dt : datetime.datetime
        Reference zero doppler start time.
    secondary_dt : datetime.datetime
        Secondary zero doppler start time.
    generation_dt : datetime.datetime
        Product generation date/time (often the processing start datetime).
    project : str, optional
        Project name, default 'OPERA'.
    level : str, optional
        Data product level, default 'L3'.
    name : str, optional
        PGE (product) name, default 'DISP-S1'.
    mode : str, optional
        The S1 acquisition mode, default 'IW'.
    version : str, optional
        Product version, e.g. '0.10'.

    Returns
    -------
    core_filename : str
        The constructed output filename (without file extension).

    """
    # Format the frame ID as a 5-digit string: e.g. 11116 -> F11116
    frame_str = f"F{frame_id:05d}"

    # Format the product version with 'v' prefix, e.g. 'v0.10'
    product_version = f"v{version}"

    ref_str = _format_dt(reference_dt)
    sec_str = _format_dt(secondary_dt)
    gen_str = _format_dt(generation_dt)

    core_filename = (
        f"{project}_{level}_{name}_{mode}_{frame_str}_{polarization}_"
        f"{ref_str}_{sec_str}_{product_version}_{gen_str}"
    )

    return core_filename


def rename_disp_s1_file(
    input_file: Path,
    output_dir: Optional[Path] = None,
    version: str = "0.10",
    dry_run: bool = True,
) -> Path:
    """Rename a DISP-S1 NetCDF file to the OPERA naming convention.

    Parameters
    ----------
    input_file : Path
        The existing NetCDF file to rename.
    output_dir : Path, optional
        Directory to place the renamed file. If not provided,
        renaming occurs in the same directory as `input_file`.
    version : str, optional
        Product version number, default is '0.10'.
    dry_run : bool
        Only print the names, don't actually rename.
        Default is True.

    Returns
    -------
    new_file_path : Path
        The path to the newly renamed file.

    """
    if Path(input_file).stem.lower().startswith("compressed"):
        new_filename = _create_disp_s1_compressed_filename(input_file=input_file)
    else:
        new_filename = _create_disp_s1_filename(input_file=input_file, version=version)
    return _rename(input_file, new_filename, output_dir=output_dir, dry_run=dry_run)


def _create_disp_s1_filename(
    input_file: Path,
    version: str = "0.10",
) -> str:
    """Create the DISP-S1 NetCDF file to the OPERA naming convention."""
    meta = _read_disp_s1_metadata(input_file)

    frame_id = meta["frame_id"]
    polarization = meta["polarization"]
    reference_dt = meta["reference_dt"]
    secondary_dt = meta["secondary_dt"]
    generation_dt = meta["generation_dt"]

    core_name = create_filename(
        frame_id=frame_id,
        polarization=polarization,
        reference_dt=reference_dt,
        secondary_dt=secondary_dt,
        generation_dt=generation_dt,
        version=version,
    )

    return f"{core_name}.nc"


def _rename(
    input_file: Path,
    new_filename: str,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Path:
    # If no output directory was provided, use the same directory as the input file
    if output_dir is None:
        output_dir = input_file.parent

    new_file_path = output_dir.joinpath(new_filename)

    if dry_run:
        click.echo(f"DRY RUN: {input_file} to {new_file_path}")
    else:
        input_file.rename(new_file_path)
        click.echo(f"Renamed {input_file} to {new_file_path}")

    return new_file_path


def _create_disp_s1_compressed_filename(
    input_file: Path,
    prod_time: datetime.datetime | None = None,
) -> str:
    """Create the compressed SLC filename folloing OPERA naming convention."""
    if not Path(input_file).stem.lower().startswith("compressed"):
        raise ValueError(f"{input_file} is not a Compressed SLC")

    try:
        ref_datetime, start_datetime, stop_datetime = get_dates(input_file)[:3]
        burst_id = get_burst_id(input_file)
    except Exception as e:
        fmt = "<base-date>_<start-date>_<end-date>_<burst_id>"
        raise ValueError(f"{input_file} name does not match {fmt}") from e

    if prod_time is None:
        prod_time = datetime.datetime.now()
    prod_time_str = _format_dt(prod_time)

    datefmt = "%Y%m%d"
    burst_id_caps = burst_id.replace("_", "-").upper()
    return (
        f"OPERA_L2_COMPRESSED-CSLC-S1_{burst_id_caps}_"
        f"{ref_datetime.strftime(datefmt)}T000000Z_{start_datetime.strftime(datefmt)}T000000Z_"
        f"{stop_datetime.strftime(datefmt)}T000000Z_{prod_time_str}_"
        "VV_v1.0.h5"
    )


@click.command()
@click.argument("input_files", type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory for the renamed file. If None, uses same as input.",
)
@click.option(
    "--version", default="0.10", help="Product version number, default is '0.10'."
)
@click.option("--dry-run", is_flag=True)
def main(input_files, output_dir, version, dry_run):
    """Rename a DISP-S1 NetCDF/Compressed HDF5 file to an official OPERA name."""
    for input_file in input_files:
        rename_disp_s1_file(
            input_file=input_file,
            output_dir=output_dir,
            version=version,
            dry_run=dry_run,
        )


if __name__ == "__main__":
    main()
