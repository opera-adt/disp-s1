#!/usr/bin/env python
"""Recompute perpendicular baseline from saved orbit data in OPERA DISP-S1 products.

This script reads orbit information from the metadata group of an OPERA DISP-S1
NetCDF file and recomputes the perpendicular baseline correction layer.

This is useful for Region 4 products which are missing the perpendicular baseline layer.

Example:
-------
    $ python scripts/recompute_perpendicular_baseline.py input.nc -o output.nc

"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import isce3
import numpy as np
import tyro
from dolphin import baseline
from isce3.core import DateTime, Orbit, StateVector
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_orbit_from_netcdf(
    nc_file: Path,
    orbit_group: str,
) -> Orbit:
    """Load orbit information from netcdf and create isce3.core.Orbit.

    Parameters
    ----------
    nc_file : Path
        Path to the OPERA DISP-S1 NetCDF file.
    orbit_group : str
        Name of orbit group, e.g. "/metadata/reference_orbit".

    Returns
    -------
    Orbit
        isce3.core.Orbit object.

    """
    with h5py.File(nc_file) as f:
        orbit = f[orbit_group]
        times = orbit["time"][:]
        pos_x = orbit["position_x"][:]
        pos_y = orbit["position_y"][:]
        pos_z = orbit["position_z"][:]
        vel_x = orbit["velocity_x"][:]
        vel_y = orbit["velocity_y"][:]
        vel_z = orbit["velocity_z"][:]
        ref_epoch_str = orbit["reference_epoch"][()].decode("utf-8")

    reference_epoch = datetime.fromisoformat(ref_epoch_str)

    positions = np.stack([pos_x, pos_y, pos_z]).T
    velocities = np.stack([vel_x, vel_y, vel_z]).T

    # Create isce3 StateVectors
    orbit_svs = []
    for t, pos, vel in zip(times, positions, velocities):
        orbit_svs.append(
            StateVector(
                DateTime(reference_epoch + timedelta(seconds=float(t))),
                pos,
                vel,
            )
        )

    return Orbit(orbit_svs)


def get_grid_coords(
    x: np.ndarray, y: np.ndarray, epsg: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert x/y grid to lon/lat.

    Parameters
    ----------
    x : np.ndarray
        1D array of x coordinates.
    y : np.ndarray
        1D array of y coordinates.
    epsg : int
        EPSG code of the projection.

    Returns
    -------
    lon : np.ndarray
        2D array of longitude coordinates.
    lat : np.ndarray
        2D array of latitude coordinates.

    """
    X, Y = np.meshgrid(x, y)
    xx = X.flatten()
    yy = Y.flatten()
    crs = CRS.from_epsg(epsg)
    utm_to_lonlat = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = utm_to_lonlat.transform(xx=xx, yy=yy, radians=False)
    lon = lon.reshape(X.shape)
    lat = lat.reshape(Y.shape)
    return lon, lat


def compute_baselines_from_orbits(
    orbit_ref: Orbit,
    orbit_sec: Orbit,
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    wavelength: float,
    height: float = 0.0,
    threshold: float = 1e-08,
    maxiter: int = 50,
    delta_range: float = 10.0,
) -> np.ndarray:
    """Compute the perpendicular baseline at a subsampled grid for two orbits.

    Parameters
    ----------
    orbit_ref : Orbit
        Reference orbit.
    orbit_sec : Orbit
        Secondary orbit.
    x : np.ndarray
        1D array of x coordinates.
    y : np.ndarray
        1D array of y coordinates.
    epsg : int
        EPSG code of the projection.
    wavelength : float
        Radar wavelength in meters.
    height : float
        Target height to use for baseline computation.
        Default = 0.0
    threshold : float
        isce3 geo2rdr: azimuth time convergence threshold in meters
        Default = 1e-8
    maxiter : int
        isce3 geo2rdr: Maximum number of Newton-Raphson iterations
        Default = 50
    delta_range : float
        isce3 geo2rdr: Step size used for computing derivative of doppler
        Default = 10.0

    Returns
    -------
    baselines : np.ndarray
        2D array of perpendicular baselines

    """
    lon_grid, lat_grid = get_grid_coords(x=x, y=y, epsg=epsg)
    lon_arr = lon_grid.ravel()
    lat_arr = lat_grid.ravel()

    ellipsoid = isce3.core.Ellipsoid()
    zero_doppler = isce3.core.LUT2d()
    side = isce3.core.LookSide.Right

    baselines = []
    for lon, lat in zip(lon_arr, lat_arr):
        llh_rad = np.deg2rad([lon, lat, height]).reshape((3, 1))
        az_time_ref, range_ref = isce3.geometry.geo2rdr(
            llh_rad,
            ellipsoid,
            orbit_ref,
            zero_doppler,
            wavelength,
            side,
            threshold=threshold,
            maxiter=maxiter,
            delta_range=delta_range,
        )
        az_time_sec, range_sec = isce3.geometry.geo2rdr(
            llh_rad,
            ellipsoid,
            orbit_sec,
            zero_doppler,
            wavelength,
            side,
            threshold=threshold,
            maxiter=maxiter,
            delta_range=delta_range,
        )

        pos_ref, velocity = orbit_ref.interpolate(az_time_ref)
        pos_sec, _ = orbit_sec.interpolate(az_time_sec)
        b = baseline.compute(
            llh_rad, pos_ref, pos_sec, range_ref, range_sec, velocity, ellipsoid
        )

        baselines.append(b)

    return np.array(baselines).reshape(lon_grid.shape)


def interpolate_data(
    data: np.ndarray, shape: tuple[int, int], method: str = "linear"
) -> np.ndarray:
    """Interpolate data to a new shape.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    shape : tuple[int, int]
        Desired output shape.
    method : str
        Interpolation method, default "linear".

    Returns
    -------
    np.ndarray
        Interpolated data.

    """
    orig_coords = [np.linspace(0, 1, s) for s in data.shape]
    new_coords = [np.linspace(0, 1, s) for s in shape]
    interp = RegularGridInterpolator(orig_coords, data, method=method)
    mesh = np.meshgrid(*new_coords, indexing="xy")
    return interp(np.array(mesh).T.astype("float32"))


def update_metadata_timestamps(
    output_file: Path,
    processing_datetime: datetime | None = None,
    update_version: bool = False,
    new_version: str | None = None,
) -> None:
    """Update metadata timestamps in the corrected file.

    Parameters
    ----------
    output_file : Path
        Path to the output file to update.
    processing_datetime : datetime, optional
        Processing datetime to use. If None, uses current time.
    update_version : bool
        Whether to update the product version.
    new_version : str, optional
        New version string. If None and update_version=True, appends ".1".

    """
    if processing_datetime is None:
        processing_datetime = datetime.now()

    with h5py.File(output_file, "a") as f:
        # Update processing_start_datetime
        old_datetime = f["/identification/processing_start_datetime"][()].decode(
            "utf-8"
        )
        logger.info(
            f"Updating processing_start_datetime from {old_datetime} to "
            f"{processing_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Delete and recreate the dataset with new value
        f["/identification/processing_start_datetime"][()] = (
            processing_datetime.strftime("%Y-%m-%d %H:%M:%S")
        ).encode("utf-8")

        # Optionally update product version
        if update_version:
            if new_version is None:
                raise ValueError("new_version must be specified if update_version=True")

            old_version = f["/identification/product_version"][()].decode("utf-8")
            logger.info(f"Updating product_version from {old_version} to {new_version}")
            f["/identification/product_version"][()] = new_version.encode("utf-8")


def recompute_perpendicular_baseline(
    input_file: Path,
    output_file: Path | None = None,
    subsample: int = 50,
    new_version: str | None = None,
) -> Path:
    """Recompute perpendicular baseline from saved orbit data.

    Parameters
    ----------
    input_file : Path
        Path to the input OPERA DISP-S1 NetCDF file.
    output_file : Path, optional
        Path to the output file. If not provided, will add "_corrected" suffix.
    subsample : int
        Subsampling factor for baseline computation, default 50.
    new_version : str
        New version string to replace in /identification/product_version.

    Returns
    -------
    Path
        Path to the output file.

    """
    input_file = Path(input_file)

    if output_file is None:
        output_file = input_file.with_stem(f"{input_file.stem}_corrected")

    output_file = Path(output_file)

    logger.info(f"Reading orbit data from {input_file}")

    # Load orbits from the netcdf file
    orbit_ref = load_orbit_from_netcdf(input_file, "/metadata/reference_orbit")
    orbit_sec = load_orbit_from_netcdf(input_file, "/metadata/secondary_orbit")

    # Get radar wavelength from identification group
    with h5py.File(input_file) as f:
        wavelength = float(f["/identification/radar_wavelength"][()])
        logger.info(f"Radar wavelength: {wavelength} m")

        # Get x/y coordinates and grid info
        x = f["/x"][:]
        y = f["/y"][:]
        shape = (len(y), len(x))
        logger.info(f"Grid shape: {shape}")

        # Get the CRS from the spatial_ref
        spatial_ref = f["/spatial_ref"]
        crs_wkt = spatial_ref.attrs["crs_wkt"]

    crs = CRS.from_wkt(crs_wkt)
    epsg = crs.to_epsg()
    logger.info(f"EPSG: {epsg}")

    # Subsample the grid for faster baseline computation
    x_sub = x[::subsample]
    y_sub = y[::subsample]
    logger.info(
        f"Computing baselines on subsampled grid: {len(y_sub)} x {len(x_sub)}"
        f" (subsample={subsample})"
    )

    baseline_arr = compute_baselines_from_orbits(
        orbit_ref=orbit_ref,
        orbit_sec=orbit_sec,
        x=x_sub,
        y=y_sub,
        epsg=epsg,
        wavelength=wavelength,
        height=0,
    )
    logger.info("Interpolating baselines to full resolution")

    # Interpolate back to full resolution
    baseline_full = interpolate_data(baseline_arr, shape=shape).astype("float32")

    # Copy the input file to output so we don't overwrite the original
    logger.info(f"Copying {input_file} to {output_file}")
    shutil.copy(input_file, output_file)

    # Update the perpendicular_baseline dataset
    logger.info("Writing perpendicular baseline to output file")
    with h5py.File(output_file, "a") as hf:
        hf["/corrections/perpendicular_baseline"][:] = baseline_full

    logger.info("Updating metadata timestamps")
    update_metadata_timestamps(
        output_file,
        processing_datetime=datetime.now(),
        new_version=new_version,
    )

    logger.info(f"Successfully wrote perpendicular baseline to {output_file}")
    return output_file


if __name__ == "__main__":
    tyro.cli(recompute_perpendicular_baseline)
