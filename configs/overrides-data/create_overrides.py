"""Create a combined algorithm overrides JSON file.

This module creates JSON configuration files containing algorithm parameter
overrides for OPERA DISP-S1 processing.
It combines manual overrides with geographically-defined overrides for
specific regions, such as Alaska and the Pacific Northwest Coast.

The region parameters are made using a single config dict (passable to `dolphin`),
which is applied to all frames touching the associated `.wkt` polygon file.

"""

import datetime
import json
from pathlib import Path
from typing import Literal

import opera_utils
from shapely import from_wkt

OVERRIDE_PATH = Path("configs/overrides-data/")
gdf_frames = opera_utils.get_frame_geodataframe()


def read_json_with_comments(path: Path) -> dict:
    """Read a JSON file while filtering out lines containing comments.

    This function reads a JSON file and removes any lines that contain "//"
    comment markers before parsing the JSON content.

    Parameters
    ----------
    path : Path
        Path to the JSON file to read.

    Returns
    -------
    dict
        Parsed JSON content as a dictionary.

    """
    lines = path.read_text().splitlines()
    return json.loads("\n".join([line for line in lines if "//" not in line]))


def read_override(name: str, path=OVERRIDE_PATH):
    """Read override configuration and geometry for a named region.

    This function reads both a WKT geometry file and a JSON configuration
    file for a specified override region.

    Parameters
    ----------
    name : str
        Name of the override region. Used to construct filenames as
        "{name}.wkt" and "{name}.json".
    path : Path, optional
        Directory path containing the override files, by default OVERRIDE_PATH.

    Returns
    -------
    tuple[shapely.Geometry, dict]
        A tuple containing:
        - Shapely geometry object from the WKT file
        - Dictionary of override parameters from the JSON file

    """
    poly = from_wkt((path / f"{name}.wkt").read_text())
    overrides = read_json_with_comments((path / f"{name}.json"))
    return poly, overrides


def form_overrides(name: Literal["alaska-middle", "west-coast-green"]):
    """Generate frame-specific overrides for a named geographic region.

    This function creates algorithm parameter overrides for all North American
    frames that intersect with a specified geographic region.

    Parameters
    ----------
    name : {"alaska-middle", "west-coast-green"}
        Name of the geographic region to process. Must be one of the
        supported region names.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping frame IDs (as strings) to their override
        parameters. All intersecting frames receive the same override
        configuration.

    Notes
    -----
    Only processes frames marked as North American (`is_north_america=True`)
    that geometrically intersect with the specified region polygon.

    """
    poly, data = read_override(name)
    frame_ids = gdf_frames[
        (gdf_frames.is_north_america) & (gdf_frames.intersects(poly))
    ].index
    return dict.fromkeys(map(str, frame_ids), data)


def combine_overrides(path: Path = OVERRIDE_PATH):
    """Combine all override sources into a single configuration dictionary.

    This function merges manual overrides with geographically-generated
    overrides for Alaska and West Coast regions.
    Later entries in the merge order take precedence over earlier ones.
    Manual entries take highest precedence.

    Parameters
    ----------
    path : Path, optional
        Directory path containing override configuration files,
        by default OVERRIDE_PATH.

    Returns
    -------
    dict[str, dict]
        Combined dictionary of frame ID to override parameters.
        Merge precedence (highest to lowest):
        1. Manual overrides from "manual-overrides-region123.json"
        2. Alaska middle region overrides
        3. West coast green region overrides

    """
    manual_overrides = read_json_with_comments(
        (path / "manual-overrides-region123.json")
    )
    alaska_overrides = form_overrides("alaska-middle")
    west_coast_overrides = form_overrides("west-coast-green")
    # keys from the rightmost dictionary take precedence in case of conflicts.

    combined = west_coast_overrides | alaska_overrides | manual_overrides
    return combined


if __name__ == "__main__":
    d = combine_overrides()
    todaystr = datetime.datetime.today().strftime("%Y-%m-%d")
    name = f"opera-disp-s1-algorithm-parameters-overrides-{todaystr}.json"
    out_path = Path(OVERRIDE_PATH / name)
    out_path.write_text(json.dumps(d, indent=2))
    print(f"Created {out_path}")
