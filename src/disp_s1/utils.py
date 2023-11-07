from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Optional

from dolphin._types import Bbox, Filename

FRAME_TO_BURST_JSON_FILE = (
    Path(__file__).parent / "data" / "opera-s1-disp-frame-to-burst.json.zip"
)


def read_zipped_json(filename: Filename):
    """Read a zipped JSON file and returns its contents as a dictionary.

    Parameters
    ----------
    filename : Filename
        The path to the zipped JSON file.

    Returns
    -------
    dict
        The contents of the zipped JSON file as a dictionary.
    """
    if Path(filename).suffix == ".zip":
        with zipfile.ZipFile(filename) as zf:
            bytes = zf.read(str(Path(filename).name).replace(".zip", ""))
            return json.loads(bytes.decode())
    else:
        with open(filename) as f:
            return json.load(f)


def get_frame_json(
    frame_id: int, json_file: Optional[Filename] = FRAME_TO_BURST_JSON_FILE
) -> dict:
    """Get the frame data for one frame ID.

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get the bounding box for.
    json_file : Filename, optional
        The path to the JSON file containing the frame-to-burst mapping.
        If `None`, uses the zip file contained in `disp_s1/data`
    Returns
    -------
    dict
        The frame data for the given frame ID.
    """
    if json_file is None:
        json_file = FRAME_TO_BURST_JSON_FILE
    js = read_zipped_json(json_file)
    return js["data"][str(frame_id)]


def get_frame_bbox(
    frame_id: int, json_file: Optional[Filename] = FRAME_TO_BURST_JSON_FILE
) -> tuple[int, Bbox]:
    """Get the bounding box of a frame from a JSON file.

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get the bounding box for.
    json_file : Filename, optional
        The path to the JSON file containing the frame-to-burst mapping.
        If `None`, uses the zip file contained in `disp_s1/data`

    Returns
    -------
    epsg : int
        EPSG code for the bounds coordinates
    tuple[float, float, float, float]
        bounding box coordinates (xmin, ymin, xmax, ymax)
    """
    frame_dict = get_frame_json(frame_id=frame_id, json_file=json_file)
    epsg = frame_dict["epsg"]
    bounds = [frame_dict[k] for k in ["xmin", "ymin", "xmax", "ymax"]]
    return epsg, bounds


def get_burst_ids_for_frame(
    frame_id: int, json_file: Optional[Filename] = FRAME_TO_BURST_JSON_FILE
) -> list[str]:
    """Get the burst IDs for one frame ID.

    Parameters
    ----------
    frame_id : int
        The ID of the frame to get the bounding box for.
    json_file : Filename, optional
        The path to the JSON file containing the frame-to-burst mapping.
        If `None`, uses the zip file contained in `disp_s1/data`

    Returns
    -------
    list[str]
        The burst IDs for the given frame ID.
    """
    frame_data = get_frame_json(frame_id, json_file)
    return frame_data["burst_ids"]
