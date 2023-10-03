import json
import zipfile
from pathlib import Path
from typing import Optional

from dolphin._types import Bbox, Filename

FRAME_TO_BURST_JSON_FILE = (
    Path(__file__) / "data" / "opera-s1-disp-frame-to-burst.json.zip"
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
    if json_file is None:
        json_file = FRAME_TO_BURST_JSON_FILE
    js = read_zipped_json(json_file)
    frame_dict = js["data"][str(frame_id)]
    epsg = frame_dict["epsg"]
    bounds = (frame_dict[k] for k in ["xmin", "ymin", "xmax", "ymax"])
    return epsg, bounds
