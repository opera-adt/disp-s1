#!/usr/bin/env python
# /// script
# dependencies = ["h5py"]
# ///
"""Convert a NetCDF4/HDF5 file into XML.

Can be used to browse metadata contained in the file with a text editor.

Creates output files with the same path as the input HDF5 with .xml extensions.
"""

import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import h5py

IGNORE_ATTRIBUTES = [
    "_Netcdf4Dimid",
    "NAME",
    "REFERENCE_LIST",
    "_Netcdf4Coordinates",
    "CLASS",
    "DIMENSION_LIST",
    "_NCProperties",
    "_nc3_strict",
]


def _to_str(value: Any) -> str:
    """Convert a value to string, decoding bytes as UTF-8.

    Parameters
    ----------
    value : Any
        The value to be converted.

    Returns
    -------
    str
        The string representation of the value.

    """
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def hdf5_to_xml(hdf5_file_path: Path | str, xml_file_path: Path | str) -> None:
    """Convert an HDF5 file to XML format.

    Parameters
    ----------
    hdf5_file_path : Path | str
        Path to the input HDF5 file.
    xml_file_path : Path | str
        Path to the output XML file.

    """
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        root = ET.Element("group", name="/")
        process_attributes(hdf5_file, root)
        path_to_element: dict[str, ET.Element] = {"/": root}

        def visitor_func(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            process_item(name, obj, path_to_element)

        hdf5_file.visititems(visitor_func)

    xml_string = ET.tostring(root, encoding="unicode")
    xml_pretty = minidom.parseString(xml_string).toprettyxml(indent="  ")

    with open(xml_file_path, "w", encoding="utf-8") as xml_file:
        xml_file.write(xml_pretty)


def process_item(
    name: str, obj: h5py.Group | h5py.Dataset, path_to_element: dict[str, ET.Element]
) -> None:
    """Process an HDF5 item (group or dataset) in place, add it to the XML structure.

    Parameters
    ----------
    name : str
        Name of the HDF5 item.
    obj : h5py.Group| h5py.Dataset
        The HDF5 object (group or dataset).
    path_to_element : dict[str, ET.Element]
        Dictionary mapping HDF5 paths to corresponding XML elements.

    """
    if "/" in name:
        parent_path_list = name.split("/")[:-1]
        parent_path = "/" + "/".join([p for p in parent_path_list if p])
    else:
        parent_path = "/"
    parent_element = path_to_element[parent_path]
    element_name = name.split("/")[-1] if name else "/"

    if isinstance(obj, h5py.Group):
        group_element = ET.SubElement(parent_element, "group", name=element_name)
        process_attributes(obj, group_element)
        full_path = "/" + name if not name.startswith("/") else name
        path_to_element[full_path] = group_element
    elif isinstance(obj, h5py.Dataset):
        dataset_element = ET.SubElement(parent_element, "dataset", name=element_name)
        process_attributes(obj, dataset_element)
        process_dataset_info(obj, dataset_element)
        full_path = "/" + name if not name.startswith("/") else name
        path_to_element[full_path] = dataset_element


def process_attributes(
    item: h5py.File | h5py.Group | h5py.Dataset, element: ET.Element
) -> None:
    """Process attributes of an HDF5 item and add them to the XML element.

    Parameters
    ----------
    item : h5py.File | h5py.Group | h5py.Dataset
        The HDF5 item whose attributes are to be processed.
    element : ET.Element
        The XML element to which attributes will be added.

    """
    for attr_name, attr_value in item.attrs.items():
        if attr_name not in IGNORE_ATTRIBUTES:
            attr_element = ET.SubElement(element, "attribute", name=attr_name)
            attr_element.text = _to_str(attr_value)


def process_dataset_info(dataset: h5py.Dataset, element: ET.Element) -> None:
    """Process dataset information and add it to the XML element.

    Parameters
    ----------
    dataset : h5py.Dataset
        The HDF5 dataset to be processed.
    element : ET.Element
        The XML element to which dataset information will be added.

    """
    shape_element = ET.SubElement(element, "shape")
    shape_element.text = str(dataset.shape)
    dtype_element = ET.SubElement(element, "dtype")
    dtype_element.text = str(dataset.dtype)

    if dataset.size <= 10:
        values_element = ET.SubElement(element, "values")
        data = dataset[...]
        if data.ndim == 0:
            item = data.item()
            values_element.text = _to_str(item)
        else:
            values_element.text = str([_to_str(v) for v in data.flatten().tolist()])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for hdf5_file_path in sys.argv[1:]:
            xml_file_path = hdf5_file_path + ".xml"
            if Path(xml_file_path).exists():
                print(f"{xml_file_path} exists: Skipping")
                continue
            hdf5_to_xml(hdf5_file_path, xml_file_path)
    else:
        print(f"Usage: {sys.argv[0]} filepath [filepath [...]]")
        raise SystemExit("No HDF5 file path given")
