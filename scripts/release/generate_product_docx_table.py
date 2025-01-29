#!/usr/bin/env python
from __future__ import annotations

import argparse

import h5py
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Pt


def create_variable_table(document, variable_data):
    """Create a table for a single variable with consistent formatting."""
    table = document.add_table(rows=0, cols=1)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Add variable name
    name_row = table.add_row()
    name_cell = name_row.cells[0]
    name_cell.text = f"Name: {variable_data['Name']}"

    # Add type, shape, and units in one row
    info_row = table.add_row()
    info_cell = info_row.cells[0]
    info_table = info_cell.add_table(rows=1, cols=3)
    info_table.style = "Table Grid"

    # Populate type, shape, units
    info_table.cell(0, 0).text = f"Type: {variable_data['Type']}"
    info_table.cell(0, 1).text = f"Shape: {variable_data['Shape']}"
    info_table.cell(0, 2).text = f"Units: {variable_data.get('Units', 'unitless')}"

    # Add description
    desc_row = table.add_row()
    desc_cell = desc_row.cells[0]
    desc_cell.text = f"Description: {variable_data['Description']}"

    # Apply consistent formatting
    for row in table.rows:
        for cell in row.cells:
            cell.paragraphs[0].paragraph_format.space_before = Pt(6)
            cell.paragraphs[0].paragraph_format.space_after = Pt(6)

            # Set background color for name rows
            if "Name:" in cell.text:
                shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="D9D9D9"/>')
                cell._tc.get_or_add_tcPr().append(shading_elm)

    document.add_paragraph()  # Add spacing between variables


def process_hdf5_item(name, item):
    """Process a single HDF5 item (dataset or group) and return its data."""
    if isinstance(item, h5py.Dataset):
        return {
            "Name": name,
            "Type": str(item.dtype),
            "Shape": str(item.shape),
            "Units": item.attrs.get("units", "unitless"),
            "Description": item.attrs.get(
                "long_name", item.attrs.get("description", "")
            ),
        }
    return None


def generate_docx_table(hdf5_path: str, output_path: str):
    """Create a Word document with formatted tables matching the provided layout."""
    document = Document()

    # Set the default font to Arial
    style = document.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    with h5py.File(hdf5_path, "r") as hf:

        def visit_item(name, item):
            if isinstance(item, h5py.Dataset):
                group_name = name.rsplit("/", 1)[0] if "/" in name else "root"
                data = process_hdf5_item(name.rsplit("/", 1)[-1], item)
                if data:
                    if (
                        not hasattr(document, "_current_group")
                        or document._current_group != group_name
                    ):
                        document.add_heading(f"Group: {group_name}", level=1)
                        document._current_group = group_name
                    create_variable_table(document, data)

        # Visit all items in the HDF5 file
        hf.visititems(visit_item)

    document.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path", help="Path to the HDF5 file")
    parser.add_argument("output_path", help="Path to the output Word docx file")
    args = parser.parse_args()
    generate_docx_table(args.hdf5_path, args.output_path)
