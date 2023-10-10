#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from dolphin.workflows.config import Workflow

from disp_s1.pge_runconfig import RunConfig


def convert_to_runconfig(
    dolphin_config_file: str,
    frame_id: str,
    processing_mode: str,
    frame_to_burst_json: str | None = None,
    algorithm_parameters_file: str = "algorithm_parameters.yaml",
    save_compressed_slc: bool = True,
    outfile: str = "runconfig.yaml",
):
    """Run the conversion CLI."""
    workflow = Workflow.from_yaml(dolphin_config_file)
    rc = RunConfig.from_workflow(
        workflow,
        frame_id=frame_id,
        frame_to_burst_json=frame_to_burst_json,
        algorithm_parameters_file=algorithm_parameters_file,
        processing_mode=processing_mode,
        save_compressed_slc=save_compressed_slc,
    )
    rc.to_yaml(outfile)


def main():
    """Run the conversion CLI."""
    parser = argparse.ArgumentParser(
        description="Convert a `dolphin_config.yaml` to `runconfig.yaml` for SDS"
    )

    parser.add_argument(
        "dolphin_config_file", type=str, help="Path to dolphin configuration YAML file."
    )
    parser.add_argument("--frame-id", required=True, type=str, help="Frame ID.")
    parser.add_argument(
        "--processing-mode", required=True, type=str, help="Processing mode."
    )
    parser.add_argument(
        "-o", "--outfile", type=str, default="runconfig.yaml", help="Output file path."
    )
    parser.add_argument(
        "--frame-to-burst-json",
        type=Path,
        help=(
            "Path to frame-to-burst mapping JSON file, summarizing DISP frame database."
        ),
    )
    parser.add_argument(
        "-a",
        "--algorithm-parameters-file",
        type=Path,
        default="algorithm_parameters.yaml",
        help="Path to algorithm parameters file.",
    )
    parser.add_argument(
        "--save-compressed-slc",
        action="store_true",
        help=(
            "Indicate in the runconfig that the compressed SLCs should be saved in the"
            " output. Note that for the historical mode, this will only save the"
        ),
    )

    args = parser.parse_args()

    convert_to_runconfig(
        dolphin_config_file=args.dolphin_config_file,
        frame_id=args.frame_id,
        processing_mode=args.processing_mode,
        frame_to_burst_json=args.frame_to_burst_json,
        algorithm_parameters_file=args.algorithm_parameters_file,
        save_compressed_slc=args.save_compressed_slc,
        outfile=args.outfile,
    )


if __name__ == "__main__":
    main()
