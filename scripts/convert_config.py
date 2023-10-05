#!/usr/bin/env python
import argparse
from pathlib import Path

from dolphin.workflows.config import Workflow

from disp_s1.pge_runconfig import RunConfig
from disp_s1.utils import FRAME_TO_BURST_JSON_FILE


def main():
    """Run the conversion CLI."""
    parser = argparse.ArgumentParser(
        description="Convert a `dolphin_config.yaml` to `runconfig.yaml` for SDS"
    )

    parser.add_argument(
        "dolphin_config_file", type=str, help="Path to dolphin configuration YAML file."
    )
    parser.add_argument("frame_id", type=str, help="Frame ID.")
    parser.add_argument("processing_mode", type=str, help="Processing mode.")
    parser.add_argument(
        "-o", "--outfile", type=str, default="runconfig.yaml", help="Output file path."
    )
    parser.add_argument(
        "--frame-to-burst-json",
        type=Path,
        default=FRAME_TO_BURST_JSON_FILE,
        help="Path to algorithm parameters file.",
    )
    parser.add_argument(
        "-a",
        "--algorithm-parameters-file",
        type=Path,
        default="algorithm_parameters.yaml",
        help="Path to algorithm parameters file.",
    )

    args = parser.parse_args()

    workflow = Workflow.from_yaml(args.dolphin_config_file)
    rc = RunConfig.from_workflow(
        workflow,
        frame_id=args.frame_id,
        frame_to_burst_json=args.frame_to_burst_json,
        algorithm_parameters_file=args.algorithm_parameters_file,
        processing_mode=args.processing_mode,
    )
    rc.to_yaml(args.outfile)


if __name__ == "__main__":
    main()
