import logging
from pathlib import Path
from typing import Any

from dolphin._types import Filename
from dolphin.workflows.config import DisplacementWorkflow

from disp_s1.pge_runconfig import RunConfig
from disp_s1.product import create_output_product

from .enums import ProcessingMode

logger = logging.getLogger(__name__)


def get_params(
    process_dir: Path,
    cslc_list: Filename,
    pair: str,
    frame_id: int,
    processing_mode: ProcessingMode,
):
    """Get required parameters and write to a dictionary."""
    # Get necessary files.
    param_dict: dict[str, Any] = {}
    logger.info("Reading files:")
    try:
        param_dict["unw_filename"] = next(
            Path(f"{process_dir}/unwrapped/").glob(f"{pair}.unw.tif")
        )
        logger.info(param_dict["unw_filename"])
    except StopIteration as e:
        logger.error("Check if the pair %s exists", pair)
        raise FileNotFoundError(f"Pair {pair} not found") from e
    param_dict["conncomp_filename"] = next(
        Path(f"{process_dir}/unwrapped/").glob(f"{pair}.unw.conncomp.tif")
    )
    logger.info(param_dict["conncomp_filename"])
    param_dict["ifg_corr_filename"] = next(
        Path(f"{process_dir}/interferograms/").glob(f"{pair}.cor")
    )
    logger.info(param_dict["ifg_corr_filename"])
    param_dict["temp_coh_filename"] = next(
        Path(f"{process_dir}/interferograms/").glob("temporal_coherence*tif")
    )
    logger.info(param_dict["temp_coh_filename"])
    param_dict["ps_mask_filename"] = next(
        Path(f"{process_dir}/interferograms/").glob("ps_mask_looked.tif")
    )
    logger.info(param_dict["ps_mask_filename"])
    param_dict["dolphin_config"] = next(Path(f"{process_dir}/").glob("dolphin*yaml"))
    logger.info(param_dict["dolphin_config"])
    param_dict["output_folder"] = Path(f"{process_dir}/disp_product_outputs/{pair}")
    param_dict["output_name"] = Path(f"{param_dict['output_folder']}/{pair}.nc")

    # Make product directory
    param_dict["output_folder"].mkdir(exist_ok=True, parents=True)

    # Read CSLC list file
    cslc_files = sorted(cslc_list.read().splitlines())
    param_dict["cslc_files"] = cslc_files

    workflow = DisplacementWorkflow.from_yaml(param_dict["dolphin_config"])
    rc = RunConfig.from_workflow(
        workflow,
        frame_id=frame_id,
        frame_to_burst_json=None,
        algorithm_parameters_file=Path(
            f"{param_dict['output_folder']}/algorithm_parameters.yaml"
        ),
        processing_mode=processing_mode,
        save_compressed_slc=True,
        output_directory=Path(f"{param_dict['output_folder']}"),
    )

    param_dict["pge_runconfig"] = rc

    logger.info("Gathered all required information.")

    return param_dict


def make_product(param_dict: dict) -> None:
    """Create product in the given directory."""
    create_output_product(
        unw_filename=param_dict["unw_filename"],
        conncomp_filename=param_dict["conncomp_filename"],
        temp_coh_filename=param_dict["temp_coh_filename"],
        ifg_corr_filename=param_dict["ifg_corr_filename"],
        output_name=param_dict["output_name"],
        corrections={},
        ps_mask_filename=param_dict["ps_mask_filename"],
        pge_runconfig=param_dict["pge_runconfig"],
        cslc_files=param_dict["cslc_files"],
    )
    logger.info("DISP-S1 product created: %s", param_dict["output_name"])
