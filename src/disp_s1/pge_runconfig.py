"""Module for creating PGE-compatible run configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, List, Optional, Union

from dolphin.workflows.config import (
    CorrectionOptions,
    DisplacementWorkflow,
    InterferogramNetwork,
    OutputOptions,
    PhaseLinkingOptions,
    PsOptions,
    UnwrapOptions,
    WorkerSettings,
)
from dolphin.workflows.config._yaml_model import YamlModel
from opera_utils import OPERA_DATASET_NAME
from pydantic import ConfigDict, Field

from .enums import ProcessingMode
from .utils import get_frame_bbox


class InputFileGroup(YamlModel):
    """Inputs for A group of input files."""

    cslc_file_list: List[Path] = Field(
        default_factory=list,
        description="list of paths to CSLC files.",
    )

    frame_id: int = Field(
        ...,
        description="Frame ID of the bursts contained in `cslc_file_list`.",
    )
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"required": ["cslc_file_list", "frame_id"]}
    )


class DynamicAncillaryFileGroup(YamlModel):
    """A group of dynamic ancillary files."""

    algorithm_parameters_file: Path = Field(
        default=...,
        description="Path to file containing SAS algorithm parameters.",
    )
    amplitude_dispersion_files: List[Path] = Field(
        default_factory=list,
        description=(
            "Paths to existing Amplitude Dispersion files (1 per burst) for PS update"
            " calculation. If none provided, computed using the input SLC stack."
        ),
    )
    amplitude_mean_files: List[Path] = Field(
        default_factory=list,
        description=(
            "Paths to an existing Amplitude Mean files (1 per burst) for PS update"
            " calculation. If none provided, computed using the input SLC stack."
        ),
    )
    geometry_files: List[Path] = Field(
        default_factory=list,
        alias="static_layers_files",
        description=(
            "Paths to the CSLC static_layer files (1 per burst) with line-of-sight"
            " unit vectors. If none provided, corrections using CSLC static_layer are"
            " skipped."
        ),
    )
    mask_file: Optional[Path] = Field(
        None,
        description=(
            "Optional Byte mask file used to ignore low correlation/bad data (e.g water"
            " mask). Convention is 0 for no data/invalid, and 1 for good data. Dtype"
            " must be uint8."
        ),
    )
    dem_file: Optional[Path] = Field(
        default=None,
        description=(
            "Path to the DEM file covering full frame. If none provided, corrections"
            " using DEM are skipped."
        ),
    )
    # TEC file in IONEX format for ionosphere correction
    ionosphere_files: Optional[List[Path]] = Field(
        default=None,
        description=(
            "List of paths to TEC files (1 per date) in IONEX format for ionosphere"
            " correction. If none provided, ionosphere corrections are skipped."
        ),
    )

    # Troposphere weather model
    troposphere_files: Optional[List[Path]] = Field(
        default=None,
        description=(
            "List of paths to troposphere weather model files (1 per date). If none"
            " provided, troposphere corrections are skipped."
        ),
    )
    model_config = ConfigDict(extra="forbid")


class StaticAncillaryFileGroup(YamlModel):
    """Group for files which remain static over time."""

    frame_to_burst_json: Union[Path, None] = Field(
        None,
        description=(
            "JSON file containing the mapping from frame_id to frame/burst information"
        ),
    )


class PrimaryExecutable(YamlModel):
    """Group describing the primary executable."""

    product_type: str = Field(
        default="DISP_S1_FORWARD",
        description="Product type of the PGE.",
    )
    model_config = ConfigDict(extra="forbid")


class ProductPathGroup(YamlModel):
    """Group describing the product paths."""

    product_path: Path = Field(
        default=...,
        description="Directory where PGE will place results",
    )
    scratch_path: Path = Field(
        default=Path("./scratch"),
        description="Path to the scratch directory.",
    )
    output_directory: Path = Field(
        default=Path("./output"),
        description="Path to the SAS output directory.",
        # The alias means that in the YAML file, the key will be "sas_output_path"
        # instead of "output_directory", but the python instance attribute is
        # "output_directory" (to match DisplacementWorkflow)
        alias="sas_output_path",
    )
    product_version: str = Field(
        default="0.2",
        description="Version of the product, in <major>.<minor> format.",
    )
    save_compressed_slc: bool = Field(
        default=False,
        description=(
            "Whether the SAS should output and save the Compressed SLCs in addition to"
            " the standard product output."
        ),
    )
    model_config = ConfigDict(extra="forbid")


class AlgorithmParameters(YamlModel):
    """Class containing all the other `DisplacementWorkflow` classes."""

    # Options for each step in the workflow
    ps_options: PsOptions = Field(default_factory=PsOptions)
    phase_linking: PhaseLinkingOptions = Field(default_factory=PhaseLinkingOptions)
    interferogram_network: InterferogramNetwork = Field(
        default_factory=InterferogramNetwork
    )
    unwrap_options: UnwrapOptions = Field(default_factory=UnwrapOptions)
    correction_options: CorrectionOptions = Field(default_factory=CorrectionOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)
    subdataset: str = Field(
        default=OPERA_DATASET_NAME,
        description="Name of the subdataset to use in the input NetCDF files.",
    )
    model_config = ConfigDict(extra="forbid")


class RunConfig(YamlModel):
    """A PGE run configuration."""

    # Used for the top-level key
    name: ClassVar[str] = "disp_s1_workflow"

    input_file_group: InputFileGroup
    dynamic_ancillary_file_group: DynamicAncillaryFileGroup
    static_ancillary_file_group: StaticAncillaryFileGroup
    primary_executable: PrimaryExecutable = Field(default_factory=PrimaryExecutable)
    product_path_group: ProductPathGroup

    # General workflow metadata
    worker_settings: WorkerSettings = Field(default_factory=WorkerSettings)

    log_file: Optional[Path] = Field(
        default=Path("output/disp_s1_workflow.log"),
        description="Path to the output log file in addition to logging to stderr.",
    )
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def model_construct(cls, **kwargs):
        """Recursively use model_construct without validation."""
        if "input_file_group" not in kwargs:
            kwargs["input_file_group"] = InputFileGroup._construct_empty()
        if "dynamic_ancillary_file_group" not in kwargs:
            kwargs[
                "dynamic_ancillary_file_group"
            ] = DynamicAncillaryFileGroup._construct_empty()
        if "static_ancillary_file_group" not in kwargs:
            kwargs[
                "static_ancillary_file_group"
            ] = StaticAncillaryFileGroup._construct_empty()
        if "product_path_group" not in kwargs:
            kwargs["product_path_group"] = ProductPathGroup._construct_empty()
        return super().model_construct(
            **kwargs,
        )

    def to_workflow(self):
        """Convert to a `DisplacementWorkflow` object."""
        # We need to go to/from the PGE format to dolphin's DisplacementWorkflow:
        # Note that the top two levels of nesting can be accomplished by wrapping
        # the normal model export in a dict.
        #
        # The things from the RunConfig that are used in the
        # DisplacementWorkflow are the input files, PS amp mean/disp files,
        # the output directory, and the scratch directory.
        # All the other things come from the AlgorithmParameters.

        cslc_file_list = self.input_file_group.cslc_file_list
        scratch_directory = self.product_path_group.scratch_path
        mask_file = self.dynamic_ancillary_file_group.mask_file
        amplitude_mean_files = self.dynamic_ancillary_file_group.amplitude_mean_files
        amplitude_dispersion_files = (
            self.dynamic_ancillary_file_group.amplitude_dispersion_files
        )

        # Load the algorithm parameters from the file
        algorithm_parameters = AlgorithmParameters.from_yaml(
            self.dynamic_ancillary_file_group.algorithm_parameters_file
        )
        param_dict = algorithm_parameters.model_dump()
        input_options = {"subdataset": param_dict.pop("subdataset")}

        # Convert the frame_id into an output bounding box
        frame_to_burst_file = self.static_ancillary_file_group.frame_to_burst_json
        frame_id = self.input_file_group.frame_id
        bounds_epsg, bounds = get_frame_bbox(
            frame_id=frame_id, json_file=frame_to_burst_file
        )
        param_dict["output_options"]["bounds"] = bounds
        param_dict["output_options"]["bounds_epsg"] = bounds_epsg

        # unpacked to load the rest of the parameters for the DisplacementWorkflow
        return DisplacementWorkflow(
            cslc_file_list=cslc_file_list,
            input_options=input_options,
            mask_file=mask_file,
            work_directory=scratch_directory,
            amplitude_mean_files=amplitude_mean_files,
            amplitude_dispersion_files=amplitude_dispersion_files,
            # These ones directly translate
            worker_settings=self.worker_settings,
            log_file=self.log_file,
            # Finally, the rest of the parameters are in the algorithm parameters
            **param_dict,
        )

    @classmethod
    def from_workflow(
        cls,
        workflow: DisplacementWorkflow,
        frame_id: int,
        processing_mode: ProcessingMode,
        algorithm_parameters_file: Path,
        frame_to_burst_json: Optional[Path] = None,
        save_compressed_slc: bool = False,
        output_directory: Optional[Path] = None,
    ):
        """Convert from a `DisplacementWorkflow` object.

        This is the inverse of the to_workflow method, although there are more
        fields in the PGE version, so it's not a 1-1 mapping.

        The arguments, like `frame_id` or `algorithm_parameters_file`, are not in the
        `DisplacementWorkflow` object, so we need to pass
        those in as arguments.

        This is can be used as preliminary setup to further edit the fields, or as a
        complete conversion.
        """
        if output_directory is None:
            # Take the output as one above the scratch
            output_directory = workflow.work_directory.parent / "output"

        # Load the algorithm parameters from the file
        algo_keys = set(AlgorithmParameters.model_fields.keys())
        alg_param_dict = workflow.model_dump(include=algo_keys)
        AlgorithmParameters(**alg_param_dict).to_yaml(algorithm_parameters_file)
        # unpacked to load the rest of the parameters for the DisplacementWorkflow

        return cls(
            input_file_group=InputFileGroup(
                cslc_file_list=workflow.cslc_file_list,
                frame_id=frame_id,
            ),
            dynamic_ancillary_file_group=DynamicAncillaryFileGroup(
                algorithm_parameters_file=algorithm_parameters_file,
                amplitude_dispersion_files=workflow.amplitude_dispersion_files,
                amplitude_mean_files=workflow.amplitude_mean_files,
                mask_file=workflow.mask_file,
                ionosphere_files=workflow.correction_options.ionosphere_files,
                troposphere_files=workflow.correction_options.troposphere_files,
                dem_file=workflow.correction_options.dem_file,
                static_layers_files=workflow.correction_options.geometry_files,
            ),
            static_ancillary_file_group=StaticAncillaryFileGroup(
                frame_to_burst_json=frame_to_burst_json,
            ),
            primary_executable=PrimaryExecutable(
                product_type=f"DISP_S1_{processing_mode.upper()}",
            ),
            product_path_group=ProductPathGroup(
                product_path=output_directory,
                scratch_path=workflow.work_directory,
                sas_output_path=output_directory,
                save_compressed_slc=save_compressed_slc,
            ),
            worker_settings=workflow.worker_settings,
            log_file=workflow.log_file,
        )
