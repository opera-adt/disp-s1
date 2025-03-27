"""Module for creating PGE-compatible run configuration files."""

from __future__ import annotations

import datetime
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, ClassVar, List, Literal, Optional, Union

from dolphin.stack import CompressedSlcPlan
from dolphin.workflows.config import (
    CorrectionOptions,
    DisplacementWorkflow,
    InterferogramNetwork,
    OutputOptions,
    PhaseLinkingOptions,
    PsOptions,
    TimeseriesOptions,
    UnwrapOptions,
    WorkerSettings,
    YamlModel,
)
from dolphin.workflows.config._common import _read_file_list_or_glob
from opera_utils import (
    OPERA_DATASET_NAME,
    PathOrStr,
    get_burst_ids_for_frame,
    get_dates,
    get_frame_bbox,
    group_by_burst,
    sort_files_by_date,
)
from pydantic import ConfigDict, Field, field_validator

from .enums import ProcessingMode


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

    _check_cslc_file_glob = field_validator("cslc_file_list", mode="before")(
        _read_file_list_or_glob
    )


class DynamicAncillaryFileGroup(YamlModel):
    """A group of dynamic ancillary files."""

    algorithm_parameters_file: Path = Field(
        default=...,
        description="Path to file containing SAS algorithm parameters.",
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
    reference_date_database_json: Union[Path, None] = Field(
        None,
        description=(
            "JSON file containing list of reference date changes for each frame"
        ),
    )


class PrimaryExecutable(YamlModel):
    """Group describing the primary executable."""

    product_type: Literal["DISP_S1_FORWARD", "DISP_S1_HISTORICAL", "DISP_S1_STATIC"] = (
        Field(description="Product type of the PGE.")
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
        default="0.3",
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

    algorithm_parameters_overrides_json: Union[Path, None] = Field(
        None,
        description=(
            "JSON file containing frame-specific algorithm parameters to override the"
            " defaults passed in the `algorithm_parameters.yaml`."
        ),
    )

    # Options for each step in the workflow
    ps_options: PsOptions = Field(default_factory=PsOptions)
    phase_linking: PhaseLinkingOptions = Field(default_factory=PhaseLinkingOptions)
    interferogram_network: InterferogramNetwork = Field(
        default_factory=InterferogramNetwork
    )
    unwrap_options: UnwrapOptions = Field(default_factory=UnwrapOptions)
    timeseries_options: TimeseriesOptions = Field(default_factory=TimeseriesOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)

    subdataset: str = Field(
        default=OPERA_DATASET_NAME,
        description="Name of the subdataset to use in the input NetCDF files.",
    )

    recommended_temporal_coherence_threshold: float = Field(
        0.6,
        description=(
            "When creating `recommended_mask`, pixels with temporal coherence below"
            " this threshold and with similarity below"
            " `recommended_similarity_threshold` are masked."
        ),
    )
    recommended_similarity_threshold: float = Field(
        0.5,
        description=(
            "When creating `recommended_mask`, pixels with similarity below this"
            " threshold and with temporal coherence below"
            " `recommended_temporal_coherence_threshold` are masked."
        ),
    )
    recommended_use_conncomp: bool = Field(
        False,
        description=(
            "When creating `recommended_mask`, use the `connected_component_label`"
            " layer to hide pixels whose label == 0."
        ),
    )
    # Extra product creation options
    spatial_wavelength_cutoff: float = Field(
        25_000,
        description=(
            "Spatial wavelength cutoff (in meters) for the spatial filter. Used to"
            " create the short wavelength displacement layer"
        ),
    )
    browse_image_vmin_vmax: tuple[float, float] = Field(
        (-0.10, 0.10),
        description=(
            "`vmin, vmax` matplotlib arguments (in meters) passed to browse image"
            " creator."
        ),
    )
    num_parallel_products: int = Field(
        3, description="Number of output products to create in parallel."
    )

    model_config = ConfigDict(extra="forbid")


class RunConfig(YamlModel):
    """A PGE run configuration."""

    # Used for the top-level key
    name: ClassVar[str] = "disp_s1_workflow"

    input_file_group: InputFileGroup
    dynamic_ancillary_file_group: DynamicAncillaryFileGroup
    static_ancillary_file_group: StaticAncillaryFileGroup
    primary_executable: PrimaryExecutable
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
            kwargs["input_file_group"] = InputFileGroup.model_construct()
        if "dynamic_ancillary_file_group" not in kwargs:
            kwargs["dynamic_ancillary_file_group"] = (
                DynamicAncillaryFileGroup.model_construct()
            )
        if "static_ancillary_file_group" not in kwargs:
            kwargs["static_ancillary_file_group"] = (
                StaticAncillaryFileGroup.model_construct()
            )
        if "primary_executable" not in kwargs:
            kwargs["primary_executable"] = PrimaryExecutable.model_construct()
        if "product_path_group" not in kwargs:
            kwargs["product_path_group"] = ProductPathGroup.model_construct()
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

        # PGE doesn't sort the CSLCs in date order (or any order?)
        cslc_file_list = sort_files_by_date(self.input_file_group.cslc_file_list)[0]
        scratch_directory = self.product_path_group.scratch_path
        mask_file = self.dynamic_ancillary_file_group.mask_file
        geometry_files = self.dynamic_ancillary_file_group.geometry_files
        ionosphere_files = self.dynamic_ancillary_file_group.ionosphere_files
        dem_file = self.dynamic_ancillary_file_group.dem_file
        frame_id = self.input_file_group.frame_id

        # Load the algorithm parameters from the file
        algorithm_parameters = AlgorithmParameters.from_yaml(
            self.dynamic_ancillary_file_group.algorithm_parameters_file,
        )
        new_parameters = _override_parameters(algorithm_parameters, frame_id=frame_id)
        # regenerate to ensure all defaults remained in updated version
        algo_params = AlgorithmParameters(**new_parameters.model_dump())
        param_dict = algo_params.model_dump()

        # Convert the frame_id into an output bounding box
        frame_to_burst_file = self.static_ancillary_file_group.frame_to_burst_json
        bounds_epsg, bounds = get_frame_bbox(
            frame_id=frame_id, json_file=frame_to_burst_file
        )

        # Check for consistency of frame and burst ids
        frame_burst_ids = set(
            get_burst_ids_for_frame(frame_id=frame_id, json_file=frame_to_burst_file)
        )
        data_burst_ids = set(group_by_burst(cslc_file_list).keys())
        mismatched_bursts = data_burst_ids - frame_burst_ids
        if mismatched_bursts:
            raise ValueError("The CSLC data and frame id do not match")

        # Setup the OPERA-specific options to adjust from dolphin's defaults
        input_options = {"subdataset": param_dict.pop("subdataset")}
        param_dict["output_options"]["bounds"] = bounds
        param_dict["output_options"]["bounds_epsg"] = bounds_epsg
        # Always turn off overviews (won't be saved in the HDF5 anyway)
        param_dict["output_options"]["add_overviews"] = False
        # Always turn off velocity (not used) in output product
        param_dict["timeseries_options"]["run_velocity"] = False
        # Always use L1 minimization for inverting unwrapped networks
        param_dict["timeseries_options"]["method"] = "L1"

        # Get the current set of expected reference dates
        reference_datetimes = _parse_reference_date_json(
            self.static_ancillary_file_group.reference_date_database_json, frame_id
        )
        # Compute the requested output indexes
        output_reference_idx, extra_reference_date = _compute_reference_dates(
            reference_datetimes,
            cslc_file_list,
            algo_params.phase_linking.compressed_slc_plan,
        )
        param_dict["phase_linking"]["output_reference_idx"] = output_reference_idx
        param_dict["output_options"]["extra_reference_date"] = extra_reference_date

        # unpacked to load the rest of the parameters for the DisplacementWorkflow
        return DisplacementWorkflow(
            cslc_file_list=cslc_file_list,
            input_options=input_options,
            mask_file=mask_file,
            work_directory=scratch_directory,
            # These ones directly translate
            worker_settings=self.worker_settings,
            correction_options=CorrectionOptions(
                ionosphere_files=ionosphere_files,
                # troposphere_files=troposphere_files,
                geometry_files=geometry_files,
                dem_file=dem_file,
            ),
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
        reference_date_json: Optional[Path] = None,
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
                mask_file=workflow.mask_file,
                ionosphere_files=workflow.correction_options.ionosphere_files,
                troposphere_files=workflow.correction_options.troposphere_files,
                dem_file=workflow.correction_options.dem_file,
                static_layers_files=workflow.correction_options.geometry_files,
            ),
            static_ancillary_file_group=StaticAncillaryFileGroup(
                frame_to_burst_json=frame_to_burst_json,
                reference_date_database_json=reference_date_json,
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


def _override_parameters(
    algorithm_parameters: AlgorithmParameters, frame_id: int
) -> AlgorithmParameters:
    param_dict = algorithm_parameters.model_dump()
    # Get the "override" file for this set of parameters
    overrides_json = param_dict.pop("algorithm_parameters_overrides_json")

    # Load any overrides for this frame
    override_params = _parse_algorithm_overrides(overrides_json, frame_id)

    # Override the dict with the new options
    param_dict = _nested_update(param_dict, override_params)
    return AlgorithmParameters(**param_dict)


def _get_first_after_selected(
    input_dates: Sequence[datetime.datetime | datetime.date],
    selected_date: datetime.datetime | datetime.date,
) -> int:
    """Find the first index of `input_dates` which falls after `selected_date`."""
    for idx, d in enumerate(input_dates):
        if d >= selected_date:
            return idx
    else:
        return -1


def _compute_reference_dates(
    reference_datetimes: Iterable[datetime.datetime],
    cslc_file_list: Iterable[PathOrStr],
    compressed_slc_plan: CompressedSlcPlan,
) -> tuple[int, datetime.datetime | None]:
    # Get the dates of the base phase (works for either compressed, or regular cslc)
    # Use one burst ID as the template.
    burst_to_file_list = group_by_burst(cslc_file_list)
    burst_id = list(burst_to_file_list.keys())[0]
    # Extract this burst id's files
    cur_files = sort_files_by_date(burst_to_file_list[burst_id])[0]
    # Mark any files beginning with "compressed" as compressed
    is_compressed = ["compressed" in str(Path(f).stem).lower() for f in cur_files]
    input_dates = [get_dates(f)[0].date() for f in cur_files]
    num_ccslc = sum(is_compressed)

    # If we have set the compressed_slc_plan to be the last per ministack,
    # we want to make the shortest baseline interferograms.
    # So we should make the output index relative to the most recent compressed SLC
    # https://github.com/isce-framework/dolphin/blob/14ac66e49a8e8e66e9b74fc9eb4f0d232ab0924c/src/dolphin/stack.py#L488
    if compressed_slc_plan == CompressedSlcPlan.LAST_PER_MINISTACK:
        output_reference_idx = max(0, num_ccslc - 1)
        # No extra reference date: this would need dolphin , as currently (2026-02-10)
        # the `sequential` workflow uses the extra date to decide what
        # `new_compressed_slc_reference_idx` should be.
        return output_reference_idx, None

    output_reference_idx = 0
    extra_reference_date: datetime.datetime | None = None
    reference_dates = sorted({d.date() for d in reference_datetimes})

    for ref_date in reference_dates:
        # Find the nearest index that is greater than or equal to the reference date
        candidate_dates = [d for d in input_dates if d >= ref_date]
        if not candidate_dates:
            continue
        nearest_idx = _get_first_after_selected(input_dates, ref_date)

        if nearest_idx == 0:
            # We're only making a change if it's after the first date
            # (we're looking for mid-stack changes)
            continue
        elif is_compressed[nearest_idx]:
            # Update the output_reference_idx for compressed SLCs
            output_reference_idx = nearest_idx
            # But if it's a compressed SLC, it's not an "extra" reference date
            # This will move forward in time as we iterate over the reference dates
        else:
            # Set extra_reference_date for non-compressed SLCs
            inp_date = input_dates[nearest_idx]
            # Don't use this SLC if it's before the requested changeover; only after
            if inp_date >= ref_date:
                extra_reference_date = inp_date

    return output_reference_idx, extra_reference_date


def _parse_reference_date_json(
    reference_date_json: Path | str | None, frame_id: int | str
):
    reference_datetimes: list[datetime.datetime] = []
    if reference_date_json is not None:
        with open(reference_date_json) as f:
            reference_data = json.load(f)
            if "data" in reference_data:
                reference_date_strs = reference_data["data"].get(str(frame_id), [])
            else:
                reference_date_strs = reference_data.get(str(frame_id), [])
            reference_datetimes = [
                datetime.datetime.fromisoformat(s) for s in reference_date_strs
            ]
    else:
        reference_datetimes = []
    return reference_datetimes


def _parse_algorithm_overrides(
    override_file: Path | str | None, frame_id: int | str
) -> dict[str, Any]:
    """Find the frame-specific parameters to override for algorithm_parameters."""
    if override_file is not None:
        with open(override_file) as f:
            overrides = json.load(f)
            if "data" in overrides:
                return overrides["data"].get(str(frame_id), {})
            else:
                return overrides.get(str(frame_id), {})
    return {}


def _nested_update(base: dict, updates: dict):
    for k, v in updates.items():
        if isinstance(v, dict):
            base[k] = _nested_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


class StaticLayersDynamicAncillaryFileGroup(YamlModel):
    """A group of dynamic ancillary files."""

    geometry_files: List[Path] = Field(
        default_factory=list,
        alias="static_layers_files",
        description=(
            "Paths to the CSLC static_layer files (1 per burst) with line-of-sight"
            " unit vectors."
        ),
    )
    rtc_static_layers_files: List[Path] = Field(
        default_factory=list,
        description="Paths to the RTC layover shadow mask files (1 per burst).",
    )
    dem_file: Path = Field(
        ...,
        description="Path to the DEM file covering full frame.",
    )
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"required": ["geometry_files, rtc_static_layers_files"]},
    )


class StaticLayersRunConfig(RunConfig):
    """Run configuration for static layers SAS."""

    # Simplest work around to keep PGE interface same:
    # ignore all the extra fields and allow all
    model_config = ConfigDict(extra="allow")
    # BUT, we no longer need dynamic_ancillary_file_group.algorithm_parameters_file
    # So we shouldn't require it
    dynamic_ancillary_file_group: StaticLayersDynamicAncillaryFileGroup = Field(
        ...,
        alias="dynamic_ancillary_file_group",
    )
