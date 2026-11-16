import subprocess
from pathlib import Path

from disp_s1.enums import ProcessingMode
from disp_s1.pge_runconfig import AlgorithmParameters, RunConfig

# Forward mode only: keep the compressed SLC's reference epoch as the output
# reference (dolphin's `interferogram_network.compressed_reference_anchor`).
# `convert_config.py` turns this into `forward_redo_reference: false` in the
# algorithm parameters file; the anchor itself is not a runconfig field.
# False keeps the default (re-reference to the second-to-last date).
COMPRESSED_REFERENCE_ANCHOR = False

# `AlgorithmParameters` fields which aren't part of the dolphin config, so they
# can't be set through `dolphin config`. Applied to the algorithm parameters file
# after `convert_config.py` writes it (values from the 20261008 files).
EXTRA_ALGORITHM_PARAMETERS = {
    "forward_mode_network_size": {
        ProcessingMode.FORWARD: 4,
        ProcessingMode.HISTORICAL: 3,
    },
    "recommended_temporal_coherence_threshold": 0.7,
    "recommended_similarity_threshold": 0.6,
    "recommended_use_conncomp": False,
    "spatial_wavelength_cutoff": 30000.0,
    "browse_image_vmin_vmax": (-0.05, 0.05),
    "num_parallel_products": 4,
}

# Values from the official v0.5.16 delivery runconfigs.
PRODUCT_VERSION = "1.1"
FRAME_TO_BURST_JSON = "opera-s1-disp-0.15.0-frame-to-burst.json.zip"
ALGORITHM_PARAMETERS_OVERRIDES_JSON = (
    "opera-disp-s1-algorithm-parameters-overrides-2025-09-17.json"
)
REFERENCE_DATE_JSON = "opera-disp-s1-reference-dates-2026-01-15.json"

# Input files for each mode, from the official v0.5.16 delivery runconfigs
# (relative to `input_slcs/` and `dynamic_ancillary_files/ionosphere_files/`).
INPUT_FILES = {
    ProcessingMode.FORWARD: {
        "slcs": [
            "compressed_slcs/OPERA_L2_COMPRESSED-CSLC-S1_F07091_T027-056725-IW1_20170524T000000Z_20160716T000000Z_20170524T000000Z_20250412T025708Z_VV_v1.0.h5",  # noqa: E501
            "compressed_slcs/OPERA_L2_COMPRESSED-CSLC-S1_F07091_T027-056726-IW1_20170524T000000Z_20160716T000000Z_20170524T000000Z_20250412T025708Z_VV_v1.0.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170325T132750Z_20240625T163704Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170325T132753Z_20240625T163704Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170406T132750Z_20240625T225442Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170406T132753Z_20240625T225442Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170418T132751Z_20240626T003055Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170418T132754Z_20240626T003055Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170430T132752Z_20240626T015700Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170430T132754Z_20240626T015700Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170605T132754Z_20240626T090220Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170605T132756Z_20240626T090220Z_S1A_VV_v1.1.h5",  # noqa: E501
        ],
        "ionosphere": [
            "jplg0840.17i",
            "jplg0960.17i",
            "jplg1080.17i",
            "jplg1200.17i",
            "jplg1440.17i",
            "jplg1560.17i",
        ],
    },
    ProcessingMode.HISTORICAL: {
        "slcs": [
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170217T132750Z_20240625T060850Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170217T132752Z_20240625T060850Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170301T132749Z_20240625T090210Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170301T132752Z_20240625T090210Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170313T132750Z_20240625T121542Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170313T132753Z_20240625T121542Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170325T132750Z_20240625T163704Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170325T132753Z_20240625T163704Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170406T132750Z_20240625T225442Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170406T132753Z_20240625T225442Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170418T132751Z_20240626T003055Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170418T132754Z_20240626T003055Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056725-IW1_20170430T132752Z_20240626T015700Z_S1A_VV_v1.1.h5",  # noqa: E501
            "OPERA_L2_CSLC-S1_T027-056726-IW1_20170430T132754Z_20240626T015700Z_S1A_VV_v1.1.h5",  # noqa: E501
        ],
        "ionosphere": [
            "jplg0480.17i",
            "jplg0600.17i",
            "jplg0720.17i",
            "jplg0840.17i",
            "jplg0960.17i",
            "jplg1080.17i",
            "jplg1200.17i",
        ],
    },
}

# Note on ionosphere/troposphere files:
# The ionosphere file download is working for a list of SLC files.
# see `ionosphere.download_ionex_for_slcs`.
# The troposphere file download is missing. Dummy files were created.
# for d in `ls input_slcs/t042_088905_iw1* | awk -F'_' '{print $5}' | cut -d'.' -f1`; do
#    touch dynamic_ancillary_files/troposphere_files/ERA5_N30_N40_W120_W110_${d}_14.grb;
# done


def setup_delivery(cfg_dir: Path, mode: ProcessingMode):
    """Set up the dolphin config file for the delivery for one mode."""
    cfg_dir.mkdir(exist_ok=True)
    outfile = f"{cfg_dir}/dolphin_config_{mode.value}.yaml"
    slc_files = " ".join(f"./input_slcs/{f}" for f in INPUT_FILES[mode]["slcs"])
    anchor_flag = ""
    if mode == ProcessingMode.FORWARD:
        anchor_flag = (
            " --interferogram-network.compressed-reference-anchor"
            if COMPRESSED_REFERENCE_ANCHOR
            else " --interferogram-network.no-compressed-reference-anchor"
        )
    cmd = (
        "dolphin config "
        f" --keep-paths-relative --work-directory scratch/{mode.value}"
        " --sx 6 --sy 3"
        # Inputs:
        f" --slc-files {slc_files} --subdataset /data/VV"
        # Phase linking stuff
        # (dolphin dropped the old flat `--single` toggle: the single-update vs.
        # sequential distinction is now handled automatically from the input
        # stack, not a CLI flag -- see CHANGELOG's "Removed `WorkflowName` for
        # separating `stack` vs `single`".)
        # Values below match the 20261008 algorithm parameters files.
        " --ms 100 --phase-linking.max-num-compressed 100"
        " --phase-linking.output-reference-idx 0"
        " --hwx 16 --hwy 8"
        " --phase-linking.compressed-slc-plan LAST_PER_MINISTACK"
        " --ps-options.amp-dispersion-threshold 0.2"
        # Dynamic ancillary files #
        ###########################
        # TODO # seasonal coherence averages
        # Ionosphere/geometry files are no longer part of the dolphin config;
        # they're passed to `convert_config.py` below instead.
        " --mask-file ./dynamic_ancillary_files/water_mask.vrt"
        #
        # Unwrapping stuff
        # (dropped the old flat `--downsample 5 5`: that only maps to
        # `unwrap-options.tophu-options.downsample-factor` now, which doesn't
        # apply to the `snaphu` method used here.)
        " --interferogram-network.max-bandwidth 4"
        # Network stuff (forward mode only)
        f"{anchor_flag}"
        " --unwrap-options.unwrap-method SNAPHU"
        " --unwrap-options.snaphu-options.ntiles 5 5"
        " --unwrap-options.snaphu-options.tile-overlap 400 400"
        " --unwrap-options.snaphu-options.n-parallel-tiles 7"
        " --unwrap-options.snaphu-options.single-tile-reoptimize"
        " --unwrap-options.run-interpolation"
        " --unwrap-options.preprocess-options.max-radius 150"
        " --unwrap-options.preprocess-options.interpolation-cor-threshold 0.001"
        " --unwrap-options.preprocess-options.interpolation-similarity-threshold 0.4"
        # SPURT options (only used if the unwrap method is switched to SPURT)
        " --unwrap-options.spurt-options.tiler-settings.max-tiles 25"
        " --unwrap-options.spurt-options.tiler-settings.target-points-per-tile 1400000"
        " --unwrap-options.spurt-options.tiler-settings.dilation-factor 0.04"
        " --unwrap-options.spurt-options.solver-settings.t-worker-count 16"
        " --unwrap-options.spurt-options.solver-settings.s-worker-count 3"
        " --unwrap-options.spurt-options.solver-settings.t-cost-type distance"
        " --unwrap-options.spurt-options.merger-settings.num-parallel-ifgs 21"
        # Time series
        " --timeseries-options.no-run-velocity"
        " --timeseries-options.correlation-threshold 0.0"
        " --output-options.no-add-overviews"
        # Worker stuff
        " --worker-settings.gpu-enabled"
        " --worker-settings.threads-per-worker 2 --n-parallel-bursts 9"
        " --worker-settings.block-shape 512 512"
        " --unwrap-options.n-parallel-jobs 4"
        f" --log-file scratch/{mode.value}/log_sas.log"
        f" --outfile {outfile}"
    )
    print(cmd)
    subprocess.run(cmd, shell=True, check=False)
    return outfile


def update_algorithm_parameters(algo_file: Path, mode: ProcessingMode):
    """Set the disp-s1-specific fields of the algorithm parameters file."""
    params = AlgorithmParameters.from_yaml(algo_file)
    for name, value in EXTRA_ALGORITHM_PARAMETERS.items():
        if isinstance(value, dict):
            value = value[mode]
        setattr(params, name, value)
    params.to_yaml(algo_file)


def update_runconfig(runconfig_file: Path):
    """Set the runconfig fields which `convert_config.py` has no option for."""
    rc = RunConfig.from_yaml(runconfig_file)
    rc.product_path_group.product_version = PRODUCT_VERSION
    rc.static_ancillary_file_group.algorithm_parameters_overrides_json = Path(
        ALGORITHM_PARAMETERS_OVERRIDES_JSON
    )
    rc.run_input_prechecks = True
    rc.to_yaml(runconfig_file)


if __name__ == "__main__":
    cfg_dir = Path("config_files")
    static_ancillary_dir = Path("static_ancillary_files")
    # Track 27 (frame 7091), from the official v0.5.16 delivery:
    frame_id = 7091
    reference_json = REFERENCE_DATE_JSON
    # Creates one file for the forward mode and one for the historical mode.
    for mode in ProcessingMode:
        output_directory = Path(f"output/{mode.value}")
        # TODO: adjust the number of
        # ionosphere files
        # troposphere files
        dolphin_cfg_file = setup_delivery(cfg_dir=cfg_dir, mode=mode)
        # Run the "convert_config.py" script in the same directory
        # as this script.
        this_dir = Path(__file__).parent
        convert_config = this_dir / "convert_config.py"
        iono_files = " ".join(
            f"./dynamic_ancillary_files/ionosphere_files/{f}"
            for f in INPUT_FILES[mode]["ionosphere"]
        )
        arg_string = (
            f" --frame-id {frame_id} "
            f" --frame-to-burst-json {static_ancillary_dir}/{FRAME_TO_BURST_JSON}"
            f" --reference_date_database_json {static_ancillary_dir}/{reference_json}"
            f" --output-directory {output_directory}"
            f" --processing-mode {mode.value} --save-compressed-slc -o"
            f" {cfg_dir}/runconfig_{mode.value}.yaml  -a"
            f" {cfg_dir}/algorithm_parameters_{mode.value}.yaml"
            f" --ionosphere-files {iono_files}"
            " --geometry-files ./dynamic_ancillary_files/static_layers/*"
        )
        cmd = f"python {convert_config} {dolphin_cfg_file} {arg_string}"
        print(cmd)
        subprocess.run(cmd, shell=True, check=False)
        update_algorithm_parameters(
            cfg_dir / f"algorithm_parameters_{mode.value}.yaml", mode
        )
        update_runconfig(cfg_dir / f"runconfig_{mode.value}.yaml")
        # Remove the `dolphin` yamls
        for f in cfg_dir.glob("dolphin_config*.yaml"):
            f.unlink()
