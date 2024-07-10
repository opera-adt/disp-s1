import subprocess
from pathlib import Path

from disp_s1.enums import ProcessingMode

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
    single_flag = "--single" if mode == ProcessingMode.FORWARD else ""
    outfile = f"{cfg_dir}/dolphin_config_{mode.value}.yaml"
    cmd = (
        "dolphin config "
        f" --keep-paths-relative --work scratch/{mode.value} --strides 6 3"
        # Inputs:
        " --slc-files ./input_slcs/*h5 --subdataset /data/VV"
        # Phase linking stuff
        f" --ministack-size 20 {single_flag}"
        # Dynamic ancillary files #
        ###########################
        # TODO # seasonal coherence averages
        # Troposphere files:
        " --troposphere-files ./dynamic_ancillary_files/troposphere_files/*"
        # Ionosphere files:
        " --ionosphere-files ./dynamic_ancillary_files/ionosphere_files/*"
        # DEM files:
        " --dem ./dynamic_ancillary_files/dem.tif"
        # Geometry files/static layers
        " --geometry-files ./dynamic_ancillary_files/static_layers/*"
        " --mask-file ./dynamic_ancillary_files/watermask.tif"
        #
        # Unwrapping stuff
        " --unwrap-method snaphu --ntiles 5 5 --downsample 5 5"
        # Worker stuff
        " --threads-per-worker 16 --n-parallel-bursts 2 --n-parallel-unwrap 2 "
        f" --log-file scratch/{mode.value}/log_sas.log"
        f" -o {outfile}"
    )
    print(cmd)
    subprocess.run(cmd, shell=True, check=False)
    return outfile


if __name__ == "__main__":
    cfg_dir = Path("config_files")
    # California, track 42, bay area:
    frame_id = 11114
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
        arg_string = (
            f" --frame-id {frame_id} "
            f" --output-directory {output_directory}"
            f" --processing-mode {mode.value} --save-compressed-slc -o"
            f" {cfg_dir}/runconfig_{mode.value}.yaml  -a"
            f" {cfg_dir}/algorithm_parameters_{mode.value}.yaml"
        )
        cmd = f"python {convert_config} {dolphin_cfg_file} {arg_string}"
        print(cmd)
        subprocess.run(cmd, shell=True, check=False)
        # Remove the `dolphin` yamls
        for f in cfg_dir.glob("dolphin_config*.yaml"):
            f.unlink()
