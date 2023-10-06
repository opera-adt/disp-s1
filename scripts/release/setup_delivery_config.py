import subprocess
from pathlib import Path

from disp_s1.enums import ProcessingMode


def setup_delivery(out_dir: Path, mode: ProcessingMode):
    """Set up the dolphin config file for the delivery for one mode."""
    out_dir.mkdir(exist_ok=True)
    single_flag = "--single" if mode == ProcessingMode.FORWARD else ""
    outfile = f"{out_dir}/dolphin_config_{mode.value}.yaml"
    cmd = (
        "dolphin config "
        " --keep-paths-relative --work scratch --strides 6 3"
        # Inputs:
        " --slc-files ./input_slcs/*h5 --subdataset /data/VV"
        # Phase linking stuff
        f" --ministack-size 20 {single_flag}"
        # Dynamic ancillary files #
        ###########################
        # PS mean/dispersion files:
        " --amplitude-mean-files ./dynamic_ancillary_files/ps_files/*mean*"
        " --amplitude-dispersion-files ./dynamic_ancillary_files/ps_files/*dispersion*"
        # TODO # seasonal coherence averages
        # "--seasonal-coherence-files dynamic_ancillary_files/seasonal_coherence_files/* "
        # Troposphere files:
        " --troposphere-files ./dynamic_ancillary_files/troposphere_files/*"
        # Ionosphere files:
        " --ionosphere-files ./dynamic_ancillary_files/ionosphere_files/*"
        # DEM files:
        " --dem ./dynamic_ancillary_files/dem.tif"
        # Geometry files/static layers
        " --geometry-files ./dynamic_ancillary_files/static_layers/*"
        " --mask-file ./dynamic_ancillary_files/watermask.flg"
        #
        # Unwrapping stuff
        " --unwrap-method snaphu --ntiles 5 5 --downsample 5 5"
        # Worker stuff
        " --threads-per-worker 16 --n-parallel-bursts 2 --n-parallel-unwrap 3 --no-gpu"
        f" -o {outfile}"
    )
    print(cmd)
    subprocess.run(cmd, shell=True)
    return outfile


if __name__ == "__main__":
    out_dir = Path("config_files")
    # Creates one file for the forward mode and one for the historical mode.
    for mode in ProcessingMode:
        # TODO: adjust the number of
        # ionosphere files
        # troposphere files
        dolphin_cfg_file = setup_delivery(out_dir=out_dir, mode=mode)
        # Run the "convert_config.py" script in the same directory
        # as this script.
        this_dir = Path(__file__).parent
        convert_config = this_dir / "convert_config.py"
        arg_string = (
            f" --frame-id 11114 --processing-mode {mode.value} -o"
            f" {out_dir}/runconfig_{mode.value}.yaml -a"
            f" {out_dir}/algorithm_parameters_{mode.value}.yaml"
        )
        cmd = f"python {convert_config} {dolphin_cfg_file} {arg_string}"
        print(cmd)
        subprocess.run(cmd, shell=True)
