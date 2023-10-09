#!/usr/bin/bash
set -e
set -x # echo on

# Input full directory:
full_dir=delivery_data_full

# output new directory:
new_dir="benchmark_data"
mkdir -p $new_dir/input_slcs
mkdir -p $new_dir/config_files
mkdir -p $new_dir/logs

# Match 2 bursts total
pattern="*08890[56]_iw1*"

# For benchmarking, we only want the input SLCs.
# All other dynamic_ancillary_files won't be used, and we don't wanna skip the PS step here,
# so we don't need the PS files either.
mkdir -p $new_dir/input_slcs
# Copy input slcs matching the pattern, keep all dates
# find "${full_dir}/input_slcs/" -name "$pattern" -exec cp {} "$new_dir/input_slcs/" \;

base_cmd="dolphin config --keep-paths-relative --threads-per-worker 16 --single --strides 6 3 \
    --unwrap-method snaphu --ntiles 3 3 --downsample 2 2 \
    --slc-files input_slcs/*.h5 --subdataset /data/VV"

cd $new_dir

# Make one without GPU
outfile="config_files/dolphin_config_cpu.yaml"
logfile="logs/dolphin_cpu.log"
cmd1="$base_cmd --work-directory scratch_cpu --no-gpu -o $outfile --log-file $logfile"
bash -c "$cmd1"

# Make one with GPU
outfile="config_files/dolphin_config_gpu.yaml"
logfile="logs/dolphin_gpu.log"
cmd2="$base_cmd --work-directory scratch_gpu -o $outfile --log-file $logfile"
bash -c "$cmd2"
