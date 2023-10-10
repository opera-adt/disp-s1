# https://gist.github.com/gmgunter/1b864e17767aeb1055f82c1d14c721b2
# User Guide's way
set -e
set -x # echo on

readonly HELP='usage: ./test_delivery.sh [small|large] [forward|historical] [test_location]

Run the SAS workflow on a small or large dataset and compare the output against
a golden dataset.

positional arguments:
small|large     the size of the dataset to use
forward|historical  the mode to run the workflow in
test_location   the location to put the test data
'

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo "$HELP"
    exit 0
elif [[ "$#" -lt 2 ]]; then
    echo 'Illegal number of parameters' >&2
    echo "$HELP"
    exit 1
fi

test_location="${3:-/tmp/test_delivery}"
test_location=$(realpath $test_location)
mkdir -p $test_location

# Clone the source.
git clone git@github.com:opera-adt/disp-s1.git
cd disp-s1
# git checkout v0.1

# # Build the docker image.
BASE="cae-artifactory.jpl.nasa.gov:16003/gov/nasa/jpl/iems/sds/infrastructure/base/jplsds-oraclelinux:8.4.230101"
TAG=${TAG:-"$(whoami)/disp-s1:0.2"}
./docker/build-docker-image.sh --tag "$TAG" --base "$BASE"

# untar the test data.
# Pick the small or large one
# $ lsh *tar
# -rw-r--r-- 1 staniewi users 144G Oct  6 16:36 delivery_data_full.tar
# -rw-r--r-- 1 staniewi users 3.1G Oct  6 17:08 delivery_data_small.tar

if [ "$1" == "small" ]; then
    tar -xf /home/staniewi/dev/beta-delivery/delivery_data_small.tar -C $test_location
    cd $test_location/delivery_data_small
else
    tar -xf /home/staniewi/dev/beta-delivery/delivery_data_full.tar -C $test_location
    cd $test_location/delivery_data_full
fi

# Run the SAS workflow.
# Pick the "historical" or "forward"
# $ ls config_files/run*
# config_files/runconfig_forward.yaml  config_files/runconfig_historical.yaml
mode=${2:-forward}
if [ "$mode" == "forward" ]; then
    echo "Running forward mode"
elif [ "$mode" == "historical" ]; then
    echo "Running historical mode"
else
    echo "Invalid mode: $mode"
    exit 1
fi
cfg_file="config_files/runconfig_${mode}.yaml"

docker run --rm -u $(id -u):$(id -g) \
    -v $PWD:/work \
    $TAG disp-s1 $cfg_file

# Compare the output against a golden dataset.
docker run \
    --rm \
    -v $PWD:/work \
    $TAG \
    python /disp-s1/scripts/release/validate_product.py --golden golden_output/*nc --test output/*nc
