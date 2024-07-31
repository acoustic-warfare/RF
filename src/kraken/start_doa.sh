#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate kraken

PYTHON_EXECUTABLE="$(which python)"

"$PYTHON_EXECUTABLE" config.py

# Clear pycache before starting if the -c flag is given
while getopts c flag
do
    case "${flag}" in
        c) sudo py3clean . ;;
    esac
done

./stop_doa.sh

cd heimdall_daq_fw/Firmware
env "PATH=$PATH" ./daq_start_sm.sh
cd ../../../kraken

"$PYTHON_EXECUTABLE" kraken_heimdall.py
