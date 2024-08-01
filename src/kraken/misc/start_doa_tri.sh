#!/bin/bash

#source /home/krakenrf/miniforge3/etc/profile.d/conda.sh #<- required for systemd auto startup (comment out eval and use source instead)
eval "$(conda shell.bash hook)"
conda activate kraken

PYTHON_EXECUTABLE="$(which python3)"

sudo "$PYTHON_EXECUTABLE" config_tri.py

# Clear pycache before starting if the -c flag is given
while getopts c flag
do
    case "${flag}" in
        c) sudo py3clean . ;;
    esac
done

./stop_doa.sh

cd ../heimdall_daq_fw/Firmware
sudo env "PATH=$PATH" ./daq_start_sm.sh
cd ../../../kraken/misc

sudo "$PYTHON_EXECUTABLE" kraken_heimdall_tri.py
