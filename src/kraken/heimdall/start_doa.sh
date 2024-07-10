#!/bin/bash

#source /home/krakenrf/miniforge3/etc/profile.d/conda.sh #<- required for systemd auto startup (comment out eval and use source instead)
eval "$(conda shell.bash hook)"
conda activate kraken

# Clear pycache before starting if the -c flag is given
while getopts c flag
do
    case "${flag}" in
        c) sudo py3clean . ;;
    esac
done

./stop_doa.sh

cd heimdall_daq_fw/Firmware
#sudo ./daq_synthetic_start.sh
sudo env "PATH=$PATH" ./daq_start_sm.sh
cd ../../../heimdall
sudo python3 kraken_heimdall.py
