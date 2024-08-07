#!/bin/bash

ldconfig

eval "$(conda shell.bash hook)"
conda activate pluto

PYTHON_EXECUTABLE="$(which python3)"

#"$PYTHON_EXECUTABLE" config.py

# Clear pycache before starting if the -c flag is given
while getopts c flag
do
    case "${flag}" in
        c) sudo py3clean . ;;
    esac
done
"$PYTHON_EXECUTABLE" main.py

