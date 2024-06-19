#
# RF environment setup
#
echo '
\033[34m ⠀⠀⣠⣤⣤⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⣿⣿⡿⣿⣿⣿⣴⣤⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠠⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⣯⣿⣿⢿⣿⢿⣿⣿⣿⣾⣴⣠⢀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⠶⣖⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⢹⣇⣿⣾⣿⣏⡾⣽⣻⣿⣯⡟⣖⣮⢄⠀⢀⠀⣄⠶⢿⡿⣟⣻⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⢸⣸⡣⢿⡾⣼⣿⣿⣿⣷⣯⣽⣟⡾⣍⡻⢢⣍⡒⠊⠁⠐⡿⠛⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⠈⠱⣝⡴⣿⢿⣿⣿⣫⣾⣿⣟⣞⣿⡵⣟⣿⢯⡹⣪⢄⣼⠃⠀⣟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⠀⠀⢫⣽⣼⡹⣶⣟⣿⣿⣿⣿⣻⣯⣿⣟⣻⢾⡵⣣⣿⢯⡺⣂⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⠀⠀⠈⢧⢺⢇⢶⠾⢿⣾⣝⡿⣯⣾⣿⢯⣯⣿⣵⡿⣽⢧⡳⣍⣷⢧⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⠀⠀⠀⠈⠿⣸⣻⣷⣿⡟⢯⣿⣺⣝⣿⣯⣾⣿⣷⣽⡯⣷⣿⣿⣻⣿⣾⣿⣦⣄⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⠀⠀⠀⠀⠈⢿⢿⣯⣱⣿⣿⣿⡟⢻⣿⣾⣟⣿⣿⣽⣿⣿⣿⢳⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⠀⠀⠀⠀⠀⠈⡧⡫⢼⣿⣾⣿⣿⡟⣿⣝⡿⣿⣫⣿⣿⣿⣯⣿⣿⢿⣟⣿⣿⣿⣿⣿⣷⣄⠀⠀\033[0m
\033[34m ⠀⠀⠀⠀⠀⠀⠀⡞⡁⡨⠇⣹⠻⢿⣿⣿⣻⣟⣿⣿⡟⣻⣹⣿⣿⣶⣿⣿⡿⣫⣿⣿⣿⣿⣿⣿⣧⡀\033[0m
\033[34m ⠀⠀⠀⠀⠀⠀⢸⢢⢻⠇⣳⣹⣾⢪⢙⡫⢝⣿⣿⣿⣿⣿⢎⣃⡿⣹⠾⡽⢯⣟⣶⢿⣻⣽⣟⣿⣿⡧\033[0m
\033[34m ⠀⠀⠀⠀⠀⠀⢘⡿⣟⢰⣷⣯⣷⣿⡆⠉⠓⡳⠭⠻⠛⠷⠦⠓⠓⣛⣟⡛⠿⢎⢓⣦⡶⠽⠟⠛⠉⠁\033[0m
\033[34m ⠀⠀⠀⠀⠀⠀⡜⢻⢷⣻⣿⣿⣿⣿⣧⡴⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⠀⠀⠀⣰⣛⣟⣀⡿⣿⢿⣿⣿⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⠀⠀⠀⠾⢿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣿⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⠀⠀⠀⠀⠀⢀⣻⣿⣿⣿⣻⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⠀⠀⠀⢀⡠⠊⢱⢡⢏⣿⣿⣿⣿⡿⣿⡕⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠀⢀⡤⠒⠁⠀⠀⠠⣍⣞⣳⣿⣿⣿⣿⣿⣇⡀⠑⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
\033[34m ⠋⠉⠘⠙⠋⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\033[0m
'

#
# mlocate, git
#
if [ ! "$(command -v locate)" ]; then
    echo "\033[1;41mERROR: Please install 'mlocate'\033[0m"
    return 0
fi

#
# Python
#
# 'venv' setup and activation
VENV_NAME=".venv"
if [ ! -d "$VENV_NAME" ]; then # will just do this if there is no .venv
    echo -e "Python environment setup"
    python3 -m venv $VENV_NAME
    $VENV_NAME/bin/python3 -m pip install -q setuptools wheel
    $VENV_NAME/bin/python3 -m pip install -q -r requirements.txt
fi
# TODO: Add explicit python min version and check
source $VENV_NAME/bin/activate
python3 --version

#
# UHD
#
UHD_VERSION_VERSION=$(uhd_config_info --version 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "$UHD_VERSION_VERSION"
else
    echo "\033[0;31mUHD not found \033[0m"
    echo "Install using: sudo apt install libuhd-dev uhd-host python3-uhd"
fi

#
# SOAPY
#
SOAPY_INFO=$(SoapySDRUtil --info 2>/dev/null)
if [ $? -eq 0 ]; then
    VERSION=$(echo "$SOAPY_INFO" | grep -oP 'Lib Version: \K[^ ]+')
    echo "SoapySDR v$VERSION"
else
    echo "\033[0;31mSoapySRD not found \033[0m"
    echo "Install SoapySRD"
fi

#
# Useful aliases
#
#alias run_test='zsh $(git rev-parse --show-toplevel)/pl/scripts/other/run_test.sh'

#echo -e '
#Welcome to RF!
#\033[4mCommands:\033[0m
#\033[1m  run_test --help         \033[0m
#\033[1m  build --help            \033[0m
#\033[1m  clean_vivado --help     \033[0m
#'

#
# Cleanup
#
unset GHDL_PATH GHDL_REQUIRED_VERSION GHDL_VERSION XILINX_PATH VENV_NAME VIVADO_VERSION
