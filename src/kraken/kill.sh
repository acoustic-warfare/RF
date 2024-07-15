#/bin/sh!
SYSTEM_OS="$(uname -s)"
PYTHON_EXECUTABLE="$(which python3)"

if [[ "$SYSTEM_OS" == "Darwin" ]];
then
    KILL_SIGNAL=9
else
    KILL_SIGNAL=64
fi

PYTHON_PIDS=$(ps ax | grep "python3 .*kraken_heimdall.py" | grep -v grep | awk '{print $1}')
if [ -n "$PYTHON_PIDS" ]; then
    sudo kill -${KILL_SIGNAL} $PYTHON_PIDS
else
    echo "No Python processes found for kraken_heimdall.py"
fi