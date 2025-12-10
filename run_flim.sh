#!/usr/bin/env bash

# Get the directory of this script (so it works no matter where you call it from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the Python from the local .venv to run FLim_plotter.py
"$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/FLim_plotter.py" "$@"