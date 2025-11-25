#!/bin/bash

#
# Pyroclast: Scalable Geophysics Models
# https://github.com/MarcelFerrari/Pyroclast

# File: share/setup-env.sh
# Description: File contains setting of environemnt variables to get
#              the shell env ready to run PyroFile contains setting of environemnt variables to get
#                           the shell env ready to run Pyroclast
# Author: Marcel Ferrari, Alexander Sotoudeh
# Copyright (c) 2024 Marcel Ferrari.

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Get the current directory where the script is located
SCRIPT_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BIN_PATH=$(realpath "$SCRIPT_PATH/../bin/")
export PYTHONPATH="$SCRIPT_PATH/../src/:$PYTHONPATH"

# Check if the file is executable, if not, make it executable
if [ ! -x "$BIN_PATH/pyroclast" ]; then
    chmod +x "$BIN_PATH/pyroclast"
fi

# Add the directory containing agi to PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    export PATH="$PATH:$BIN_PATH"
fi

# Source poetry venv
source "${SCRIPTPATH}/../.venv/bin/activate"

# add python path
export OLD_PYTHONPATH="${PYTHONPATH}"
export PYTHONPATH="$( realpath "${SCRIPTPATH}/../src" ):${PYTHONPATH}"

RUNNER_PATH="$( realpath "${SCRIPTPATH}/../src/benchmark/runner.py" )"
PRINTER_PATH="$( realpath "${SCRIPTPATH}/../src/benchmark/printer.py" )"

alias runner="python3 ${RUNNER_PATH}"
alias printer="python3 ${PRINTER_PATH}"

echo "Setup complete."
