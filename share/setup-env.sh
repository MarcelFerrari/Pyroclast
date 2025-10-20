#!/bin/bash

# Pyroclast: Scalable Geophysics Models
# https://github.com/MarcelFerrari/Pyroclast
#
# File: share/setup-env.sh
# Description: Convenience script to add Pyroclast binaries to the PATH.
#
# Author: Marcel Ferrari
# Copyright (c) 2025 Marcel Ferrari.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Get the current directory where the script is located
SCRIPT_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BIN_PATH=$(realpath "$SCRIPT_PATH/../bin/")

# Check if the file is executable, if not, make it executable
if [ ! -x "$BIN_PATH/pyroclast" ]; then
    chmod +x "$BIN_PATH/pyroclast"
fi

# Add the directory containing agi to PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    export PATH="$PATH:$BIN_PATH"
fi

echo "Setup complete."
