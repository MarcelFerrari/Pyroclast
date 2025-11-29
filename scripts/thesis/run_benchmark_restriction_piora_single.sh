#!/bin/bash

# TODO customize
PCD="/opt/thesis/Pyroclast"
# goto dir and init poetry
cd $PCD

# Check we have everything needed for running our stuff
poetry install --all-groups --all-extras
source $PCD/.venv/bin/activate
source $PCD/share/setup-env-alex.sh

cd $PCD/scripts/thesis/

# Create a directory for the benchmark
dest_dir="$PCD/../restriction/$(date -Iminutes)"
mkdir "$dest_dir"


# Smaller problem scale
numactl --physcpubind=0 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 1024 -Y 1024 \
    -c 1 2 4 8 16 32 64 128 \
    --samples 15 \
    --output $dest_dir

numactl --physcpubind=0 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 256 -Y 256 \
    -c 1 2 4 8 16 32 64 128 \
    --samples 15 \
    --output $dest_dir

numactl --physcpubind=0 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 64 -Y 64 \
    -c 1 2 4 8 16 32 64 128 \
    --samples 15 \
    --output $dest_dir