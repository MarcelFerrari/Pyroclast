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


numactl --physcpubind=0-127 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 16384 -Y 16384 \
    -c 1 2 4 8 16 32 64 128
    --samples 15 \
    --output $dest_dir

numactl --physcpubind=0-127 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 8192 -Y 8192 \
    -c 1 2 4 8 16 32 64 128
    --samples 15 \
    --output $dest_dir

numactl --physcpubind=0-127 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 4096 -Y 4096 \
    -c 1 2 4 8 16 32 64 128
    --samples 15 \
    --output $dest_dir

numactl --physcpubind=0-127 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 2048 -Y 2048 \
    -c 1 2 4 8 16 32 64 128
    --samples 15 \
    --output $dest_dir

numactl --physcpubind=0-127 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 1024 -Y 1024 \
    -c 1 2 4 8 16 32 64 128
    --samples 15 \
    --output $dest_dir

numactl --physcpubind=0-127 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 512 -Y 512 \
    -c 1 2 4 8 16 32 64 128
    --samples 15 \
    --output $dest_dir

numactl --physcpubind=0-127 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 256 -Y 256 \
    -c 1 2 4 8 16 32 64 128
    --samples 15 \
    --output $dest_dir

numactl --physcpubind=0-127 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 128 -Y 128 \
    -c 1 2 4 8 16 32 64 128
    --samples 15 \
    --output $dest_dir

numactl --physcpubind=0-127 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 64 -Y 64 \
    -c 1 2 4 8 16 32 64 128
    --samples 15 \
    --output $dest_dir