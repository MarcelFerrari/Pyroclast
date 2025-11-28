#!/bin/bash

PCD="/home/alisot2000/Documents/02_ETH/Bachelor_Thesis/Pyroclast"

# goto dir and init poetry
cd $PCD

# Check we have everything needed for running our stuff
poetry install --all-groups --all-extras
source $PCD/.venv/bin/activate
source $PCD/share/setup-env-alex.sh

cd $PCD/scripts/thesis/

numactl --physcpubind=0,2,4,6,8,10,12,14 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 16384 -Y 16384 \
    -c 1 2 4 8 \
    --samples 15

numactl --physcpubind=0,2,4,6,8,10,12,14 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 8192 -Y 8192 \
    -c 1 2 4 8 \
    --samples 15

numactl --physcpubind=0,2,4,6,8,10,12,14 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 4096 -Y 4096 \
    -c 1 2 4 8 \
    --samples 15

numactl --physcpubind=0,2,4,6,8,10,12,14 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 2048 -Y 2048 \
    -c 1 2 4 8 \
    --samples 15

numactl --physcpubind=0,2,4,6,8,10,12,14 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 1024 -Y 1024 \
    -c 1 2 4 8 \
    --samples 15

numactl --physcpubind=0,2,4,6,8,10,12,14 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 512 -Y 512 \
    -c 1 2 4 8 \
    --samples 15

numactl --physcpubind=0,2,4,6,8,10,12,14 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 256 -Y 256 \
    -c 1 2 4 8 \
    --samples 15

numactl --physcpubind=0,2,4,6,8,10,12,14 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 128 -Y 128 \
    -c 1 2 4 8 \
    --samples 15

numactl --physcpubind=0,2,4,6,8,10,12,14 python3 -u parallel_restriction.py \
    --scale 2.5 \
    -l 2 \
    -X 64 -Y 64 \
    -c 1 2 4 8 \
    --samples 15