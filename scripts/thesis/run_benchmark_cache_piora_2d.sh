#!/bin/bash

PCD="/opt/thesis/Pyroclast"

# goto dir and init poetry
cd $PCD

# Check we have everything needed for running our stuff
poetry install --all-groups --all-extras
source $PCD/.venv/bin/activate
source $PCD/share/setup-env-alex.sh

cd $PCD/src/benchmark

#  run bigger cache test.
# TODO number of CPUS from previous example
numactl --physcpubind=0-7 python3 -u runner.py -m jacobi_fuse_cache_unroll_jitter \
   --cpu 8 \
   --samples 3 \
   --unroll 4 \
   --cache_a 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 512 \
   --cache_b 16 32 48 64 80 96 112 128 \
   --test smoother \
   --iterations 32 \
   --jitter 3 \
   --dimension 16384
