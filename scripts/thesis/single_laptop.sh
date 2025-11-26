#!/bin/bash

PCD="/home/alisot2000/Documents/02_ETH/Bachelor_Thesis/Pyroclast"

# goto dir and init poetry
cd $PCD

# Check we have everything needed for running our stuff
poetry install --all-groups --all-extras
source $PCD/share/setup-env-alex.sh

cd $PCD/src/benchmark
env
#python3 -u runner.py -m jacobi_fuse_cache rb_gs_fuse_cache -c 1 2 4 8 -s 15 -u 4 -a 32 -b 32 -i 1024 -t smoother -P -j 3 -d 64 90 128 181 256 362 512
python3 -u runner.py -m jacobi_fuse_cache rb_gs_fuse_cache base_jacobi --cpu 1 2 4 8     --samples 15     --unroll 4     --cache_a 32     --cache_b 32     --iterations 1024     --test smoother     --print-table     --jitter 3     --dimension 64 90 128 181 256 362 512