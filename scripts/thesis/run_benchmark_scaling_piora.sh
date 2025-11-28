#!/bin/bash

PCD="/opt/thesis/Pyroclast"

# goto dir and init poetry
cd $PCD

# Check we have everything needed for running our stuff
poetry install --all-groups --all-extras
source $PCD/share/setup-env-alex.sh

cd $PCD/src/benchmark

# Run small cpu benchmarks
# python3 -u runner.py -D
numcactl --physcpubind=0-127 python3 -u runner.py -m base_jacobi base_rb_gs jacobi_fuse jacobi_fuse_cache jacobi_fuse_cache_jitter jacobi_fuse_cache_unroll_jitter rb_gs_fuse rb_gs_fuse_cache rb_gs_fuse_cache_jitter \
    --cpu 1 2 4 8 16 32 \
    --samples 15 \
    --unroll 4 \
    --cache_a 32 \
    --cache_b 32 \
    --iterations 1024 \
    --test smoother \
    --jitter 3 \
    --dimension 64 90 128

# medium benchmark
numcactl --physcpubind=0-127 python3 -u runner.py -m base_jacobi base_rb_gs jacobi_fuse jacobi_fuse_cache jacobi_fuse_cache_jitter jacobi_fuse_cache_unroll_jitter rb_gs_fuse rb_gs_fuse_cache rb_gs_fuse_cache_jitter \
    --cpu 1 2 4 8 16 32 64  \
    --samples 15 \
    --unroll 4 \
    --cache_a 32 \
    --cache_b 32 \
    --iterations 256 \
    --test smoother \
    --jitter 3 \
    --dimension 181 256 362 512

# big benchmark
numcactl --physcpubind=0-127 python3 -u runner.py -m jacobi_fuse_cache jacobi_fuse_cache_unroll_jitter rb_gs_fuse_cache \
    --cpu 4 8 16 32 64 96 128 \
    --samples 15 \
    --unroll 4 \
    --cache_a 32 \
    --cache_b 32 \
    --iterations 128 \
    --test smoother \
    --jitter 3 \
    --dimension 724 1024 1448 2048 2896 4096

# GPU small
numcactl --physcpubind=0-127 python3 -u runner.py -m gpu_jacobi gpu_rb_gs \
    --cpu 8 \
    --samples 15 \
    --unroll 4 \
    --cache_a 32 \
    --cache_b 32 \
    --iterations 1024 \
    --test smoother \
    --jitter 3 \
    --dimension 64 90 128

# gpu medium
numcactl --physcpubind=0-127 python3 -u runner.py -m gpu_jacobi gpu_rb_gs \
    --cpu 8 \
    --samples 15 \
    --unroll 4 \
    --cache_a 32 \
    --cache_b 32 \
    --iterations 256 \
    --test smoother \
    --jitter 3 \
    --dimension 181 256 362 512

# GPU big
numcactl --physcpubind=0-127 python3 -u runner.py -m gpu_jacobi gpu_rb_gs \
    --cpu 8 \
    --samples 15 \
    --unroll 4 \
    --cache_a 32 \
    --cache_b 32 \
    --iterations 128 \
    --test smoother \
    --jitter 3 \
    --dimension 724 1024 1448 2048 2896 4096
