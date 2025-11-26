#!/bin/bash

PCD="/opt/thesis/Pyroclast"

# goto dir and init poetry
cd $PCD

# Check we have everything needed for running our stuff
poetry install --all-groups --all-extras
source $PCD/.venv/bin/activate
source $PCD/share/setup-env-alex.sh

cd $PCD/src/benchmark

# Run small cpu benchmarks
# python3 -u runner.py -D
python3 -u runner.py -m base_jacobi base_rb_gs jacobi_fuse jacobi_fuse_cache jacobi_fuse_cache_jitter jacobi_fuse_cache_unroll_jitter rb_gs_fuse rb_gs_fuse_cache rb_gs_fuse_cache_jitter \
    --cpu 1 2 4 8 16 32 \
    --samples 15 \
    --unroll 4 \
    --cache_a 32 \
    --cache_b 32 \
    --iterations 1024 \
    --test smoother \
    --print-table \
    --jitter 3 \
    --dimension 64 90 128 181 256 362 512

# Medium benchmark
python3 -u runner.py -m jacobi_fuse_cache jacobi_fuse_cache_jitter jacobi_fuse_cache_unroll_jitter rb_gs_fuse_cache rb_gs_fuse_cache_jitter \
    --cpu 4 8 16 32 64 96 128 \
    --samples 15 \
    --unroll 4 \
    --cache_a 32 \
    --cache_b 32 \
    --iterations 128 \
    --test smoother \
    --print-table \
    --jitter 3 \
    --dimension 512 724 1024 1448 2048 2896 4096

# GPU small
python3 -u runner.py -m gpu_jacobi gpu_rb_gs \
    --cpu 8 \
    --samples 15 \
    --unroll 4 \
    --cache_a 32 \
    --cache_b 32 \
    --iterations 1024 \
    --test smoother \
    --print-table \
    --jitter 3 \
    --dimension 64 90 128 181 256 362 512 724 1024 1448 2048

# GPU big
python3 -u runner.py -m gpu_jacobi gpu_rb_gs \
    --cpu 8 \
    --samples 15 \
    --unroll 4 \
    --cache_a 32 \
    --cache_b 32 \
    --iterations 128 \
    --test smoother \
    --print-table \
    --jitter 3 \
    --dimension  2048 2896 4096
