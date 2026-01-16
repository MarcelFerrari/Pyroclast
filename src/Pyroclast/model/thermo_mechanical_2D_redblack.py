"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: thermo_mechanical_2D_redblack.py
Description: GPU-accelerated Red-Black Gauss-Seidel thermal solver using CUDA.
             Synchronization between red/black phases is achieved via kernel launch
             boundaries and CUDA stream synchronization.

Author: Marcel Ferrari, Andreas Rohwedder
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
import math

from Pyroclast.model.stokes_2D_mg import IncompressibleStokes2DMG
from Pyroclast.logging import get_logger

logger = get_logger(__name__)

# GPU imports
try:
    import cupy as cp
    from numba import cuda, types
    from Pyroclast.gpu_utils import launch_2D, get_numba_stream
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPU libraries not available. Falling back to CPU.")


# ============================================
# CUDA Kernels for Red-Black Gauss-Seidel
# ============================================

if GPU_AVAILABLE:
    @cuda.jit(fastmath=True, cache=True)
    def redblack_gauss_seidel_sweep_gpu(
        T,        # (ny1, nx1) solution array (updated in-place)
        T0,       # (ny1, nx1) previous timestep solution
        kvx,      # (ny1, nx1) conductivity at vertical faces
        kvy,      # (ny1, nx1) conductivity at horizontal faces
        rhocp,    # (ny1, nx1)
        dx, dy, dt,
        is_red    # boolean: True for red points, False for black points
    ):
        """
        Red-Black Gauss-Seidel GPU kernel using cooperative groups pattern.
        
        Red-black ordering allows parallelization of Gauss-Seidel:
        - Red points: (i+j) % 2 == 0
        - Black points: (i+j) % 2 == 1
        
        Since red points only depend on black neighbors and vice versa,
        we can update all red points in parallel, then all black points.
        """
        # Global thread indices
        j0 = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        i0 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        
        # Grid strides
        j_stride = cuda.blockDim.x * cuda.gridDim.x
        i_stride = cuda.blockDim.y * cuda.gridDim.y
        
        ny1, nx1 = T.shape
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        
        # Determine starting parity for this thread
        # We want i+j to match the target parity (red=0, black=1)
        target_parity = 0 if is_red else 1
        
        # Grid-stride loop
        for i in range(1 + i0, ny1 - 1, i_stride):
            for j in range(1 + j0, nx1 - 1, j_stride):
                # Check if this point matches the current color
                if (i + j) % 2 != target_parity:
                    continue
                
                # Face conductivities
                kvx1 = kvx[i, j-1]
                kvx2 = kvx[i, j]
                kvy1 = kvy[i-1, j]
                kvy2 = kvy[i, j]
                
                # Diagonal term
                A_diag = (
                    rhocp[i, j] / dt
                    + (kvx1 + kvx2) * inv_dx2
                    + (kvy1 + kvy2) * inv_dy2
                )
                
                # RHS: only time derivative term
                rhs = (rhocp[i, j] / dt) * T0[i, j]
                
                # Off-diagonal contribution
                # For red points, all neighbors are black (already updated)
                # For black points, all neighbors are red (already updated)
                sum_nb = (
                    (-kvx1 * inv_dx2) * T[i, j-1] +
                    (-kvx2 * inv_dx2) * T[i, j+1] +
                    (-kvy1 * inv_dy2) * T[i-1, j] +
                    (-kvy2 * inv_dy2) * T[i+1, j]
                )
                
                # Gauss-Seidel update: solve for T[i,j]
                T[i, j] = (rhs - sum_nb) / A_diag


    @cuda.jit(fastmath=True, cache=True)
    def apply_thermal_BC_kernel(T, T_BC_TOP, T_BC_BOTTOM):
        """Apply boundary conditions on GPU."""
        ny1, nx1 = T.shape
        
        # Thread index for column (x-direction)
        j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        
        if j < nx1:
            # Top boundary: (T[0, :] + T[1, :])/2 = T_BC_TOP
            T[0, j] = 2.0 * T_BC_TOP - T[1, j]
            
            # Bottom boundary: (T[-1, :] + T[-2, :])/2 = T_BC_BOTTOM
            T[ny1-1, j] = 2.0 * T_BC_BOTTOM - T[ny1-2, j]
        
        # Synchronize before applying Neumann BC
        cuda.syncthreads()
        
        # Thread index for row (y-direction)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        
        if i < ny1:
            # Left boundary: Neumann (insulating)
            T[i, 0] = T[i, 1]
            
            # Right boundary: Neumann (insulating)
            T[i, nx1-1] = T[i, nx1-2]


    @cuda.jit(fastmath=True, cache=True)
    def thermal_residual_kernel(
        res_partial,  # (num_blocks,) output: partial residual squared per block
        T,            # (ny1, nx1) current iterate
        T0,           # (ny1, nx1) previous timestep solution
        kvx,          # (ny1, nx1) conductivity at vertical faces
        kvy,          # (ny1, nx1) conductivity at horizontal faces
        rhocp,        # (ny1, nx1)
        dx, dy, dt
    ):
        """Compute energy norm residual in parallel using block-level reduction."""
        # Shared memory for block-level reduction
        shared = cuda.shared.array(256, dtype=types.float64)
        
        # Global thread indices
        j0 = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        i0 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        
        # Thread index within block (1D flattened)
        tid = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
        
        # Grid strides
        j_stride = cuda.blockDim.x * cuda.gridDim.x
        i_stride = cuda.blockDim.y * cuda.gridDim.y
        
        ny1, nx1 = T.shape
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        
        # Accumulate local residual
        local_res2 = 0.0
        
        for i in range(1 + i0, ny1 - 1, i_stride):
            for j in range(1 + j0, nx1 - 1, j_stride):
                kvx1 = kvx[i, j-1]
                kvx2 = kvx[i, j]
                kvy1 = kvy[i-1, j]
                kvy2 = kvy[i, j]
                
                A_diag = (
                    rhocp[i, j] / dt
                    + (kvx1 + kvx2) * inv_dx2
                    + (kvy1 + kvy2) * inv_dy2
                )
                
                rhs = (rhocp[i, j] / dt) * T0[i, j]
                
                sum_nb = (
                    (-kvx1 * inv_dx2) * T[i, j-1] +
                    (-kvx2 * inv_dx2) * T[i, j+1] +
                    (-kvy1 * inv_dy2) * T[i-1, j] +
                    (-kvy2 * inv_dy2) * T[i+1, j]
                )
                
                # Residual r = rhs - (A*T)
                r = rhs - (sum_nb + A_diag * T[i, j])
                
                # Energy norm: weight by inverse of diagonal
                local_res2 += (r * r) / A_diag
        
        # Store in shared memory
        if tid < 256:
            shared[tid] = local_res2
        else:
            shared[0] = 0.0  # Safety for blocks with >256 threads
        
        cuda.syncthreads()
        
        # Block-level reduction
        s = 128
        while s > 0:
            if tid < s and tid + s < 256:
                shared[tid] += shared[tid + s]
            cuda.syncthreads()
            s //= 2
        
        # Write block result
        if tid == 0:
            block_id = cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.x
            res_partial[block_id] = shared[0]


class ThermoMechanical2D(IncompressibleStokes2DMG):
    """
    Thermo-mechanical 2D solver using GPU-accelerated Red-Black Gauss-Seidel
    for the thermal diffusion equation.
    """
    
    def __init__(self, ctx):
        super().__init__(ctx)
        
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "Red-Black Gauss-Seidel GPU solver requires CuPy and Numba CUDA. "
                "Please install them or use a CPU solver."
            )
        
        self.use_gpu = True
        logger.info("Initialized Red-Black Gauss-Seidel GPU thermal solver")

    def solve(self, ctx):
        # Step 1: Solve the Stokes problem
        super().solve(ctx)

        # Step 2: Solve the thermal problem using Red-Black Gauss-Seidel on GPU
        s, p, o = ctx
        
        # Transfer data to GPU if not already there
        T = cp.asarray(s.T0.copy())
        T0_gpu = cp.asarray(s.T0)
        kvx_gpu = cp.asarray(s.kvx)
        kvy_gpu = cp.asarray(s.kvy)
        rhocp_gpu = cp.asarray(s.rhocp)
        
        # Boundary conditions
        T_BC_TOP = 1573.0     # K
        T_BC_BOTTOM = 1573.0  # K
        
        # Kernel launch configuration
        ny1, nx1 = T.shape
        block_size, grid_size = launch_2D((ny1, nx1))
        
        # Get Numba stream for synchronization with CuPy
        stream = get_numba_stream()
        
        # Iteration parameters
        max_iter = 500
        
        # Allocate array for residual reduction
        num_blocks = grid_size[0] * grid_size[1]
        res_partial = cp.zeros(num_blocks, dtype=cp.float64)
        
        logger.info(f"Starting Red-Black Gauss-Seidel (GPU): grid={grid_size}, block={block_size}")
        
        for it in range(max_iter):
            # RED sweep: update all red points (i+j even)
            redblack_gauss_seidel_sweep_gpu[grid_size, block_size, stream](
                T, T0_gpu, kvx_gpu, kvy_gpu, rhocp_gpu,
                s.dx, s.dy, s.dt,
                True  # is_red
            )
            
            # Synchronize after red sweep
            stream.synchronize()
            
            # BLACK sweep: update all black points (i+j odd)
            redblack_gauss_seidel_sweep_gpu[grid_size, block_size, stream](
                T, T0_gpu, kvx_gpu, kvy_gpu, rhocp_gpu,
                s.dx, s.dy, s.dt,
                False  # is_red = False (black)
            )
            
            # Synchronize after black sweep
            stream.synchronize()
            
            # Apply boundary conditions
            bc_blocks = (nx1 + 255) // 256
            bc_threads = min(256, max(nx1, ny1))
            apply_thermal_BC_kernel[bc_blocks, bc_threads, stream](
                T, T_BC_TOP, T_BC_BOTTOM
            )
            
            stream.synchronize()
            
            # Compute residual every iteration for monitoring
            thermal_residual_kernel[grid_size, block_size, stream](
                res_partial, T, T0_gpu, kvx_gpu, kvy_gpu, rhocp_gpu,
                s.dx, s.dy, s.dt
            )
            stream.synchronize()
            
            # Final reduction on CPU
            total_res2 = float(cp.sum(res_partial))
            npts = (ny1 - 2) * (nx1 - 2)
            res = math.sqrt(total_res2 / npts) if npts > 0 else 0.0
            
            print(f"[Red-Black GS GPU] iter {it:4d}  residual = {res:.6e}")
        
        # Copy result back to state
        s.T0[:, :] = cp.asnumpy(T)
        
        logger.info("Red-Black Gauss-Seidel thermal solve complete")

    def dump(self, ctx):
        s, p, o = ctx

        # Dump state to file
        with open(f"frame_{str(self.frame).zfill(self.zpad)}.npz", 'wb') as f:
            np.savez(f, vx=s.vx, vy=s.vy, p=s.p,
                    rho=s.rho, etab=s.etab, etap=s.etap,
                    T=s.T0, kx=s.kvx, ky=s.kvy, alpha=s.alpha, rhocp=s.rhocp
                    )
        
        logger.info(f"Frame {self.frame} written to file.")
        self.frame += 1  # Increment frame counter
