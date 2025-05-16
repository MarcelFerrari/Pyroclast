import numba as nb
import numpy as np
from Pyroclast.profiling import timer

# Numba compiled functions - Not part of the class
@nb.njit(cache = True, inline='always')
def idx(ny1, nx1, i, j, q):
    # Helper function to map 2D indices to 1D index
    # q: variable index (0: P, 1: vx, 2: vy)
    # i: matrix row index (y-index)
    # j: matrix column index (x-index)
    return  q * nx1 * ny1 + i * nx1 + j
    
@nb.njit(cache = True, inline='always')
def insert(i_idx, j_idx, vals, ptr, i, j, v):
    i_idx[ptr[0]] = i
    j_idx[ptr[0]] = j
    vals[ptr[0]] = v
    # Increment current index
    ptr[0] += 1
   
@timer.time_function("Model Solve", "Assemble")
@nb.njit(cache=True)
def assemble_matrix(nx1, ny1, dx, dy, etap, etab):
    # Assemble matrix in COO format
    n_eqs = 3               # Number of equations to solve
    n_rows = nx1*ny1*n_eqs  # Number of rows in the matrix        
    max_nnz = 12            # Maximum number of non-zero elements (~12 per row)

    # Preallocate memory for COO format
    i_idx = np.zeros((max_nnz*n_rows,), dtype=np.int32)
    j_idx = np.zeros((max_nnz*n_rows,), dtype=np.int32)
    vals = np.zeros((max_nnz*n_rows,), dtype=np.float64)
    ptr = np.array([0], dtype=np.int32)
    # BC = -1
    # Loop over the grid
    for i in range(ny1):
        for j in range(nx1):
            # Continuity equation (P)
            kij = idx(ny1, nx1, i, j, 0)

            # Set P = 0 for ghost nodes
            if i == 0 or j == 0 or i == ny1 - 1 or j == nx1 - 1:
                insert(i_idx, j_idx, vals, ptr, kij, kij, 1.0)
            elif i == 1 and j == 1:
                insert(i_idx, j_idx, vals, ptr, kij, kij, 1.0)
            else:
                # vx coefficients
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j, 1), 1.0/dx)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j-1, 1), -1.0/dx)

                # vy coefficients
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j, 2), 1.0/dy)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i-1, j, 2), -1.0/dy)

            # 2) x-momentum equation (vx)
            kij = idx(ny1, nx1, i, j, 1)
            
            if j == 0 or j >= nx1-2: # Last two nodes in x-direction
                insert(i_idx, j_idx, vals, ptr, kij, kij, 1.0)
            elif i == 0 or i == ny1 - 1: # First and last nodes in y-direction
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j, 1), 1.0)
                # if i == 0: # Top boundary
                #     insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i+1, j, 1), BC)
                # else: # Bottom boundary
                #     insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i-1, j, 1), BC)
            else:
                # Extract viscosity values
                etaA = etap[i, j]
                etaB = etap[i, j+1]
                eta1 = etab[i-1, j]
                eta2 = etab[i, j]

                # vx coefficients
                vx1_coeff = 2*etaA/dx**2
                vx2_coeff = eta1/dy**2
                vx3_coeff = -eta1/dy**2 - eta2/dy**2 - 2*etaA/dx**2 - 2*etaB/dx**2
                vx4_coeff = eta2/dy**2
                vx5_coeff = 2*etaB/dx**2

                # Store vx coefficients
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j-1, 1), vx1_coeff) #vx1 = vx(i, j-1)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i-1, j, 1), vx2_coeff) #vx2 = vx(i-1, j)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j, 1), vx3_coeff) #vx3 = vx(i, j)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i+1, j, 1), vx4_coeff) #vx4 = vx(i+1, j)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j+1, 1), vx5_coeff) #vx5 = vx(i, j+1)

                # vy coefficients
                vy1_coeff = eta1/(dx*dy)
                vy2_coeff = -eta2/(dx*dy)
                vy3_coeff = -eta1/(dx*dy)
                vy4_coeff = eta2/(dx*dy)

                #Store vy coefficients
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i-1, j, 2), vy1_coeff)   #vy1 = vy(i-1, j)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j, 2), vy2_coeff)     #vy2 = vy(i, j)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i-1, j+1, 2), vy3_coeff) #vy3 = vy(i-1, j+1)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j+1, 2), vy4_coeff)   #vy4 = vy(i, j+1)
                        
                # -dP/dx
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j+1, 0), -1.0/dx)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j, 0), 1.0/dx)
                
            # 3) y-momentum equation (vy)
            kij = idx(ny1, nx1, i, j, 2)

            if i == 0 or i >= ny1 - 2: # Last two nodes in y-direction
                insert(i_idx, j_idx, vals, ptr, kij, kij, 1.0)
            elif j == 0 or j == nx1 - 1:
                insert(i_idx, j_idx, vals, ptr, kij, kij, 1.0)
                # if j == 0:
                #     insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j+1, 2), BC)
                # else:
                #     insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j-1, 2), BC)
            else:
                # Extract viscosity values
                etaA = etap[i, j]
                etaB = etap[i+1, j]
                eta1 = etab[i, j-1]
                eta2 = etab[i, j]

                # vy coefficients
                vy1_coeff = eta1/dx**2
                vy2_coeff = 2*etaA/dy**2
                vy3_coeff = -2*etaA/dy**2 - 2*etaB/dy**2 - eta1/dx**2 - eta2/dx**2
                vy4_coeff = 2*etaB/dy**2
                vy5_coeff = eta2/dx**2

                # Store vy coefficients
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j-1, 2), vy1_coeff) #vy1 = vy(i, j-1)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i-1, j, 2), vy2_coeff) #vy2 = vy(i-1, j)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j, 2), vy3_coeff) #vy3 = vy(i, j)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i+1, j, 2), vy4_coeff) #vy4 = vy(i+1, j)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j+1, 2), vy5_coeff) #vy5 = vy(i, j+1)

                #vx coefficients
                vx1_coeff = eta1/(dx*dy)
                vx2_coeff = -eta1/(dx*dy)
                vx3_coeff = -eta2/(dx*dy)
                vx4_coeff = eta2/(dx*dy)

                # Store vx coefficients
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j-1, 1), vx1_coeff) #vx1 = vx(i, j-1)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i+1, j-1, 1), vx2_coeff) #vx2 = vx(i+1, j-1)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j, 1), vx3_coeff) #vx3 = vx(i, j)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i+1, j, 1), vx4_coeff) #vx4 = vx(i+1, j)

                # -dP/dy
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i+1, j, 0), -1.0/dy)
                insert(i_idx, j_idx, vals, ptr, kij, idx(ny1, nx1, i, j, 0), 1.0/dy)
                
    return i_idx, j_idx, vals

