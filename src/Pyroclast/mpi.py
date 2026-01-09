# domain_decomposition.py
import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Utilities for managing MPI Cartesian communicators
# Singleton pattern to ensure only one Cartesian communicator is created
_cart_comm = None
def create_cart_comm(*args, **kwargs):
    global _cart_comm
    if _cart_comm is None and MPI is not None:
        comm = MPI.COMM_WORLD
        _cart_comm = comm.Create_cart(*args, **kwargs)
    return _cart_comm

def get_cart_comm():
    return _cart_comm

# We use COMM_WORLD rank to shard filenames
def get_shard_filename(fname):
    if MPI is None:
        return fname
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    zfill = len(str(size - 1))
    return f"{fname}.rank_{str(rank).zfill(zfill)}"


def halo_exchange_2D(grid_values, blocking=True):
    comm = get_cart_comm()
    rank = comm.Get_rank()
    gi, gj = comm.Get_coords(rank)
    py, px = comm.Get_topo()[0]  # dims (rows, cols)
    ny1, nx1 = grid_values.shape  # Local grid size including halos
    dtype = grid_values.dtype

    # Function to get shifted rank coordinates
    def shift_coords(di, dj):
        i, j = gi + di, gj + dj

        # If out of bounds, return MPI.PROC_NULL
        if i < 0 or i >= py or j < 0 or j >= px:
            return MPI.PROC_NULL

        return comm.Get_cart_rank((i, j))

    # Need to detect edge ranks
    is_top_edge = (gi == 0)
    is_bottom_edge = (gi == py - 1)
    is_left_edge = (gj == 0)
    is_right_edge = (gj == px - 1)
    
    # Standard halo row goes from 1 to -2
    # That is nx1 - 3 elements
    # However, if we are at the edge of the global domain,
    # we extend the domain by 1 cell.
    # This is because halo rows become boundary rows in that case.
    row_size = nx1 - 3
    if is_left_edge or is_right_edge:
        row_size += 1
    jmin = 1 if not is_left_edge else 0
    jmax = -2 if not is_right_edge else -1

    # Similarly for halo columns
    col_size = ny1 - 3
    if is_top_edge or is_bottom_edge:
        col_size += 1
    imin = 1 if not is_top_edge else 0
    imax = -2 if not is_bottom_edge else -1
    
    comm = get_cart_comm()

    # We need to perform 8-way halo exchange
    N_rank = shift_coords(-1, 0)  # North rank coordinates
    S_rank = shift_coords(1, 0)  # South rank coordinates
    E_rank = shift_coords(0, 1)  # East rank coordinates
    W_rank = shift_coords(0, -1) # West rank coordinates
    NE_rank = shift_coords(-1, 1) # North-East rank coordinates
    NW_rank = shift_coords(-1, -1) # North-West rank coordinates
    SE_rank = shift_coords(1, 1)  # South-East rank coordinates
    SW_rank = shift_coords(1, -1) # South-West rank coordinates

    # Prepare source and destination buffers for halo exchange
    reqs = []

    # North halo
    if N_rank != MPI.PROC_NULL:
        # Post receives
        N_recv_buf = np.empty(row_size, dtype=dtype)
        reqs.append(comm.Irecv(N_recv_buf, source=N_rank))

        # Post sends
        N_send_buf = grid_values[1, jmin:jmax].copy()
        reqs.append(comm.Isend(N_send_buf, dest=N_rank))

    # South halo
    if S_rank != MPI.PROC_NULL:
        # Post receives
        S_recv_buf = np.empty(row_size, dtype=dtype)
        reqs.append(comm.Irecv(S_recv_buf, source=S_rank))

        # Post sends
        S_send_buf = grid_values[-3, jmin:jmax].copy()
        reqs.append(comm.Isend(S_send_buf, dest=S_rank))

    # West halo
    if W_rank != MPI.PROC_NULL:
        # Post receives
        W_recv_buf = np.empty(col_size, dtype=dtype)
        reqs.append(comm.Irecv(W_recv_buf, source=W_rank))

        # Post sends
        W_send_buf = grid_values[imin:imax, 1].copy()
        reqs.append(comm.Isend(W_send_buf, dest=W_rank))

    # East halo
    if E_rank != MPI.PROC_NULL:
        # Post receives
        E_recv_buf = np.empty(col_size, dtype=dtype)
        reqs.append(comm.Irecv(E_recv_buf, source=E_rank))

        # Post sends
        E_send_buf = grid_values[imin:imax, -3].copy()
        reqs.append(comm.Isend(E_send_buf, dest=E_rank))

    # North-West halo
    if NW_rank != MPI.PROC_NULL:
        # Post receives
        NW_recv_buf = np.empty(1, dtype=dtype)
        reqs.append(comm.Irecv(NW_recv_buf, source=NW_rank))

        # Post sends
        NW_send_buf = np.array([grid_values[1, 1]], dtype=dtype)
        reqs.append(comm.Isend(NW_send_buf, dest=NW_rank))
    
    # South-West halo
    if SW_rank != MPI.PROC_NULL:
        # Post receives
        SW_recv_buf = np.empty(1, dtype=dtype)
        reqs.append(comm.Irecv(SW_recv_buf, source=SW_rank))

        # Post sends
        SW_send_buf = np.array([grid_values[-3, 1]], dtype=dtype)
        reqs.append(comm.Isend(SW_send_buf, dest=SW_rank))

    # North-East halo
    if NE_rank != MPI.PROC_NULL:
        # Post receives
        NE_recv_buf = np.empty(1, dtype=dtype)
        reqs.append(comm.Irecv(NE_recv_buf, source=NE_rank))

        # Post sends
        NE_send_buf = np.array([grid_values[1, -3]], dtype=dtype)
        reqs.append(comm.Isend(NE_send_buf, dest=NE_rank))

    # South-East halo
    if SE_rank != MPI.PROC_NULL:
        # Post receives
        SE_recv_buf = np.empty(1, dtype=dtype)
        reqs.append(comm.Irecv(SE_recv_buf, source=SE_rank))

        # Post sends
        SE_send_buf = np.array([grid_values[-3, -3]], dtype=dtype)
        reqs.append(comm.Isend(SE_send_buf, dest=SE_rank))

    class MPIFuture:
        def wait(self):
            # Wait for all communications to complete
            MPI.Request.Waitall(reqs)

            # Update halo regions with received data
            if N_rank != MPI.PROC_NULL:
                grid_values[0, jmin:jmax] = N_recv_buf
            
            if S_rank != MPI.PROC_NULL:
                grid_values[-2, jmin:jmax] = S_recv_buf

            if W_rank != MPI.PROC_NULL:
                grid_values[imin:imax, 0] = W_recv_buf

            if E_rank != MPI.PROC_NULL:
                grid_values[imin:imax, -2] = E_recv_buf

            if NW_rank != MPI.PROC_NULL:
                grid_values[0, 0] = NW_recv_buf

            if SW_rank != MPI.PROC_NULL:
                grid_values[-2, 0] = SW_recv_buf

            if NE_rank != MPI.PROC_NULL:
                grid_values[0, -2] = NE_recv_buf
            
            if SE_rank != MPI.PROC_NULL:
                grid_values[-2, -2] = SE_recv_buf

            return grid_values
    
    rax = MPIFuture()
    if blocking:
        return rax.wait()
    else:
        return rax