# domain_decomposition.py
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

def halo_exchange_2D(dsts, srcs, wait = True):
    if MPI is None:
        raise RuntimeError("MPI is not available")

    comm = get_cart_comm()
    coords = comm.Get_coords(comm.Get_rank())
    dims = comm.Get_topo()[0]

    # Neighbor ranks (None = boundary)
    def rank(i, j):
        if 0 <= i < dims[0] and 0 <= j < dims[1]:
            return comm.Get_cart_rank((i, j))
        return MPI.PROC_NULL

    i, j = coords
    neighbors = {
        'up':    rank(i - 1, j),
        'down':  rank(i + 1, j),
        'left':  rank(i, j - 1),
        'right': rank(i, j + 1),
        'nw':    rank(i - 1, j - 1),
        'ne':    rank(i - 1, j + 1),
        'sw':    rank(i + 1, j - 1),
        'se':    rank(i + 1, j + 1),
    }

    # Unpack source and destination buffers (1D or 2D slices)
    up_src, down_src, left_src, right_src, nw_src, ne_src, sw_src, se_src = srcs
    up_dst, down_dst, left_dst, right_dst, nw_dst, ne_dst, sw_dst, se_dst = dsts

    reqs = []

    # Cardinal directions
    if neighbors['up'] != MPI.PROC_NULL:
        reqs.append(comm.Isend(down_src, dest=neighbors['up']))
        reqs.append(comm.Irecv(up_dst, source=neighbors['up']))
    if neighbors['down'] != MPI.PROC_NULL:
        reqs.append(comm.Isend(up_src, dest=neighbors['down']))
        reqs.append(comm.Irecv(down_dst, source=neighbors['down']))
    if neighbors['left'] != MPI.PROC_NULL:
        reqs.append(comm.Isend(right_src, dest=neighbors['left']))
        reqs.append(comm.Irecv(left_dst, source=neighbors['left']))
    if neighbors['right'] != MPI.PROC_NULL:
        reqs.append(comm.Isend(left_src, dest=neighbors['right']))
        reqs.append(comm.Irecv(right_dst, source=neighbors['right']))

    # Diagonal directions
    if neighbors['nw'] != MPI.PROC_NULL:
        reqs.append(comm.Isend(se_src, dest=neighbors['nw']))
        reqs.append(comm.Irecv(nw_dst, source=neighbors['nw']))
    if neighbors['ne'] != MPI.PROC_NULL:
        reqs.append(comm.Isend(sw_src, dest=neighbors['ne']))
        reqs.append(comm.Irecv(ne_dst, source=neighbors['ne']))
    if neighbors['sw'] != MPI.PROC_NULL:
        reqs.append(comm.Isend(ne_src, dest=neighbors['sw']))
        reqs.append(comm.Irecv(sw_dst, source=neighbors['sw']))
    if neighbors['se'] != MPI.PROC_NULL:
        reqs.append(comm.Isend(nw_src, dest=neighbors['se']))
        reqs.append(comm.Irecv(se_dst, source=neighbors['se']))

    if wait:
        MPI.Request.Waitall(reqs)
        return None
    else:
        return reqs
