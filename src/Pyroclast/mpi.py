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