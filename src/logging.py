"""Central logging utilities with optional MPI support."""

import logging as py_logging

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except Exception:
    rank = 0

class MPIRankFilter(py_logging.Filter):
    """Filter INFO messages to only emit from rank 0."""
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
    def filter(self, record: py_logging.LogRecord) -> bool:
        if record.levelno == py_logging.INFO and self.rank != 0:
            return False
        return True

def get_logger(name: str = None) -> py_logging.Logger:
    """Return a configured logger with MPI rank filtering."""
    logger = py_logging.getLogger(name)
    if not logger.handlers:
        handler = py_logging.StreamHandler()
        fmt = '%(asctime)s - %(levelname)s - %(message)s'
        handler.setFormatter(py_logging.Formatter(fmt))
        handler.addFilter(MPIRankFilter(rank))
        logger.addHandler(handler)
        logger.setLevel(py_logging.DEBUG)
        logger.propagate = False
    return logger
