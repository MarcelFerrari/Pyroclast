import numpy as np
import numba as nb
from numba.typed import List
# Direction constants (fixed order)
DIR_N, DIR_S, DIR_E, DIR_W = 0, 1, 2, 3
DIR_NE, DIR_NW, DIR_SE, DIR_SW = 4, 5, 6, 7
DIR_NO_MIGRATE = 8 # Special value for no migration
N_DIRS = 8

@nb.njit(cache=True, inline='always')
def get_dir(x, y, xmin, xmax, ymin, ymax):
    went_N = y < ymin  # Note: y increases downwards
    went_S = y >= ymax
    went_E = x >= xmax
    went_W = x < xmin
    
    if went_N:
        return DIR_NE if went_E else DIR_NW if went_W else DIR_N
    elif went_S:
        return DIR_SE if went_E else DIR_SW if went_W else DIR_S
    elif went_E:
        return DIR_E
    elif went_W:
        return DIR_W
    return DIR_NO_MIGRATE

@nb.njit(cache=True, parallel=True)
def flag_migrating_markers(nm, xm, ym, xmin, xmax, ymin, ymax, n_threads=nb.get_num_threads()):
    """
    Detect markers that migrate out of the local domain.
    """
    n_still = np.zeros(n_threads, dtype=np.int64)  # Total number of still markers
    n_out = np.zeros((n_threads, N_DIRS), dtype=np.int64)
    
    # First pass: count migrating markers
    n_markers_per_threads = nm // n_threads
    rem = nm % n_threads

    for tid in nb.prange(n_threads):
        # Compute start and end indices for this thread
        start = tid * n_markers_per_threads + min(tid, rem)
        end = start + n_markers_per_threads + (1 if tid < rem else 0)
        
        # Local counters to avoid false sharing
        local_n_still = np.zeros(1, dtype=np.int64) # Single element array
                                                    # Otherwise numba goes crazy trying
                                                    # to ensure parallel reduction safety
        local_n_out = np.zeros(N_DIRS, dtype=np.int64)

        for m in range(start, end):
            dir_idx = get_dir(xm[m], ym[m], xmin, xmax, ymin, ymax)
            if dir_idx == DIR_NO_MIGRATE:
                local_n_still[0] += 1 # Remember this is an array of size 1
            else:
                local_n_out[dir_idx] += 1
        
        n_still[tid] = local_n_still[0]
        n_out[tid, :] = local_n_out

    # Reduce counts across threads
    n_still = n_still.sum()
    n_out = n_out.sum(axis=0)

    return n_still, n_out


@nb.njit(cache=True, parallel=True)
def compact_markers(nm, xm, ym, xmin, xmax, ymin, ymax, marker_properties,
                    out_marker_buffs, n_threads = nb.get_num_threads()):
    # We copy and compact migrating markers into their respective buffers
    # We do this in parallel with two passes:
    # 1. Count how many markers go to each direction per thread (partial prefix sum)
    # 2. Copy markers into buffers using the computed offsets
    # This allows to process data in parallel

    # First pass: count markers per direction per thread
    # N_DIRS + 1 to account for no-migration case DIR_NO_MIGRATE
    offsets = np.zeros((n_threads + 1, N_DIRS + 1), dtype=np.int64)

    n_markers_per_thread = nm // n_threads
    rem = nm % n_threads

    for tid in nb.prange(n_threads):
        start = tid * n_markers_per_thread + min(tid, rem)
        end = start + n_markers_per_thread + (1 if tid < rem else 0)

        # Local array to avoid false sharing
        local_offsets = np.zeros(N_DIRS + 1, dtype=np.int64)
        for m in range(start, end):
            dir_idx = get_dir(xm[m], ym[m], xmin, xmax, ymin, ymax)
            local_offsets[dir_idx] += 1
                
        offsets[tid + 1, :] = local_offsets

    # Compute global offsets (partial prefix sum over threads)
    for tid in range(1, n_threads + 1):
        offsets[tid, :] += offsets[tid - 1, :]

    # Second pass: copy markers into buffers
    for tid in nb.prange(n_threads):
        start = tid * n_markers_per_thread + min(tid, rem)
        end = start + n_markers_per_thread + (1 if tid < rem else 0)

        # Local pointers for each direction
        ptr = np.empty(N_DIRS + 1, dtype=np.int64)
        ptr[:] = offsets[tid, :]

        for m in range(start, end):
            dir_idx = get_dir(xm[m], ym[m], xmin, xmax, ymin, ymax)

            # Copy marker properties to the appropriate buffer
            for prop_idx in range(len(marker_properties)):
                out_marker_buffs[dir_idx][prop_idx][ptr[dir_idx]] = \
                                         marker_properties[prop_idx][m]

            ptr[dir_idx] += 1

def to_nb_container(nested_tuple):
    outer = List()
    for inner_tuple in nested_tuple:
        inner = List()
        for arr in inner_tuple:
            inner.append(arr)
        outer.append(inner)
    return outer