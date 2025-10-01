import numba as nb
import numpy as np


# @nb.njit(cache=True, parallel=True)
# def update_values(array: np.ndarray, x, y, threads: int):
#     for t in nb.prange(threads):
#         start = (y / threads) * t
#         end = y if t + 1 == threads else (y / threads) * (t + 1)
#
#         for j in range(x):
#             start_i = start if (start + j) % 2 == 0 else start + 1
#             end_i = end
#             for i in range(start_i, end_i):
#                 array[i, j] = array[i, j] * 2


# @nb.njit()
# def test_return(i: int, j: int, th: int):
#     a = j if (i + 2) % 2 == 0 else th
#     return a


@nb.njit()
def test_return(i: int, j: int, th: int):
    a = j * ((i + 2) % 2 == 0) + th * ((i + 2) % 2 == 1)
    return a

if __name__ == "__main__":
    nx = ny = 4096
    ny1 = ny + 1
    nx1 = nx + 1

    a = np.random.rand(ny1, nx1) * 1e19 + 1e19

    th = nb.get_num_threads()

    # update_values(a, x=nx, y=ny, threads=th)
    test_return(nx, 12, th)