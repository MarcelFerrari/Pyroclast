from numba import cuda

print(f"Max Dim Block X: {cuda.get_current_device().MAX_BLOCK_DIM_X}")
print(f"Max Dim Block Y: {cuda.get_current_device().MAX_BLOCK_DIM_Y}")
print(f"Max Dim Block Z: {cuda.get_current_device().MAX_BLOCK_DIM_Z}")
print(f"Max Threads per Block: {cuda.get_current_device().MAX_THREADS_PER_BLOCK}")
print(f"block_dim_x * block_dim_y * block_dim_z <= Max Thread per Block")

print(f"Max Grid Dim X: {cuda.get_current_device().MAX_GRID_DIM_X}")
print(f"Max Grid Dim Y: {cuda.get_current_device().MAX_GRID_DIM_Y}")
print(f"Max Grid Dim Z: {cuda.get_current_device().MAX_GRID_DIM_Z}")
