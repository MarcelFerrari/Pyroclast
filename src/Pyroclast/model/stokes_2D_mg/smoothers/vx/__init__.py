"""
INFO: This package contains sub implementations that only compute the vx update. (needed to different vx update schemes)
"""
from _inline_vx import (cpu_inline_loop_body_vx,
                        cpu_prep_vx_cache,
                        cpu_compute_coeffs_vx,
                        cpu_compute_neighbor_sum_vx,
                        gpu_inline_loop_body_vx,
                        gpu_compute_coeffs_vx,
                        gpu_compute_neighbor_sum_vx)