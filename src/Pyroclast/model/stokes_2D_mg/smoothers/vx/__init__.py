"""
INFO: This package contains sub implementations that only compute the vx update. (needed to different vx update schemes)
"""
from _inline_vx import inline_loop_body_vx, prep_vx_cache, compute_coeffs, compute_neighbor_sum