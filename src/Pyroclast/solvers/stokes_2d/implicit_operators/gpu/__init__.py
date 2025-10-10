from .implicit_operators import (
    # ========= Operators =========
    vx_operator,
    vx_residual,
    vy_operator,
    vy_residual,
    p_operator,
    p_residual,
    uzawa_velocity_rhs,
    uzawa_vx_operator,
    uzawa_vx_residual,
    uzawa_vy_operator,
    uzawa_vy_residual,
    
    # ========= Energy norms =========
    compute_p_energy_norm,
    compute_vx_energy_norm,
    compute_vy_energy_norm,
)
