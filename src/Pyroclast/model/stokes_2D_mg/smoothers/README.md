# The Smoothers Subpackage

As this project is co-developed as part of Marcel Ferrari's Master's Thesis and Alexander Sotoudeh's Bachelor's Thesis, there's some segmentation in 
ownership and modularity of a given submodule.

The smoother package will contain all implementations of the smoother routine that are part of the Bachelor's Thesis:
```python
def _vx_rb_gs_sweep(*args, **kwargs):
    ...

def _vy_red_black_gs_sweep(*args, **kwargs):
    ...

def velocity_smoother(nx1, ny1,
                      dx, dy,
                      etap, etab,
                      vx, vy,
                      relax_v, BC,
                      vx_rhs, vy_rhs, max_iter):
    """
    Full Uzawa smoother for velocity and pressure.
    """
    for _ in range(max_iter):
        vx = _vx_rb_gs_sweep(nx1, ny1,
                            dx, dy,
                            etap, etab,
                            vx, vy,
                            relax_v, vx_rhs, BC)
        
        vy = _vy_red_black_gs_sweep(nx1, ny1,
                                    dx, dy,
                                    etap, etab,
                                    vx, vy,
                                    relax_v, vy_rhs, BC)
        
    return vx, vy
```

The exact structure of the outer smoother loop as well as the inner loops and the loop bodies is subject to 
change and experimentation. The general idea for files are:
- If they are public: they expose `velocity_smoother` (a common definition will be provided later) and other submethods (tho this is not standardized)
- If they are private, they don't follow any standard and the user needs to be aware of the file contents.

# TODOS:
Imporvement Ideas:
Red-Black-Gauss-Seidel
- Fully Staggered / Four loops fused, LTR 
- Fully Staggered / Four Loops fused, LTR + random offset
- Half Staggered / VX, VY Red fused, VX, VY Black Fused
- Half Staggered / VX, VY Red fused, VX, VY Black Fused + LTR
- Half Staggered / VX, VY Red fused, VX, VY Black Fused + LTR + Random Offset

Jacobi:
- Fully staggered / Two loops fused
- Fully staggered / Two loops fused, LTR
- Fully staggered / Two Loops fused, LTR + Rando Offset



## File Table of Contents
| File                   | Has Inline | Has Blocking A Thread | Has Blocking A Stride | Has Time Blocking  | Loop Fusion | Has VX | Has VY | Has Smoother | Comment                                                                                          |
|------------------------|------------|-----------------------|-----------------------|--------------------|-------------|--------|--------|--------------|--------------------------------------------------------------------------------------------------|
| `base_rb_gs.py`        | F          | F                     | F                     | F                  | F           | T      | T      | T            |                                                                                                  |
| `base_rb_gs_reorder`   | F          | F                     | F                     | F                  | F           | T      | F      | F            | Reordered Computation for Fused-Multiply-Addition                                                |
| `vx._inline_vx`        | F          | F                     | F                     | F                  | F           | T      | F      | F            | Contains shared loop bodies (functions to be inlined)                                            |
| `vx.base_vx_gs`        | F          | F                     | F                     | F                  | F           | T      | F      | F            | Testing Pure Gauss Seidel                                                                        |
| `vx.base_vx_jacobi`    | F          | F                     | F                     | F                  | F           | T      | F      | F            | Testing Pure Jacobi Iteration                                                                    |
| `vx.caching_vx_jacobi` | T          | F                     | F                     | F                  | F           | T      | F      | F            | Testing Caching of Coefficients with Jacobi                                                      |
| `vx.caching_vx_rb_gs`  | T          | F                     | F                     | F                  | F           | T      | F      | F            | Testing Caching of Coefficients with Red-Black Gauss-Seidel                                      |
| `vx.inline_vx_gs`      | T          | F                     | F                     | F                  | F           | T      | F      | F            | Checking no performance impact when using inlined functions Gauss-Seidel                         |
| `vx.inline_vx_jacobi`  | T          | F                     | F                     | F                  | F           | T      | F      | F            | Checking no performance impact when using inlined functions Jacobi                               |
| `vx.inline_vx_rb_gs`   | T          | F                     | F                     | F                  | F           | T      | F      | F            | Checking no performance impact when using inlined functions Red-Black Gauss-Seidel               |
| `vx.mk_sg_vx_rb_gs_v1` | T          | T                     | F                     | F                  | T           | T      | F      | F            | Implementation 1 of Red-Black Loop Fusion AND Blocking                                           |
| `vx.mk_vx_jacobi_v1`   | T          | T                     | F                     | F                  | F           | T      | F      | F            | Implementation 1 of Jacobi Thread Blocking                                                       |
| `vx.mk_vx_rb_gs`       | T          | T                     | F                     | F                  | F           | T      | F      | F            | Red-Black Gauss-Seidel with Thread Blocking                                                      |
| `vx.mk_vx_rb_gs_v2`    | T          | F                     | T                     | F                  | F           | T      | F      | F            | Red-Black Gauss-Seidel with Stride Blocking                                                      |
| `vx.sg_vx_rb_gs_v1`    | T          | T                     | T                     | F                  | T           | T      | F      | F            | Implementation 1 of Red-Black Gauss-Seidel (Internal If)                                         |
| `vx.sg_vx_rb_gs_v2`    | T          | T                     | F                     | F                  | T           | T      | F      | F            | Implementation 2 of Red-Black Gauss-Seidel (Start and End Unrolled)                              |
| `vx.sg_vx_rb_gs_v3`    | T          | T                     | F                     | F                  | T           | T      | F      | F            | Implementation 3 of Red-Black Gauss-Seidel (Start and End Unrolled with Support for small grids) |
 
