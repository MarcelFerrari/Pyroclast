# The Profiler Subpackage

As this project is co-developed as part of Master's Thesis and a Bachelor's Thesis, there's some segmentation in 
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