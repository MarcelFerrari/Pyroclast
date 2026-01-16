"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: examples/stokes_2D/circular_inclusion_mg/render_animation_T_only.py
Description: Render animation for the multigrid circular inclusion example (temperature only).

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tqdm

# Load all .npz frames
fnames = sorted(f for f in os.listdir() if f.startswith("frame_") and f.endswith(".npz"))
print(f"Found {len(fnames)} frames")

if len(fnames) == 0:
    raise RuntimeError("No frame_*.npz files found in the current directory.")

frames = [np.load(fname) for fname in fnames]

# Initialize figure and axes (single panel)
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
plt.tight_layout()

# Load first frame to determine shape and initial data
s0 = frames[0]
T0 = s0["T"]
ny, nx = T0.shape

# Create initial imshow object for temperature
im_T = ax.imshow(T0, cmap="jet")  # keep same colormap as original script
ax.set_title("Temperature")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Add colorbar
fig.colorbar(im_T, ax=ax)

# Progress bar
pbar = tqdm.tqdm(total=len(frames), desc="Animating frames", unit="frame")

# Update function
def animate(i):
    s = frames[i]
    im_T.set_data(s["T"])
    pbar.update(1)
    pbar.set_postfix(frame=i)

# Create and save animation
ani = anim.FuncAnimation(fig, animate, frames=len(frames), interval=100)
ani.save("animation_T_only.mp4", writer="ffmpeg", fps=5)

pbar.close()
