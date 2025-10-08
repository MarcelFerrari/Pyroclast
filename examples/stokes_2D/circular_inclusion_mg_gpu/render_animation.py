import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tqdm

# Load all .npz frames
fnames = sorted(f for f in os.listdir() if f.startswith('frame_') and f.endswith('.npz'))
# fnames = fnames[:1]
print(f'Found {len(fnames)} frames')

frames = [np.load(fname) for fname in fnames]

# Initialize figure and axes
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
plt.tight_layout()

# Load first frame to determine shapes
s0 = frames[0]
ny, nx = s0['vx'].shape  # Assuming vx and vy have same shape

# Create a grid for quiver arrows
step = 40  # Controls arrow density
y, x = np.mgrid[0:ny:step, 0:nx:step]

# Manual scaling for visibility
scale_factor = 1e13

# Create initial imshow objects
images = [
    axs[0, 0].imshow(s0['rho'], cmap='jet'),
    axs[0, 1].imshow(np.log10(s0['etab']), cmap='jet'),
    axs[0, 2].imshow(np.log10(s0['etap']), cmap='jet'),
    axs[1, 0].imshow(s0['vx'], cmap='jet'),
    axs[1, 1].imshow(s0['vy'], cmap='jet'),
    axs[1, 2].imshow(s0['p'], cmap='jet'),
]

# Add vector field (quiver) arrows — scaled manually
quiv_axes = [0, 1, 2]
quivers = []
for i in quiv_axes:
    q = axs[0, i].quiver(x, y,
                        s0['vx'][::step, ::step] * scale_factor,
                        -s0['vy'][::step, ::step] * scale_factor,  # <-- flip Y
                        color='white')
    quivers.append(q)

# Set titles and labels
titles = ['Density', 'log10(etab)', 'log10(etap)', 'vx', 'vy', 'Pressure']
for ax, title in zip(axs.flat, titles):
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# Add colorbars
for ax, im in zip(axs.flat, images):
    fig.colorbar(im, ax=ax)

# Progress bar
pbar = tqdm.tqdm(total=len(frames), desc='Animating frames', unit='frame')

# Update function
def animate(i):
    s = frames[i]
    images[0].set_data(s['rho'])
    images[1].set_data(np.log10(s['etab']))
    images[2].set_data(np.log10(s['etap']))
    images[3].set_data(s['vx'])
    images[4].set_data(s['vy'])
    images[5].set_data(s['p'])

    # Update quiver arrows — scaled manually
    for idx, qi in enumerate(quivers):
        qi.set_UVC(s['vx'][::step, ::step] * scale_factor,
           -s['vy'][::step, ::step] * scale_factor)  # <-- flip Y

    pbar.update(1)
    pbar.set_postfix(frame=i)

# Create and save animation
ani = anim.FuncAnimation(fig, animate, frames=len(frames), interval=100)
ani.save('animation.mp4', writer='ffmpeg', fps=5) 
pbar.close()
