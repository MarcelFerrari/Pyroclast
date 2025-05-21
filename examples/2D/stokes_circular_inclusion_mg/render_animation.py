import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tqdm

# Open all frames
fnames = []
for f in os.listdir():
    if f.startswith('frame_') and f.endswith('.pkl'):
        fnames.append(f)

print(f'Found {len(fnames)} frames')
fnames.sort()

frames = []
for path in fnames:
    f=pickle.load(open(path, 'rb'))
    frames.append(f)

# Animate a 2x3 plot of rho, log10(etab), log10(etap), vx, vy
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

framezero = True
pbar = tqdm.tqdm(total=len(frames), desc='Animating frames', unit='frame')
def animate(i):
    global framezero

    s = frames[i]['state']
    rho = s['rho']
    etab = s['etab']
    etap = s['etap']
    
    rho_plt = axs[0, 0].imshow(rho, cmap='jet')
    axs[0, 0].set_title('Density')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
        

    etab_plt = axs[0, 1].imshow(np.log10(etab), cmap='jet')
    axs[0, 1].set_title('log10(etab)')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
        

    etap_plt = axs[0, 2].imshow(np.log10(etap), cmap='jet')
    axs[0, 2].set_title('log10(etap)')
    axs[0, 2].set_xlabel('x')
    axs[0, 2].set_ylabel('y')
        
    
    vx = s['vx']
    vy = s['vy']
    p = s['p']
    
    vx_plt = axs[1, 0].imshow(vx, cmap='jet')
    axs[1, 0].set_title('vx')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')

    vy_plt = axs[1, 1].imshow(vy, cmap='jet')
    axs[1, 1].set_title('vy')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')

    p_plt = axs[1, 2].imshow(p, cmap='jet')
    axs[1, 2].set_title('Pressure')
    axs[1, 2].set_xlabel('x')
    axs[1, 2].set_ylabel('y')
        
    if framezero:
        fig.colorbar(rho_plt, ax=axs[0, 0])
        fig.colorbar(etab_plt, ax=axs[0, 1])
        fig.colorbar(etap_plt, ax=axs[0, 2])
        fig.colorbar(vx_plt, ax=axs[1, 0])
        fig.colorbar(vy_plt, ax=axs[1, 1])
        fig.colorbar(p_plt, ax=axs[1, 2])
        plt.suptitle(f'Frame {i}')
        plt.tight_layout()
        framezero = False
    
    # Update pbar
    pbar.update(1)
    pbar.set_postfix(frame=i)

ani = anim.FuncAnimation(fig, animate, frames=len(frames), interval=100)
ani.save('animation.mp4', writer='ffmpeg', fps=10)