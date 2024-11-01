"""
- Numerically solve the monodomain Aliev Panfilov model used in EP-PINNS paper with the same parameter values.
- Uses the Euler Method.
- Generate a spiral wave with heterogeneity.
- Output animations.
- Code adapted from Matlab to Python.  
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from numba import jit, prange

# Aliev-Panfilov model parameters
D0 = 0.05  # Base diffusion coefficient
Dfac = 0.1  # Diffusion heterogeneity factor for the spiral
a = 0.01
b = 0.15
k = 8.0
epsilon_0 = 0.002
mu1 = 0.2
mu2 = 0.3

# Grid and time parameters
nx, ny = 60, 60  # Grid size
dx = 0.1  # Space step (mm)
dt = 0.01  # Time step (s)
tfin = 100.0  # Simulation end time (s)
time_steps = int(tfin / dt)  # Number of time steps

# Stimulus parameters
stimulus_positions = [(20, 20)]
stimulus_times = [0.5]
stimulus_duration = 0.01
stimulus_amplitude = 1.0

# Initialize u (potential) and v (recovery) arrays
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))

# Set up a region with modified diffusion for heterogeneity
D = D0 * np.ones((nx, ny))
D[4*nx//10:5*nx//10, 4*ny//10:5*ny//10] *= Dfac

# Function for periodic boundary conditions
@jit(nopython=True)
def apply_periodic_boundary(arr):
    arr[0, :] = arr[-2, :]
    arr[-1, :] = arr[1, :]
    arr[:, 0] = arr[:, -2]
    arr[:, -1] = arr[:, 1]

# Epsilon function for the Aliev-Panfilov model
@jit(nopython=True)
def epsilon(u, v):
    return epsilon_0 + (mu1 * v) / (u + mu2)

# Function to apply external stimuli
@jit(nopython=True)
def apply_stimuli(u, t, dt, stimulus_times, stimulus_positions):
    for stim_time in stimulus_times:
        if stim_time <= t < stim_time + stimulus_duration:
            for (i, j) in stimulus_positions:
                u[i-1:i+1, j-1:j+1] += stimulus_amplitude

# Runge-Kutta integration for the simulation
@jit(nopython=True, parallel=True)
def runge_kutta_step(u, v, D, t, periodic=False, stimulus_times=None, stimulus_positions=None):
    u_old, v_old = u.copy(), v.copy()

    laplacian = (
        D[1:-1, 1:-1] * (
            u_old[:-2, 1:-1] + u_old[2:, 1:-1] + 
            u_old[1:-1, :-2] + u_old[1:-1, 2:] - 
            4 * u_old[1:-1, 1:-1]
        ) / (dx**2)
    )

    du_dt = (
        laplacian + k * u_old[1:-1, 1:-1] *
        (u_old[1:-1, 1:-1] - a) * (1 - u_old[1:-1, 1:-1]) - 
        u_old[1:-1, 1:-1] * v_old[1:-1, 1:-1]
    )

    dv_dt = (
        epsilon(u_old[1:-1, 1:-1], v_old[1:-1, 1:-1]) *
        (-v_old[1:-1, 1:-1] - k * u_old[1:-1, 1:-1] *
        (u_old[1:-1, 1:-1] - b - 1))
    )

    u[1:-1, 1:-1] += dt * du_dt
    v[1:-1, 1:-1] += dt * dv_dt

    if periodic:
        apply_periodic_boundary(u)
        apply_periodic_boundary(v)

    # Apply external stimuli if within stimulus time
    apply_stimuli(u, t, dt, stimulus_times, stimulus_positions)

# Preallocate array for u at specific time steps
all_u_at_t = np.zeros((int(tfin / 0.1), nx * ny + 1))
frame_idx = 0

frames = []

for step in prange(time_steps):
    t = step * dt
    runge_kutta_step(u, v, D, t, stimulus_times = np.array(stimulus_times), stimulus_positions = stimulus_positions)

    # Store current u values every 0.1 seconds
    if step % int(0.1 / dt) == 0:
        all_u_at_t[frame_idx, 0] = t
        all_u_at_t[frame_idx, 1:] = u.flatten()
        frame_idx += 1

    # Save frames for animation every 50 steps
    if step % 50 == 0:
        frames.append(u.copy())

# Create DataFrame and save CSV outside of computation loop for efficiency
columns = ['Time'] + [f'u_{i}_{j}' for i in range(nx) for j in range(ny)]
u_df = pd.DataFrame(all_u_at_t[:frame_idx], columns=columns)
u_df.to_csv('all_u_at_t_Euler.csv', index=False)

# Create plot layout for 3D and 2D contour animations
fig = plt.figure(figsize=(12, 6))
ax3d = fig.add_subplot(121, projection='3d')
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
ax2d = fig.add_subplot(122)
contour = ax2d.contourf(X, Y, frames[0], cmap='viridis')
plt.colorbar(contour, ax=ax2d)
ax2d.set_title("2D Contour of Potential (u)")

# Update function for animation
def update(frame):
    ax3d.clear()
    ax3d.plot_surface(X, Y, frame, cmap='viridis')
    ax3d.set_zlim(0, 3)
    ax3d.set_title("3D Surface of Potential (V$_m$)")
    ax2d.clear()
    contour = ax2d.contourf(X, Y, frame, cmap='viridis')
    ax2d.set_title("2D Contour of Potential (V$_m$)")
    return contour

# Save animation as .mp4 file
ani = FuncAnimation(fig, update, frames=frames, interval=50)
writer = FFMpegWriter(fps=15, metadata=dict(artist='Simulation'), bitrate=500)
ani.save('Optimised_AP_Stimulus_Euler.mp4', writer=writer)

plt.close(fig)

# %%
