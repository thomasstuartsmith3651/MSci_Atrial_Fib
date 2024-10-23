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
nx, ny = 60, 60  # Grid size - 50 mm x 50 mm (500 x 500). # for 6mm x 6mm, use 60, 60 --> 600 x 600 mesh.
dx = 0.1           # Space step (mm) - spatial cell size 
dt = 0.01          # Time step (s)
tfin = 100.0         # Simulation end time (s)
time_steps = int(tfin / dt)  # Number of time steps

# Stimulus parameters
stimulus_positions = [(20, 20)] # [(10, 10), (80, 80)] # Example positions for stimuli
stimulus_times = [0.5]  # [0.5, 2.0] # Stimuli applied at these seconds
stimulus_duration = 0.01  # Duration of each stimulus
stimulus_amplitude = 1.0  # Strength of the stimulus

# Initialize u (potential) and v (recovery) arrays
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))

# Set up a region with modified diffusion for heterogeneity
D = D0 * np.ones((nx, ny))
D[4*nx//10:5*nx//10, 4*ny//10:5*ny//10] *= Dfac  # Lower diffusion in central region

# Initial conditions to generate a spiral wave (optional)
'''
u[:50, 50:] = 1.0  # Trigger initial wave in one quadrant
v[30:70, 30:70] = 0.5  # Initial refractory region
'''
u[:50, 50:] = 0.0  # Trigger initial wave in one quadrant
v[30:70, 30:70] = 0.0  # Initial refractory region

# Function for periodic boundary conditions
def apply_periodic_boundary(arr):
    arr[0, :] = arr[-2, :]
    arr[-1, :] = arr[1, :]
    arr[:, 0] = arr[:, -2]
    arr[:, -1] = arr[:, 1]

# Epsilon function for the Aliev-Panfilov model
def epsilon(u, v):
    return epsilon_0 + (mu1 * v) / (u + mu2)

# Function to apply external stimuli
def apply_stimuli(u, t, dt):
    for stim_time in stimulus_times:
        if stim_time <= t < stim_time + stimulus_duration:
            for (i, j) in stimulus_positions:
                u[i-1:i+1, j-1:j+1] += stimulus_amplitude  # u[i-1:i+2, j-1:j+2] += stimulus_amplitude

all_u_at_t = []
# Runge-Kutta integration for the simulation
def runge_kutta_step(u, v, t, periodic = False):
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
    
    if periodic == True:
        apply_periodic_boundary(u)
        apply_periodic_boundary(v)
    
    # Apply external stimuli if within stimulus time
    apply_stimuli(u, t, dt)
    
    # Store current u values at every 1 time step --> 100 rows.
    if t % 0.1 == 0:
        u_flat = u.flatten()
        all_u_at_t.append([t] + u_flat.tolist())

# Store frames for the animation
frames = []

for step in range(time_steps):
    t = step * dt
    runge_kutta_step(u, v, t)
    
    # Save frames for animation every 50 steps
    if step % 50 == 0:
        frames.append(u.copy())

# Create a DataFrame with columns for time and u values for each spatial point
columns = ['Time'] + [f'u_{i}_{j}' for i in range(nx) for j in range(ny)]
u_df = pd.DataFrame(all_u_at_t, columns = columns)
u_df.to_csv('all_u_at_t_Euler.csv', index=False)  # Save to CSV file

# Create plot layout for 3D and 2D contour animations
fig = plt.figure(figsize=(12, 6))

# 3D plot setup
ax3d = fig.add_subplot(121, projection='3d')
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

# 2D contour plot setup
ax2d = fig.add_subplot(122)
contour = ax2d.contourf(X, Y, frames[0], cmap='viridis')
plt.colorbar(contour, ax=ax2d)
ax2d.set_title("2D Contour of Potential (u)")

# Update function for animation
def update(frame):
    # Update 3D plot
    ax3d.clear()
    ax3d.plot_surface(X, Y, frame, cmap='viridis')
    ax3d.set_zlim(0, 3)
    ax3d.set_title("3D Surface of Potential (V$_m$)")
    
    # Update 2D contour plot
    ax2d.clear()
    contour = ax2d.contourf(X, Y, frame, cmap='viridis')
    ax2d.set_title("2D Contour of Potential (V$_m$)")
    return contour

# Save animation as .mp4 file
ani = FuncAnimation(fig, update, frames=frames, interval=50)
writer = FFMpegWriter(fps=15, metadata=dict(artist='Simulation'), bitrate=500)
ani.save('AP_HeteroSpiral_Stimulus_Euler.mp4', writer=writer)

plt.close(fig)

# %%
