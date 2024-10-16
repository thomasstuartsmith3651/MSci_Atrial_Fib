"""
- Numerically solve the monodomain Aliev Panfilov model used in EP-PINNS paper with the same parameter values.
- Generate a spiral wave heterogeneity.
- Output animations.
- Code adapted from Matlab to Python.  
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Aliev-Panfilov model parameters
D = 0.05  # Diffusion coefficient
a = 0.01
b = 0.15
k = 8.0
epsilon_0 = 0.002
mu1 = 0.2
mu2 = 0.3

# Grid and time parameters
nx, ny = 100, 100  # Grid size
dx = 0.1           # Space step (cm)
dt = 0.01 #0.01          # Time step (s)
tfin = 20.0         # Simulation end time (s)
time_steps = int(tfin / dt)  # Number of time steps

# Initialize u (potential) and v (recovery) arrays
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))

# Initial conditions to generate a spiral wave
u[:50, 50:] = 1.0  # Trigger wave in one quadrant
v[30:70, 30:70] = 0.5  # Refractory region to disrupt symmetry

# Function for periodic boundary conditions
def apply_periodic_boundary(arr):
    arr[0, :] = arr[-2, :]
    arr[-1, :] = arr[1, :]
    arr[:, 0] = arr[:, -2]
    arr[:, -1] = arr[:, 1]

# Epsilon function for the Aliev-Panfilov model
def epsilon(u, v):
    return epsilon_0 + (mu1 * v) / (u + mu2)

# Runge-Kutta integration for the simulation
def runge_kutta_step(u, v):
    u_old, v_old = u.copy(), v.copy()
    
    laplacian = (
        D * (
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
    
    apply_periodic_boundary(u)
    apply_periodic_boundary(v)

# Store frames for the animation
frames = []

for step in range(time_steps):
    runge_kutta_step(u, v)
    
    # Save frames for animation every 50 steps
    if step % 50 == 0:
        frames.append(u.copy())

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
    ax3d.set_zlim(0, 1)
    ax3d.set_title("3D Surface of Potential (u)")
    
    # Update 2D contour plot
    ax2d.clear()
    contour = ax2d.contourf(X, Y, frame, cmap='viridis')
    ax2d.set_title("2D Contour of Potential (u)")
    return contour

# Save animation as .mp4 file
ani = FuncAnimation(fig, update, frames=frames, interval=50)
writer = FFMpegWriter(fps=15, metadata=dict(artist='Simulation'), bitrate=500)
ani.save('AP_2D_Spiral.mp4', writer=writer)

plt.close(fig)
