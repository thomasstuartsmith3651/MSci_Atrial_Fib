"""
- Numerically solve the monodomain Aliev Panfilov model used in EP-PINNS paper with the same parameter values.
- Output animations.
- Code adapted from Matlab to Python.  
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Model parameters (adapted based on common values for Aliev-Panfilov)
D = 0.05         # Diffusion coefficient
a = 0.01         # Aliev-Panfilov model parameter a
b = 0.15         # Aliev-Panfilov model parameter b
k = 8.0          # Aliev-Panfilov model parameter k
epsilon_0 = 0.002
mu1 = 0.2
mu2 = 0.3

# Simulation parameters
nx, ny = 100, 100  # Grid size
dx = 0.1           # Space step (cm)
dt = 0.01          # Time step (s)
total_time = 5.0   # Total simulation time (s)
time_steps = int(total_time / dt)  # Number of time steps

# Initialize u and v (potential and recovery variables)
u = np.zeros((nx, ny))
v = np.zeros((nx, ny))

# Set initial conditions (small square pulse in the center)
u[45:55, 45:55] = 1.0

# Periodic boundary function
def apply_periodic_boundary(arr):
    arr[0, :] = arr[-2, :]
    arr[-1, :] = arr[1, :]
    arr[:, 0] = arr[:, -2]
    arr[:, -1] = arr[:, 1]

# Define epsilon(u, v) function for Aliev-Panfilov model
def epsilon(u, v):
    return epsilon_0 + (mu1 * v) / (u + mu2)

# Runge-Kutta integration for each time step
def runge_kutta_step(u, v):
    u_old, v_old = u.copy(), v.copy()
    
    laplacian = (
        u_old[:-2, 1:-1] + u_old[2:, 1:-1] + 
        u_old[1:-1, :-2] + u_old[1:-1, 2:] - 
        4 * u_old[1:-1, 1:-1]
    ) / (dx**2)
    
    du_dt = (
        D * laplacian + k * u_old[1:-1, 1:-1] * 
        (u_old[1:-1, 1:-1] - a) * (1 - u_old[1:-1, 1:-1]) - 
        u_old[1:-1, 1:-1] * v_old[1:-1, 1:-1]
    )
    
    dv_dt = (
        epsilon(u_old[1:-1, 1:-1], v_old[1:-1, 1:-1]) *
        (-v_old[1:-1, 1:-1] - k * u_old[1:-1, 1:-1] * 
        (u_old[1:-1, 1:-1] - b - 1))
    )

    # Runge-Kutta updates
    u[1:-1, 1:-1] += dt * du_dt
    v[1:-1, 1:-1] += dt * dv_dt
    
    # Apply periodic boundary conditions
    apply_periodic_boundary(u)
    apply_periodic_boundary(v)

# Prepare frames for animation
frames = []

for step in range(time_steps):
    runge_kutta_step(u, v)
    
    # Save frames for animation every 50 steps for performance
    if step % 50 == 0:
        frames.append(u.copy())

# Create the 3D animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

def update(frame):
    ax.clear()
    ax.plot_surface(X, Y, frame, cmap='viridis')
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Potential (u)')
    plt.title("Aliev-Panfilov Model Simulation")

# Save the animation as an mp4 file
ani = FuncAnimation(fig, update, frames=frames, interval=50)
writer = FFMpegWriter(fps=15, metadata=dict(artist='Simulation'), bitrate=1800)
ani.save('AP_2D.mp4', writer=writer)

plt.close(fig)
