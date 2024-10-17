"""
- Numerically solve the monodomain Aliev Panfilov model used in EP-PINNS paper with the same parameter values.
- Uses RK4 method. 
- Generate a spiral wave with heterogeneity.
- Output animations.
- Code adapted from Matlab to Python.  
"""
import numpy as np
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
nx, ny = 500, 500  # Grid size - 50 mm x 50 mm # for 6mm x 6mm, use 60, 60 --> 600 x 600 mesh.
dx = 0.1           # Space step (mm) - spatial cell size 
dt = 0.01          # Time step (s)
tfin = 100.0         # Simulation end time (s)
time_steps = int(tfin / dt)  # Number of time steps

# Stimulus parameters
stimulus_positions = [(200, 200)] # [(10, 10), (80, 80)] # Example positions for stimuli
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
                u[i-1:i+2, j-1:j+2] += stimulus_amplitude

# RK4 Step Function
def rk4_step(f, u, v, laplacian, dt):
    k1_u, k1_v = f(u, v, laplacian)
    k2_u, k2_v = f(u + 0.5*dt*k1_u, v + 0.5*dt*k1_v, laplacian)
    k3_u, k3_v = f(u + 0.5*dt*k2_u, v + 0.5*dt*k2_v, laplacian)
    k4_u, k4_v = f(u + dt*k3_u, v + dt*k3_v, laplacian)
    
    u_next = u + (dt/6) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
    v_next = v + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return u_next, v_next

# System function for the Aliev-Panfilov model
def aliev_panfilov_system(u, v, laplacian):
    du_dt = (
        laplacian + k * u * (u - a) * (1 - u) - u * v
    )
    dv_dt = (
        epsilon(u, v) * (-v - k * u * (u - b - 1))
    )
    return du_dt, dv_dt

# Simulation loop with 4th-order Runge-Kutta
frames = []
for step in range(time_steps):
    t = step * dt
    
    u_old, v_old = u.copy(), v.copy()
    
    # Compute Laplacian with variable diffusion D
    laplacian = (
        D[1:-1, 1:-1] * (
            u_old[:-2, 1:-1] + u_old[2:, 1:-1] + 
            u_old[1:-1, :-2] + u_old[1:-1, 2:] - 
            4 * u_old[1:-1, 1:-1]
        ) / (dx**2)
    )
    
    # Perform RK4 step
    u_next, v_next = rk4_step(lambda u, v, lap: aliev_panfilov_system(u, v, lap), 
                              u_old[1:-1, 1:-1], v_old[1:-1, 1:-1], laplacian, dt)
    
    # Update u and v arrays
    u[1:-1, 1:-1] = u_next
    v[1:-1, 1:-1] = v_next

    apply_periodic_boundary(u)
    apply_periodic_boundary(v)
    apply_stimuli(u, t, dt)

    # Save frames for animation every 50 steps
    if step % 50 == 0:
        frames.append(u.copy())

# Animation setup (same as before)
fig = plt.figure(figsize=(12, 6))
ax3d = fig.add_subplot(121, projection='3d')
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
ax2d = fig.add_subplot(122)
contour = ax2d.contourf(X, Y, frames[0], cmap='viridis')
plt.colorbar(contour, ax=ax2d)
ax2d.set_title("2D Contour of Potential (u)")

def update(frame):
    ax3d.clear()
    ax3d.plot_surface(X, Y, frame, cmap='viridis')
    ax3d.set_zlim(0, 1)
    ax3d.set_title("3D Surface of Potential (V$_m$)")
    ax2d.clear()
    contour = ax2d.contourf(X, Y, frame, cmap='viridis')
    ax2d.set_title("2D Contour of Potential (V$_m$)")
    return contour

ani = FuncAnimation(fig, update, frames=frames, interval=50)
writer = FFMpegWriter(fps=15, metadata=dict(artist='Simulation'), bitrate=500)
ani.save('AP_HeteroSpiral_Stimulus_RK4.mp4', writer=writer)

plt.close(fig)