"""
- Numerically solve the Eikonal model.
- Generate wave with heterogeneity.
- Output animations.
"""

import numpy as np
import skfmm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Parameters for the simulation grid
nx, ny = 500, 500     # Grid size
dx = 0.1              # Spatial resolution
tfin = 50.0            # Total simulation time (s)
c_max = 1.0           # Maximum conduction speed

# Define the conduction speed field (velocity)
C = c_max * np.ones((nx, ny))
# Introduce heterogeneity (slower speed in the center region)
C[4*nx//10:5*nx//10, 4*ny//10:5*ny//10] = c_max * 0.1

# Define initial condition: a circle at the center as the starting wavefront
phi = np.ones((nx, ny))
x, y = np.meshgrid(np.linspace(0, nx*dx, nx), np.linspace(0, ny*dx, ny))
circle_radius = 1.0  # Radius of the initial wavefront
center = (nx*dx/4, ny*dx/4)
mask = (x - center[0])**2 + (y - center[1])**2 < circle_radius**2
phi[mask] = -1  # Negative level set for the initial wavefront location

# Solve the Eikonal equation using skfmm.travel_time
arrival_time = skfmm.travel_time(phi, speed=C)

# Set up the figure for 3D and 2D animations
fig = plt.figure(figsize=(12, 6))
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

# Prepare frames for animation
max_frames = 50
arrival_time_clipped = np.clip(arrival_time, 0, np.nanmax(arrival_time))
frame_indices = np.linspace(0, arrival_time_clipped.max(), max_frames)
frames = [(arrival_time_clipped <= t) * arrival_time_clipped for t in frame_indices]

# Initialize 2D contour plot and color bar
contour = ax2d.contourf(X, Y, frames[0], cmap='viridis', levels=50)
color_bar = fig.colorbar(contour, ax=ax2d, orientation='vertical')
ax2d.set_title("2D Contour of Arrival Time")

# Function to update the plot for each frame
def update(frame):
    ax3d.clear()
    
    # Plot the 3D surface plot for arrival time
    ax3d.plot_surface(X, Y, frame, cmap='viridis', edgecolor='none')
    ax3d.set_title("3D Surface of Arrival Time")
    ax3d.set_zlim(0, np.nanmax(arrival_time))
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Arrival Time')
    
    # Update 2D contour plot without recreating color bar
    for c in ax2d.collections: 
        c.remove()  # Remove existing contours
    ax2d.contourf(X, Y, frame, cmap='viridis', levels=50)
    
    return contour

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
writer = FFMpegWriter(fps=10, metadata=dict(artist='Simulation'), bitrate=500)
ani.save('Eikonal_2D_3D_FMM_Propagation.mp4', writer=writer)

plt.close(fig)




'''
import numpy as np
import skfmm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Parameters for the simulation grid
nx, ny = 200, 200     # Grid size
dx = 0.1              # Spatial resolution
tfin = 5.0            # Total simulation time (s)
c_max = 1.0           # Maximum conduction speed

# Define the conduction speed field (velocity)
C = c_max * np.ones((nx, ny))
# Introduce heterogeneity (slower speed in the center region)
C[nx//4:3*nx//4, ny//4:3*ny//4] = c_max * 0.5

# Define initial condition: a circle at the center as the starting wavefront
phi = np.ones((nx, ny))
x, y = np.meshgrid(np.linspace(0, nx*dx, nx), np.linspace(0, ny*dx, ny))
circle_radius = 1.0  # Radius of the initial wavefront
center = (nx*dx/2, ny*dx/2)
mask = (x - center[0])**2 + (y - center[1])**2 < circle_radius**2
phi[mask] = -1  # Negative level set for the initial wavefront location

# Solve the Eikonal equation using skfmm.travel_time
arrival_time = skfmm.travel_time(phi, speed=C)

# Set up the figure for 3D and 2D animations
fig = plt.figure(figsize=(12, 6))
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)
X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

# Prepare frames for animation
max_frames = 50
arrival_time_clipped = np.clip(arrival_time, 0, np.nanmax(arrival_time))
frame_indices = np.linspace(0, arrival_time_clipped.max(), max_frames)
frames = [(arrival_time_clipped <= t) * arrival_time_clipped for t in frame_indices]

# Function to update the plot for each frame
def update(frame):
    ax3d.clear()
    ax2d.clear()
    
    # Plot the 3D surface plot for arrival time
    ax3d.plot_surface(X, Y, frame, cmap='viridis', edgecolor='none')
    ax3d.set_title("3D Surface of Arrival Time")
    ax3d.set_zlim(0, np.nanmax(arrival_time))
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Arrival Time')
    
    # Plot the 2D contour plot for arrival time
    contour = ax2d.contourf(X, Y, frame, cmap='viridis', levels=50)
    ax2d.set_title("2D Contour of Arrival Time")
    fig.colorbar(contour, ax=ax2d, orientation='vertical')
    
    return contour

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
writer = FFMpegWriter(fps=10, metadata=dict(artist='Simulation'), bitrate=500)
ani.save('Eikonal_2D_3D_FMM_Propagation.mp4', writer=writer)

plt.close(fig)

'''