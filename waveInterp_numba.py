'''Author: Hiroki Kozuki'''
#%%
'''Imports'''
import time
import numpy as np
import matplotlib.pyplot as plt
from modules import *

from numba import jit
from mpl_toolkits.mplot3d import Axes3D

#%%
'''Load data'''

data = "ElectrogramData.xlsx"

start = time.time()

L = loadData(data, n = 600) # Load electrogram data
print(L.coordinates().shape) # (16, 2) - x,y positions of electrodes 
print(L.ele_signals().shape) # (2201, 16) - 2201 voltage data at 16 electrodes.

time_index = 1740 # row 1742 in excel

print(L.ele_signals().iloc[time_index].shape) # (16, ) 
print(L.ele_signals().iloc[time_index])
print(L.time_data().shape)   # (2201, ) - 2201 time points. 

end = time.time()
print("time take (new)", end - start)

# %%
'''Traditional method: using velocity map to interpolate potential field using wave equation:
   Accelerated with Numba (multithreading on CPU), more intuitive than Jax'''

# Parameters
grid_size = 600  # 600 x 600 grid
domain_length = 12.0  # 12 mm x 12 mm domain
dx = domain_length / grid_size
dy = domain_length / grid_size

# Electrode positions (4x4 grid, separated by 4 mm)
electrode_spacing = 4.0  # mm
electrode_indices = [
    (int(i * grid_size / 3), int(j * grid_size / 3))
    for i in range(0, 4) for j in range(0, 4)
]
print(electrode_indices) # [(0, 0), (0, 200), (0, 400), (0, 600), (200, 0), (200, 200), ..., (600, 400), (600, 600)]


'''For a single time-step'''

# Voltage values at electrode positions
electrode_values = np.array(L.ele_signals().iloc[time_index]) # convert to numpy array (easier to work with)
print(electrode_values)

# Initialize voltage grid
u = np.zeros((grid_size + 1, grid_size + 1))
print(u.shape)
print(u)

# Set electrode voltages as hard constraints
for (i, j), value in zip(electrode_indices, electrode_values):
    u[i, j] = value

# Velocity profile c(x, y) (e.g., spatially varying velocity)
c = 1000 * np.ones((grid_size + 1, grid_size + 1))  # Uniform velocity , 1 m/s = 1000 mm/s
x = np.linspace(0, domain_length, grid_size)
y = np.linspace(0, domain_length, grid_size)
X, Y = np.meshgrid(x, y)
# For plotting electrode points:
x_elec = np.array(L.x_position()) ; y_elec = np.array(L.y_position()) 


# Jacobi iteration to solve for u(x, y)
@jit(nopython=True)
def jacobi_iter(u, c, electrode_indices, electrode_values, tolerance, max_iterations):
    grid_size = u.shape[0] - 1
    c_x_plus = (c[1:, :] + c[:-1, :]) / 2
    c_x_minus = c_x_plus
    c_y_plus = (c[:, 1:] + c[:, :-1]) / 2
    c_y_minus = c_y_plus

    for iteration in range(max_iterations):
        u_new = u.copy()
        for i in range(1, grid_size):
            for j in range(1, grid_size):
                if (i, j) not in electrode_indices: # Skip electrode points
                    u_new[i, j] = (
                        c_x_plus[i, j] * u[i + 1, j] + c_x_minus[i - 1, j] * u[i - 1, j] +
                        c_y_plus[i, j] * u[i, j + 1] + c_y_minus[i, j - 1] * u[i, j - 1]
                    ) / (c_x_plus[i, j] + c_x_minus[i - 1, j] + c_y_plus[i, j] + c_y_minus[i, j - 1])

        for (i, j), value in zip(electrode_indices, electrode_values):
            u_new[i, j] = value

        # Check convergence
        if np.max(np.abs(u_new - u)) < tolerance:
            return u_new, iteration
        u = u_new
    return u, max_iterations # output max_iter if method doesn't reach convergence earlier. 

# Run Jacobi iteration
tolerance = 1e-6
max_iterations = 100000
start = time.time()
u, iteration = jacobi_iter(u, c, electrode_indices, electrode_values, tolerance, max_iterations)
end = time.time()
print(f"Time taken (Numba): {end - start:.2f} seconds")
print(f"Converged in {iteration} iterations")
print(u.shape)
print(u)

# Plot the interpolated voltage field in 2D
plt.figure(figsize=(8, 6))
plt.imshow(u, extent=[0, domain_length, 0, domain_length], origin='lower', cmap='viridis')
plt.colorbar(label='Voltage (mV)')
plt.scatter(x_elec, y_elec, electrode_values, c='red', label='Electrodes')
# plt.title(f'Interpolated Voltage Field (tol={tolerance}, max_iter={max_iterations})')
plt.title(f'2D Interpolated Voltage Field (t_index={time_index})', fontsize=16)
plt.xlim(-1.0, 13.0) ; plt.ylim(-1.0, 13.0)
plt.xlabel('x [mm]') ; plt.ylabel('y [mm]')
plt.legend()
plt.show()

# Plot the interpolated voltage field in 3D
# Create a figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Create a meshgrid for plotting
X, Y = np.meshgrid(
    np.linspace(0, domain_length, grid_size + 1), 
    np.linspace(0, domain_length, grid_size + 1)
)
# Plot the surface
surf = ax.plot_surface(
    X, Y, u, cmap='viridis', edgecolor='none', alpha=0.9
)
# Customize the plot
ax.set_title(f'3D Interpolated Voltage Field (t_index={time_index})', fontsize=16)
ax.set_xlabel('X [mm]', fontsize=12)
ax.set_ylabel('Y [mm]', fontsize=12)
ax.set_zlabel('Voltage [mV]', fontsize=12)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Voltage (mV)')
# Add scatter points for electrode positions
ax.scatter(x_elec, y_elec, np.zeros_like(x_elec), color='red', label='Electrodes', s=50)
# Show legend and plot
ax.legend()
plt.show()

# %%
