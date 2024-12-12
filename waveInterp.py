'''Author: Hiroki Kozuki'''
#%%
'''Imports'''
import time
import numpy as np
import matplotlib.pyplot as plt
from modules import *

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
tolerance = 1e-6
max_iterations = 10000
for iteration in range(max_iterations):
    u_new = u.copy()
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            if (i, j) not in electrode_indices:  # Skip electrode points
                c_x_plus = (c[i+1, j] + c[i, j]) / 2
                c_x_minus = (c[i-1, j] + c[i, j]) / 2
                c_y_plus = (c[i, j+1] + c[i, j]) / 2
                c_y_minus = (c[i, j-1] + c[i, j]) / 2
                
                u_new[i, j] = (
                    c_x_plus * u[i+1, j] + c_x_minus * u[i-1, j]
                    + c_y_plus * u[i, j+1] + c_y_minus * u[i, j-1]
                ) / (c_x_plus + c_x_minus + c_y_plus + c_y_minus)
    # Check convergence
    if np.max(np.abs(u_new - u)) < tolerance:
        print(f"Converged after {iteration} iterations")
        break
    u = u_new

# Plot the interpolated voltage field
plt.figure(figsize=(8, 6))
plt.imshow(u, extent=[0, domain_length, 0, domain_length], origin='lower', cmap='viridis')
plt.colorbar(label='Voltage (mV)')
plt.scatter(x_elec, y_elec, c='red', label=f'Electrodes')
plt.title(f'2D Interpolated Voltage Field (t_index={time_index})')
plt.xlim(-1.0, 13.0) ; plt.xlim(-1.0, 13.0)
plt.xlabel('x [mm]') ; plt.ylabel('y [mm]')
plt.legend()
plt.show()

#%%
