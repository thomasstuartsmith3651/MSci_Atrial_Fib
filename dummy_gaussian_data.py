import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from test_cross_correlation_modules import *

#%%

# Simulation parameters
num_points = 7  # Number of points in the line
distance_between_electrodes = 4 # Distance (mm)
#angle = np.radians(30)
distance_between_points = distance_between_electrodes # Distance between each point
time_steps = 4069  # Number of time steps
time = np.linspace(0, 2, time_steps)  # Time array
pulse_center = 0.3  # Center of the Gaussian pulse
pulse_width = 0.0005  # Width of the Gaussian pulse
speed = 700  # Propagation speed (mm/s)
attenuation_factor = 1 # Amplitude attenuation per point

# Create datasets
datasets = []
pulse_centers = []  # To track the pulse center positions in time
for i in range(4):
    delay = (i * distance_between_points) / speed  # Time delay for the ith point
    amplitude = np.exp(-((time - (pulse_center + delay)) ** 2) / (2 * pulse_width**2))
    amplitude *= attenuation_factor**i  # Apply attenuation
    datasets.append(amplitude)
    pulse_centers.append(pulse_center + delay)  # Store the pulse center position in time
    

# Plotting the datasets
plt.figure(figsize=(10, 6))
for i, (dataset, center) in enumerate(zip(datasets, pulse_centers)):
    plt.plot(time, dataset, label=f"Point {i+1}")
    # Annotate pulse center on the plot
    # plt.annotate(
    #     f"Center: {center:.2f}", 
    #     xy=(center, np.exp(-0.5 / pulse_width**2) * attenuation_factor**i),  # Approx. peak
    #     xytext=(center, 1.1 * np.exp(-0.5 / pulse_width**2) * attenuation_factor**i),
    #     arrowprops=dict(arrowstyle="->", color='black'),
    #     fontsize=10
    # )

# Customize the plot
plt.title("Gaussian Pulse Propagation with Pulse Centers")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# diagonally propagating gaussian pulse plane wave (propagate from top left corner to bottom right corner)

# electrode_3 = datasets[0]

# electrode_2 = datasets[1]
# electrode_7 = datasets[1]

# electrode_1 = datasets[2]
# electrode_6 = datasets[2]
# electrode_11 = datasets[2]

# electrode_0 = datasets[3]
# electrode_5 = datasets[3]
# electrode_10 = datasets[3]
# electrode_15 = datasets[3]

# electrode_4 = datasets[4]
# electrode_9 = datasets[4]
# electrode_14 = datasets[4]


# electrode_8 = datasets[5]
# electrode_13 = datasets[5]

# electrode_12 = datasets[6]

# horizontally propagating gaussian pulse plane wave (propagate from left to right)

electrode_0 = datasets[0]
electrode_1 = datasets[0]
electrode_2 = datasets[0]
electrode_3 = datasets[0]

electrode_4 = datasets[1]
electrode_5 = datasets[1]
electrode_6 = datasets[1]
electrode_7 = datasets[1]

electrode_8 = datasets[2]
electrode_9 = datasets[2]
electrode_10 = datasets[2]
electrode_11 = datasets[2]

electrode_12 = datasets[3]
electrode_13 = datasets[3]
electrode_14 = datasets[3]
electrode_15 = datasets[3]

data_arr = np.stack([electrode_0, electrode_1, electrode_2, electrode_3, electrode_4, electrode_5, electrode_6, electrode_7, electrode_8, electrode_9, electrode_10, electrode_11, electrode_12, electrode_13, electrode_14, electrode_15], axis = 1)
data_df = pd.DataFrame(data_arr)

#%%
path = '/Users/candace_chung/Desktop/Candace Chung Files/ICL/Academics/Year 4/MSci Project/code/MSci_Atrial_Fib/test_gaussian_data_BACKUP.xlsx'
data_df.to_excel(path, sheet_name="Signals")

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
alpha = 0.01  # Controls the width of the Gaussian envelope
k = 2 * np.pi / 10  # Wave number (spatial frequency)
omega = 2 * np.pi / 20  # Angular frequency
default_theta = np.radians(30)  # Default propagation angle (30 degrees)

# Grid
x = np.linspace(0, 12, 500)
y = np.linspace(0, 12, 500)
x, y = np.meshgrid(x, y)

# Time for animation
time_steps = np.linspace(0, 40, 100)

# Function to calculate the wave pulse
def gaussian_plane_wave(x, y, t, theta):
    # Components along x and y
    x_comp = x * np.cos(theta)
    y_comp = y * np.sin(theta)
    
    delay = (i * distance_between_points) / speed  # Time delay for the ith point
    # Gaussian envelope and wave
    # envelope = np.exp(-alpha * (x**2 + y**2))
    # wave = np.cos(k * (x_comp + y_comp) - omega * t)
    amplitude = np.exp(-((time - (pulse_center + delay)) ** 2) / (2 * pulse_width**2))
    return amplitude

# Function to calculate amplitude at a specific coordinate
def calculate_amplitude_at_point(x_coord, y_coord, t, theta=default_theta):
    x_comp = x_coord * np.cos(theta)
    y_comp = y_coord * np.sin(theta)
    envelope = np.exp(-alpha * (x_coord**2 + y_coord**2))
    wave = np.cos(k * (x_comp + y_comp) - omega * t)
    return envelope * wave

# Example usage of amplitude calculation
x_coord, y_coord = 10, 10  # Example coordinates
t = 10  # Example time step
amplitude = calculate_amplitude_at_point(x_coord, y_coord, t)
print(f"Amplitude at ({x_coord}, {y_coord}) at time {t}: {amplitude}")

#%%
# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_title("Gaussian Plane Wave Pulse")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Initialize plot
im = ax.imshow(np.zeros_like(x), extent=[-50, 50, -50, 50], cmap='viridis', origin='lower')

# Update function for animation
def update(frame):
    t = frame  # Current time step
    z = gaussian_plane_wave(x, y, t, default_theta)
    im.set_data(z)
    return [im]

# Create animation
ani = FuncAnimation(fig, update, frames=time_steps, interval=50, blit=True)

plt.show()

