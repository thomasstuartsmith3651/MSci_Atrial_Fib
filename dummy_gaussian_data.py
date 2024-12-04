import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from test_cross_correlation_modules import *

#%%

# Simulation parameters
num_points = 7  # Number of points in the line
distance_between_electrodes = 4
distance_between_points = distance_between_electrodes * np.sqrt(2)/2  # Distance between each point
time_steps = 2000  # Number of time steps
time = np.linspace(0, 40, time_steps)  # Time array
pulse_center = 5  # Center of the Gaussian pulse
pulse_width = 0.5  # Width of the Gaussian pulse
speed = 1.0  # Propagation speed (units per time unit)
attenuation_factor = 1 # Amplitude attenuation per point

# Create datasets
datasets = []
pulse_centers = []  # To track the pulse center positions in time
for i in range(num_points):
    delay = (i * distance_between_points) / speed  # Time delay for the ith point
    amplitude = np.exp(-((time - (pulse_center + delay)) ** 2) / (2 * pulse_width**2))
    amplitude *= attenuation_factor**i  # Apply attenuation
    datasets.append(amplitude)
    pulse_centers.append(pulse_center + delay)  # Store the pulse center position in time
    

# Plotting the datasets
plt.figure(figsize=(10, 6))
for i, (dataset, center) in enumerate(zip(datasets, pulse_centers)):
    plt.plot(time, dataset, label=f"Point {i+1} (Time {i * distance_between_points})")
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
electrode_3 = datasets[0]

electrode_2 = datasets[1]
electrode_7 = datasets[1]

electrode_1 = datasets[2]
electrode_6 = datasets[2]
electrode_11 = datasets[2]

electrode_0 = datasets[3]
electrode_5 = datasets[3]
electrode_10 = datasets[3]
electrode_15 = datasets[3]

electrode_4 = datasets[4]
electrode_9 = datasets[4]
electrode_14 = datasets[4]

electrode_8 = datasets[5]
electrode_13 = datasets[5]

electrode_12 = datasets[6]

data_arr = np.stack([electrode_0, electrode_1, electrode_2, electrode_3, electrode_4, electrode_5, electrode_6, electrode_7, electrode_8, electrode_9, electrode_10, electrode_11, electrode_12, electrode_13, electrode_14, electrode_15], axis = 1)
data_df = pd.DataFrame(data_arr)

#%%
path = '/Users/candace_chung/Desktop/Candace Chung Files/ICL/Academics/Year 4/MSci Project/code/MSci_Atrial_Fib/test_gaussian_data_BACKUP.xlsx'
data_df.to_excel(path, sheet_name="Signals")
