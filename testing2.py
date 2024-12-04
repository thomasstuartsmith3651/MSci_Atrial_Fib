#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:49:32 2024

@author: candace_chung
"""

import numpy as np
import matplotlib.pyplot as plt
from test_cross_correlation_modules import *

#%%

# Simulation parameters
num_points = 4  # Number of points in the line
distance_between_points = 4  # Distance between each point
time_steps = 1000  # Number of time steps
time = np.linspace(0, 20, time_steps)  # Time array
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
    plt.plot(time, dataset, label=f"Point {i+1} (Distance {i * distance_between_points})")
    # Annotate pulse center on the plot
    plt.annotate(
        f"Center: {center:.2f}", 
        xy=(center, np.exp(-0.5 / pulse_width**2) * attenuation_factor**i),  # Approx. peak
        xytext=(center, 1.1 * np.exp(-0.5 / pulse_width**2) * attenuation_factor**i),
        arrowprops=dict(arrowstyle="->", color='black'),
        fontsize=10
    )

# Customize the plot
plt.title("Gaussian Pulse Propagation with Pulse Centers")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

# Output datasets
for i, dataset in enumerate(datasets):
    print(f"Dataset for Point {i+1} (First 10 Values):\n{dataset[:10]}\n")


#%%

data = "ElectrogramData.xlsx"
A = AnalyseDataExcel(data)
corr_mode = "full"
peak_num = 0
corr_threshold = 0.65
dist = 12

signal1 = datasets[0]
signal2 = datasets[3]

plt.plot(time, signal1)
plt.plot(time, signal2)
plt.show()
#%%
RXY, index_delays = A.simpleCorrelate(signal1, signal2, corr_mode)

plt.plot(index_delays, RXY)
plt.show()
#%%
minTimeDelay = dist/2
best_timeDelay, max_RXY = A.maxRXY_timeDelay(RXY, index_delays, minTimeDelay)

print(best_timeDelay, max_RXY)

velocity_vector, max_RXY = A.velocity(signal1, signal2, corr_mode, 2, dist)

print(velocity_vector, max_RXY)

#%%

array = [1,2,3,4,5,6,7]
array[-2:-1]