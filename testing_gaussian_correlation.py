#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:48:09 2024

@author: candace_chung
"""


import time
import pandas as pd
import numpy as np
from scipy import integrate
from scipy.signal.windows import kaiser
from scipy.signal import correlate
import scipy.signal as sps
import matplotlib.pyplot as plt
from test_cross_correlation_modules import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
import pywt
from scipy.stats import zscore

"""
For some reason my laptop can't find this file in my local directory so I need the line below to import the file.
"""
import sys 
sys.path.append('/Users/candace_chung/Desktop/Candace Chung Files/ICL/Academics/Year 4/MSci Project/code/MSci_Atrial_Fib')

#%%

data = "test_gaussian_data_20.xlsx"
L = LoadDataExcel(data)
S = L.ele_signals()
times = L.time_data()
coord = L.coordinates()

plt.plot(times, S[3], label = "electrode 3")
#plt.plot(times, S[2], label = "electrode 2")
plt.plot(times, S[6], label = "electrode 6")
#plt.plot(times, S[5], label = "electrode 5")
plt.plot(times, S[9], label = "electrode 9")
#plt.plot(times, S[8], label = "electrode 8")
plt.plot(times, S[12], label = "electrode 12")
plt.legend()
plt.show()


#%%
A = AnalyseDataExcel(data)
peak_num = 0
num_vectors = 3
v_min = 0.4
v_max = 2
corr_threshold = 0.75

ele_1 = 2
ele_2 = 7
e1 = S[ele_1]
e2 = S[ele_2]
peaks1, _ = sps.find_peaks(e1, height = 0.9)
peaks2, _ = sps.find_peaks(e2, height = 0.9)

peak_time_e1 = times[peaks1][0]
peak_time_e2 = times[peaks2][0]
time_diff = peak_time_e2 - peak_time_e1
print(peak_time_e1, peak_time_e2, time_diff)

plt.plot(times, e1, label = "electrode 2")
plt.plot(times, e2, label = "electrode 7")
plt.legend()
plt.show()

#%%

ind_shifts = A.findEGMPeak(e1)

window_offset = ind_shifts[peak_num]

e1_w, e2_w = A.windowSignal(e1, e2, window_offset)

plt.plot(times, e1, label = "electrode 2")
plt.plot(times, e1_w, label = "electrode 2 windowed")
plt.plot(times, e2, label = "electrode 3")
plt.plot(times, e2_w, label = "electrode 3 windowed")
#plt.plot(times, padded_kaiser, label = "kaiser window")
#plt.vlines((250 - 407/2) * 20/1000, 0, 1, color = 'black', label = "min time of window")
#plt.vlines((250 + 407/2) * 20/1000, 0, 1, color = 'black', label = "max time of window")
plt.legend()
plt.show()

#%%
# Perform correlation
RXY, index_delays = A.simpleCorrelate(e1_w, e2_w)
peaksRXY, _ = sps.find_peaks(RXY, height = 0.9)
print(peaksRXY)
peak_index_e1 = index_delays[peaksRXY[0]]
print(peak_index_e1)
peakRXY_val = RXY[peaksRXY[0]]
print(peak_index_e1, peakRXY_val)
plt.plot(index_delays, RXY)
plt.plot(peak_index_e1, peakRXY_val, "o")
plt.show()

#%%

# Calculate minimum time delay based on velocity
minTimeDelay = np.linalg.norm(coord[ele_2] - coord[ele_1]) * 0.001 / v_max
maxTimeDelay = np.linalg.norm(coord[ele_2] - coord[ele_1]) * 0.001 / v_min
neg_time_x = -minTimeDelay
neg_time_x1 = -maxTimeDelay
pos_time_x = minTimeDelay
pos_time_x1 = maxTimeDelay
time_y = np.linspace(np.min(RXY), np.max(RXY), 50)

# Find the best time delay and its corresponding RXY value

best_timeDelay, max_RXY = A.maxRXY_timeDelay(RXY, index_delays, minTimeDelay, maxTimeDelay, corr_threshold)
#print(best_timeDelay, max_RXY)
print("CALCULATED", best_timeDelay, "EXPECTED", time_diff)

# Plotting
plt.plot(index_delays / (2034.5), RXY, label="Cross-Correlation RXY")
#plt.axvline(neg_time_x, color='orange', linestyle="--", label="Negative Min Time Delay")
#plt.axvline(pos_time_x, color='green', linestyle="--", label="Positive Min Time Delay")
plt.plot(best_timeDelay, max_RXY, 'ro', label="Best Time Delay")

# Shade the excluded regions
plt.axvspan(neg_time_x1, neg_time_x, color='orange', alpha=0.3, label = "Valid Negative Time Delay")
plt.axvspan(pos_time_x, pos_time_x1, color='green', alpha=0.3, label = "Valid Positive Time Delay")

# Add title and labels
plt.title(f"Electrode {ele_1} and {ele_2}")
plt.xlabel("Time Delay (s)")
plt.ylabel("RXY")

# Add legend
plt.legend()

# Show the plot
plt.show()

print(best_timeDelay)
#%%
#velocity_vector, max_RXY = A.electrodePairVelocity(ele_1, ele_2, v_min, v_max, ind_shifts[peak_num], corr_threshold)

#print(ele_1, ele_2, velocity_vector, max_RXY, best_timeDelay)

velocity_vector, _ = A.electrodePairVelocity(2, 3, v_min, v_max, ind_shifts[peak_num], corr_threshold)
velocity_vector2, _ = A.electrodePairVelocity(2, 6, v_min, v_max, ind_shifts[peak_num], corr_threshold)
print("CHECK", velocity_vector, velocity_vector2)
#%%
v_guess = A.guessVelocity_LSQ(0, 6, 9, v_min, v_max, peak_num, corr_threshold)

#%%

data = "test_gaussian_data_70.xlsx"

A = AnalyseDataExcel(data)
peak_num = 0
num_vectors = 0.9
v_min = 0.4
v_max = 2
corr_threshold = 0.75

# Your existing code to compute vectors and origins
vectors, origins = A.velocityGuessMap(v_min, v_max, peak_num, corr_threshold)

x_origins = [x[0] for x in origins]
y_origins = [x[1] for x in origins]
x_vectors = [x[0] for x in vectors]
y_vectors = [x[1] for x in vectors]

# Calculate vector magnitudes
magnitudes = [np.sqrt(vx**2 + vy**2) for vx, vy in vectors]

# Define electrode positions

electrode_x = [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]
electrode_y = [0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12]

# Convert electrode positions into a grid
grid_x = np.unique(electrode_x)
grid_y = np.unique(electrode_y)

# Draw horizontal grid lines
for y in grid_y:
    plt.plot([grid_x[0], grid_x[-1]], [y, y], color="black", linewidth=0.5)

# Draw vertical grid lines
for x in grid_x:
    plt.plot([x, x], [grid_y[0], grid_y[-1]], color="black", linewidth=0.5)

# Draw diagonal lines (top-left to bottom-right) within each square grid
for i in range(len(grid_x) - 1):
    for j in range(len(grid_y) - 1):
        x_start, x_end = grid_x[i], grid_x[i + 1]
        y_start, y_end = grid_y[j + 1], grid_y[j]  # Reversed y-coordinates for top-left to bottom-right
        plt.plot([x_start, x_end], [y_start, y_end], color="black", linestyle="--", linewidth=0.5)

# Plot electrodes and vectors
plt.plot(electrode_x, electrode_y, "o", label="Electrodes", color="red")
quiver = plt.quiver(x_origins, y_origins, x_vectors, y_vectors, color="blue")

# Annotate magnitudes near each vector
for x, y, mag in zip(x_origins, y_origins, magnitudes):
    plt.text(x, y, f"{mag:.4f}", fontsize=8, color="black", ha='right', va='bottom')

# Add grid and axes
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()
plt.title("Vector plot (velocity vectors in m/s)")
plt.xlabel("X position (mm)")
plt.ylabel("Y position (mm)")
plt.show()

#%%

# data = "test_gaussian_data_45.xlsx"

# A = AnalyseDataExcel(data)
# peak_num = 0
# num_vectors = 0.9
# v_min = 0.4
# v_max = 2

# # List of e1 values to plot
# e1_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Replace with the actual range or list of e1 values

# # Set up the figure and color map
# plt.figure(figsize=(12, 6))  # Adjust the figure size
# cmap = plt.cm.viridis

# # Electrode positions
# electrode_x = [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]
# electrode_y = [0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12]

# # Initialize variables to track global min/max for axis limits
# global_x_min, global_x_max = float('inf'), float('-inf')
# global_y_min, global_y_max = float('inf'), float('-inf')

# # Collect all max_RXY values for normalization
# all_max_RXY = []

# for e1 in e1_values:
#     # Call the velocityMap function to get data
#     _, _, max_RXY_arr = A.velocityMap(e1, v_min, v_max, peak_num, num_vectors)
#     all_max_RXY.extend(max_RXY_arr)

# # Define normalization for color mapping
# norm = plt.Normalize(vmin=np.min(all_max_RXY), vmax=np.max(all_max_RXY))

# vectors_arr = []
# vector_magnitudes = []

# # Loop through e1 values to plot vectors
# for e1 in e1_values:
#     # Call the velocityMap function
#     vectors, origin, max_RXY_arr = A.velocityMap(e1, v_min, v_max, peak_num, num_vectors)
    
#     """CHECK VECTOR MAGNITUDE"""
#     vectors_arr.append((e1, vectors))
#     tot_vector = np.sum(vectors, axis = 1)
#     tot_vector_magnitude = np.linalg.norm(tot_vector)
#     vector_magnitudes.append(tot_vector_magnitude)
#     """CHECK VECTOR MAGNITUDE"""
    
#     # Extract x and y components from the vectors
#     x_vectors = [v[0] for v in vectors]  # x components
#     y_vectors = [v[1] for v in vectors]  # y components

#     # Create origins for all vectors
#     x_origins = [origin[0]] * len(vectors)
#     y_origins = [origin[1]] * len(vectors)

#     # Add quiver plot for the current e1
#     plt.quiver(
#         x_origins, y_origins, x_vectors, y_vectors,
#         max_RXY_arr,  # Use magnitudes to color the vectors
#         angles='xy', scale_units='xy', scale=1,
#         cmap=cmap, norm=norm, alpha=0.6,  # Apply the normalization
#     )

#     # Update global min/max for axis limits
#     x_positions = [x_orig + x_vec for x_orig, x_vec in zip(x_origins, x_vectors)]
#     y_positions = [y_orig + y_vec for y_orig, y_vec in zip(y_origins, y_vectors)]
#     global_x_min = min(global_x_min, *x_positions)
#     global_x_max = max(global_x_max, *x_positions)
#     global_y_min = min(global_y_min, *y_positions)
#     global_y_max = max(global_y_max, *y_positions)

# # Plot electrode positions
# plt.plot(electrode_x, electrode_y, "o", label="Electrodes", color="red")

# # Add a colorbar
# cbar = plt.colorbar(pad=0.15)
# cbar.set_label('Cross-Correlation')

# # Set dynamic axis limits based on global min/max of vectors
# plt.xlim(global_x_min - 1, global_x_max + 1)
# plt.ylim(global_y_min - 1, global_y_max + 1)

# plt.axhline(0, color='black', linewidth=0.5)
# plt.axvline(0, color='black', linewidth=0.5)
# plt.grid()

# # Label the plot
# plt.title("Velocity Plot for Left Column of Electrodes")
# plt.xlabel("x-position (a.u.)")
# plt.ylabel("y-position (a.u.)")

# # Add the legend outside the plot
# plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))  # Adjust legend position

# # Adjust layout to prevent overlap between elements
# plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for the legend

# # Show the plot
# plt.show()