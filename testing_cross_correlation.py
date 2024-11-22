#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:13:32 2024

@author: candace_chung
"""

import time
import pandas as pd
import numpy as np
from scipy import integrate
from scipy.signal.windows import kaiser
from scipy.signal import correlate
import matplotlib.pyplot as plt
from test_cross_correlation_modules import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
#%%

#TESTING WITH SIMPLER DATA (REGULAR HEARTBEAT)

data = "ElectrogramData.xlsx"
#coord = L.coordinates()
#print(S)
#print(coord)

A = AnalyseDataExcel(data)
corr_mode = "same"

e1 = 8
e2 = 9
v = 2  # velocity in m/s

# Perform correlation
RXY, index_delays = A.simpleCorrelate(e1, e2, corr_mode)

# Calculate minimum time delay based on velocity
minTimeDelay = 4 * 0.001 / v
neg_time_x = -minTimeDelay
pos_time_x = minTimeDelay
time_y = np.linspace(np.min(RXY), np.max(RXY), 50)

# Find the best time delay and its corresponding RXY value
best_timeDelay, max_RXY = A.maxRXY_timeDelay(RXY, index_delays, minTimeDelay)
#print(best_timeDelay, max_RXY)

# Plotting
plt.plot(index_delays / 2034.5, RXY, label="Cross-Correlation RXY")
plt.axvline(neg_time_x, color='orange', linestyle="--", label="Negative Min Time Delay")
plt.axvline(pos_time_x, color='green', linestyle="--", label="Positive Min Time Delay")
plt.plot(best_timeDelay, max_RXY, 'ro', label="Best Time Delay")

# Shade the excluded regions
plt.axvspan(np.min(index_delays / 2034.5), neg_time_x, color='orange', alpha=0.3)
plt.axvspan(pos_time_x, np.max(index_delays / 2034.5), color='green', alpha=0.3)

# Add title and labels
plt.title(f"Electrode {e1} and {e2}")
plt.xlabel("Time Delay (s)")
plt.ylabel("RXY")

# Add legend
plt.legend()

# Show the plot
plt.show()

#%%
L = LoadDataExcel(data)
time = L.time_data()
S = L.ele_signals()

plt.plot(time, S[4], label = "electrode 4")
plt.plot(time, S[11], label = "electrode 11")
plt.legend()
plt.show()

#%%

plt.plot(time, S[4], label = "electrode 4")
plt.plot(time, S[12], label = "electrode 12")
plt.legend()
plt.show()


#%%
data = "ElectrogramData.xlsx"
A = AnalyseDataExcel(data)
corr_mode = "same"

# List of e1 values to plot
e1_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14, 15]  # Replace with the actual range or list of e1 values

# Set up the figure and color map
plt.figure(figsize=(12, 6))  # Adjust the figure size
cmap = plt.cm.viridis

# Initialize variables to track global min/max for the axis limits
global_x_min, global_x_max = float('inf'), float('-inf')
global_y_min, global_y_max = float('inf'), float('-inf')

# Normalize the color scale across all e1 values
all_max_RXY = []
for e1 in e1_values:
    _, _, max_RXY_arr = A.velocityMap(e1, corr_mode, 2)
    all_max_RXY.extend(max_RXY_arr)
norm = plt.Normalize(vmin=np.min(all_max_RXY), vmax=np.max(all_max_RXY))

# Electrode positions
electrode_x = [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]
electrode_y = [0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12]

# Loop through e1 values and overlay their velocity vectors
for e1 in e1_values:
    # Call the velocityMap function for the current e1
    vectors, origin, max_RXY_arr = A.velocityMap(e1, corr_mode, 2)

    # Extract x and y components from the vectors
    x_vectors = [v[0] for v in vectors]  # x components
    y_vectors = [v[1] for v in vectors]  # y components

    # Create origins for all vectors
    x_origins = [origin[0]] * len(vectors)
    y_origins = [origin[1]] * len(vectors)

    # Add quiver plot for the current e1
    plt.quiver(
        x_origins, y_origins, x_vectors, y_vectors,
        max_RXY_arr,  # Use magnitudes to color the vectors
        angles='xy', scale_units='xy', scale=1,
        cmap=cmap, norm=norm, alpha=0.6,  # Set alpha for overlay visibility
    )

    # Update global min/max for axis limits
    x_positions = [x_orig + x_vec for x_orig, x_vec in zip(x_origins, x_vectors)]
    y_positions = [y_orig + y_vec for y_orig, y_vec in zip(y_origins, y_vectors)]
    global_x_min = min(global_x_min, *x_positions)
    global_x_max = max(global_x_max, *x_positions)
    global_y_min = min(global_y_min, *y_positions)
    global_y_max = max(global_y_max, *y_positions)

# Plot electrode positions
plt.plot(electrode_x, electrode_y, "o", label="Electrodes", color="red")

# Add a colorbar
cbar = plt.colorbar(pad=0.15)
cbar.set_label('Cross-Correlation')

# Set dynamic axis limits based on global min/max of vectors
plt.xlim(global_x_min - 1, global_x_max + 1)
plt.ylim(global_y_min - 1, global_y_max + 1)

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()

# Label the plot
#plt.title("Combined Velocity Plot for All Electrodes (Velocity vectors in m/s)")
plt.title("Velocity Plot for Left Column of Electrodes (Velocity vectors in m/s)")
plt.xlabel("x-position (mm)")
plt.ylabel("y-position (mm)")

# Add the legend outside the plot
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))  # Adjust legend position

# Adjust layout to prevent overlap between elements
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for the legend

# Show the plot
plt.show()

#%%
e1 = 5
vectors, origin, max_RXY_arr = A.velocityMap(e1, corr_mode, 2)

# Extract x and y components from the vectors
x_vectors = [v[0] for v in vectors]  # x components
y_vectors = [v[1] for v in vectors]  # y components

# Create origins for all vectors (all start at the same point)
x_origins = [origin[0]] * len(vectors)
y_origins = [origin[1]] * len(vectors)

# Electrode positions
electrode_x = [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]
electrode_y = [0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12]

# Plot the vectors
norm = plt.Normalize(vmin=np.min(max_RXY_arr), vmax=np.max(max_RXY_arr))
cmap = plt.cm.viridis

plt.figure(figsize=(12, 6))  # Adjust the figure size
quiver_plot = plt.quiver(
    x_origins, y_origins, x_vectors, y_vectors,
    max_RXY_arr,  # Use magnitudes to color the vectors
    angles='xy', scale_units='xy', scale=1,
    cmap=cmap, norm=norm, label='Velocity Vector (m/s)'
)
plt.plot(electrode_x, electrode_y, "o", label = "electrodes")

# Add a colorbar to the right
cbar = plt.colorbar(quiver_plot, pad=0.15)  # Adjust pad to avoid overlap
cbar.set_label('Cross-Correlation')

# Set axis limits
x_min = min(np.floor(np.min(x_vectors)) + x_origins[0], 0) - 0.5
x_max = max(np.ceil(np.max(x_vectors)) + x_origins[0], 12) + 0.5
y_min = min(np.floor(np.min(y_vectors)) + y_origins[0], 0) - 0.5
y_max = max(np.ceil(np.max(y_vectors)) + y_origins[0], 12) + 0.5
x_lim = (x_min, x_max)
y_lim = (y_min, y_max)
# x_lim = (-13, 13)
# y_lim = (-13, 25)
plt.xlim(x_lim)
plt.ylim(y_lim)

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()

# Label the plot
plt.title("Electrode %i Velocity Plot"%(e1))
plt.xlabel("x-position (mm)")
plt.ylabel("y-position (mm)")

# Add the legend outside the plot
plt.legend(loc='upper right', bbox_to_anchor=(-0.2, 1))  # Moves legend further right

# Adjust layout to prevent overlap between elements
plt.tight_layout(rect=[0.2, 0, 1, 1])  # Leaves space for the colorbar and legend

# Show the plot
plt.show()

#%%

# #TESTING CROSS-CORRELATION METHOD

# data = "ElectrogramData.xlsx"

# #A = AnalyseDataExcel(data, [0.1, 50], [0, 2 * np.pi], -1)
# A = AnalyseDataExcel(data, [0.1, 50], [0, 2 * np.pi], -1)
# #A = AnalyseDataExcel(data, [0.1, 65], [0, 2 * np.pi], -1)
# #A = AnalyseDataExcel(data, [0.1, 100], [0, 2 * np.pi], -1)
# #A = AnalyseDataExcel(data, [0.1, 120], [0, 2 * np.pi], -1) ############
# #A = AnalyseDataExcel(data, [0.1, 150], [0, 2 * np.pi], -1)
# #A = AnalyseDataExcel(data, [0.1, 175], [0, 2 * np.pi], -1)

# #X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 1, 500, 500)
# X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 4, 500, 500)
# #X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 5, 500, 500)
# #X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 6, 500, 500)
# #X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 10, 500, 500) ###########
# #X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 14, 500, 500)
# #X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 15, 500, 500)

#%%

#plots for RXY vs velocity and angle
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# contour = ax1.contourf(X, Y, RXY_matrix, levels=50, cmap='viridis')
# ax1.set_xlabel('Angle (radians)')
# ax1.set_ylabel('Velocity (mm/s)')
# ax1.set_title('RXY vs Velocity and Angle')
# plt.colorbar(contour, ax=ax1, label='RXY')
# ax1.legend()

# # **3D Surface Plot (right)**
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.set_zlim(np.min(RXY_matrix), np.max(RXY_matrix))
# p = ax2.plot_surface(X, Y, RXY_matrix, cmap=cm.Blues)
# ax2.set_xlabel('Angle (rad.)')
# ax2.set_ylabel('Velocity (mm/s)')
# ax2.set_zlabel('RXY')
# sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=colors.Normalize(vmin=np.min(RXY_matrix), vmax=np.max(RXY_matrix)))
# sm.set_array([])  # Set an empty array for the ScalarMappable
# fig.colorbar(sm, ax=ax2, location='right', pad = 0.1)

# # Show the figure with both subplots
# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

#%%

#plots for RXY vs x and y velocity component
# Create the figure and subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': None})

# # **Contour Plot (left)**
# contour = ax1.contourf(VX, VY, RXY_matrix, levels=50, cmap='viridis')
# ax1.set_xlabel('Vx (mm/s)')
# ax1.set_ylabel('Vy (mm/s)')
# ax1.set_title('RXY vs Velocity')
# plt.colorbar(contour, ax=ax1, label='RXY')

# # Uncomment this line if `optimal_angle` and `optimal_velocity` are defined:
# # ax1.scatter(np.degrees(optimal_angle), optimal_velocity, color='red', label='Max Correlation', zorder=10)
# ax1.legend()

# # **3D Surface Plot (right)**
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.set_zlim(np.min(RXY_matrix), np.max(RXY_matrix))
# p = ax2.plot_surface(VX, VY, RXY_matrix, cmap=cm.Blues)
# ax2.set_xlabel('Vx (mm/s)')
# ax2.set_ylabel('Vy (mm/s)')
# ax2.set_zlabel('RXY')
# ax2.set_title('RXY Surface Plot')

# # Add a colorbar for the 3D surface
# sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=colors.Normalize(vmin=np.min(RXY_matrix), vmax=np.max(RXY_matrix)))
# sm.set_array([])  # Set an empty array for the ScalarMappable
# fig.colorbar(sm, ax=ax2, location='right', pad=0.1)

# # Show the figure with both subplots
# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()

# #%%

# #PLOT VECTOR

# origin = [0, 0]  # Starting point of the vector (x, y)
# vector = [3, 4]  # Vector components (vx, vy)

# # Plot the vector
# plt.quiver(*origin, *vector, angles='xy', scale_units='xy', scale=1, color='r')
# plt.xlim(-1, 5)
# plt.ylim(-1, 5)
# plt.grid()
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)

# # Label the plot
# plt.title("2D Vector")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")

# # Show the plot
# plt.show()