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
import scipy.signal as sps
import matplotlib.pyplot as plt
from test_cross_correlation_modules import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
import pywt
from scipy.stats import zscore

#%%

def dwt_decompose(signal, wavelet='db4', level=4):
    """
    Decomposes a signal using discrete wavelet transform (DWT).

    Args:
        signal (array): Input time series signal.
        wavelet (str): Wavelet type (default: 'db4').
        level (int): Decomposition level (default: None, maximum level).

    Returns:
        coeffs (list): Approximation and detail coefficients.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs

def cross_correlation(signal1, signal2):
    """
    Computes cross-correlation between two signals using scipy.signal.correlate.

    Args:
        signal1 (array): First input signal.
        signal2 (array): Second input signal.

    Returns:
        max_corr (float): Maximum cross-correlation coefficient.
    """
    RXY = correlate(signal1, signal2, mode='full', method='direct')
    index_delays = sps.correlation_lags(len(e1), len(e2), mode = "full")
    max_ind = RXY.argmax()
    max_corr, max_delay = RXY[max_ind], index_delays[max_ind]
    return max_corr, max_delay

def dwt_cross_correlation(signal1, signal2, wavelet='db4', level=4):
    """
    Computes DWT-based cross-correlation between two signals.

    Args:
        signal1 (array): First input signal.
        signal2 (array): Second input signal.
        wavelet (str): Wavelet type (default: 'db4').
        level (int): Decomposition level (default: None, maximum level).

    Returns:
        corr_coeffs (list): Cross-correlation coefficients for each DWT level.
    """
    # Normalize signals to zero mean and unit variance
    signal1 = zscore(signal1)
    signal2 = zscore(signal2)

    # Decompose signals using DWT
    coeffs1 = dwt_decompose(signal1, wavelet, level)
    coeffs2 = dwt_decompose(signal2, wavelet, level)

    # Compute cross-correlation for each decomposition level
    corr_coeffs = []
    ind_lags = []
    for c1, c2 in zip(coeffs1, coeffs2):
        # Ensure the signals are the same length for correlation
        min_len = min(len(c1), len(c2))
        max_corr, max_delay = cross_correlation(c1[:min_len], c2[:min_len])
        corr_coeffs.append(max_corr)
        ind_lags.append(max_delay)
    return corr_coeffs, ind_lags

# Create a Gaussian pulse
def generate_gaussian_pulse(length, std_dev, shift=0):
    """
    Generate a Gaussian pulse with optional time shift.

    Args:
        length (int): Length of the signal.
        std_dev (float): Standard deviation of the Gaussian.
        shift (int): Number of samples to shift the Gaussian pulse.

    Returns:
        signal (ndarray): Generated Gaussian pulse.
    """
    pulse = sps.gaussian(length, std_dev)
    if shift > 0:
        pulse = np.roll(pulse, shift)
    return pulse

#%%
# Parameters
length = 1024   # Length of the signals
std_dev = 50    # Standard deviation of the Gaussian
shift = 20      # Time shift for the second signal

# Generate two signals: one shifted
signal1 = generate_gaussian_pulse(length, std_dev, shift=0)
signal2 = generate_gaussian_pulse(length, std_dev, shift=shift)

# Visualize the signals
plt.figure(figsize=(12, 6))
plt.plot(signal1, label='Original Gaussian Pulse', alpha=0.8)
plt.plot(signal2, label=f'Shifted Gaussian Pulse (Shift={shift})', alpha=0.8)
plt.legend()
plt.title('Gaussian Pulses')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

delay = np.argmax(correlate(signal1, signal2, mode='full')) - len(signal1) + 1
print("Detected Shift (Original Signal):", delay)
# Import functions and perform DWT-based cross-correlation
wavelet = 'db4'
corr_coeffs, ind_lags = dwt_cross_correlation(signal1, signal2, wavelet=wavelet)

# Print the results
print("Cross-Correlation Coefficients at Each Level:")
for level, (corr, lag) in enumerate(zip(corr_coeffs, ind_lags), start=1):
    downsampling_factor = 2**level
    adjusted_delay = lag / downsampling_factor
    print(f"Level {level}: Max Corr = {corr:.4f}, Delay = {adjusted_delay:.2f} samples")

#%%# Visualize cross-correlation coefficients across levels
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(corr_coeffs) + 1), corr_coeffs, color='skyblue', edgecolor='k', alpha=0.7)
plt.xlabel('DWT Level')
plt.ylabel('Maximum Cross-Correlation')
plt.title('DWT-Based Cross-Correlation for Gaussian Pulses')
plt.grid()
plt.show()

#%%

#TESTING WITH SIMPLER DATA (REGULAR HEARTBEAT)

data = "ElectrogramData.xlsx"
L = LoadDataExcel(data)
S = L.ele_signals()
coord = L.coordinates()
#print(S)
#print(coord)

A = AnalyseDataExcel(data)

e1 = S[2]
e2 = S[7]

corr_coeffs, ind_lags = dwt_cross_correlation(e1, e2)

plt.plot(ind_lags, corr_coeffs, "o")
plt.show()

#%%
data = "ElectrogramData.xlsx"
L = LoadDataExcel(data)
S = L.ele_signals()
coord = L.coordinates()

ele_1 = 8
ele_2 = 9
e1 = S[ele_1]
e2 = S[ele_2]
e14 = S[14]
ind_shift = 0
v = 2  # velocity in m/s
indices = np.arange(0, len(S[ele_1]), 1)
ind_shifts = A.findEGMPeak(e1)
e1_w, e2_w = A.windowSignal(e1, e2, ind_shifts[3])
print(ind_shifts)
plt.plot(indices, e1)
plt.plot(indices, e1_w)
plt.show()

#%%
plt.plot(indices, e2)
#plt.plot(peaks, e14[peaks], "x", label='Peaks')

plt.plot(indices, e2_w)
#plt.plot(indices, e2_w)
plt.show()

#%%

data = "ElectrogramData.xlsx"
L = LoadDataExcel(data)
S = L.ele_signals()
coord = L.coordinates()
#print(S)
#print(coord)

A = AnalyseDataExcel(data)
v = 2  # velocity in m/s
u = 0.4
peak_num = 2

ele_1 = 2
ele_2 = 14
e1 = S[ele_1]
e2 = S[ele_2]

ind_shifts = A.findEGMPeak(e1)

window_offset = ind_shifts[peak_num]

e1_w, e2_w = A.windowSignal(e1, e2, window_offset)

# Perform correlation
RXY, index_delays = A.simpleCorrelate(e1_w, e2_w)

# Calculate minimum time delay based on velocity
minTimeDelay = np.linalg.norm(coord[ele_2] - coord[ele_1]) * 0.001 / v_max
maxTimeDelay = np.linalg.norm(coord[ele_2] - coord[ele_1]) * 0.001 / v_min
neg_time_x = -minTimeDelay
neg_time_x1 = -maxTimeDelay
pos_time_x = minTimeDelay
pos_time_x1 = maxTimeDelay
time_y = np.linspace(np.min(RXY), np.max(RXY), 50)

# Find the best time delay and its corresponding RXY value

best_timeDelay, max_RXY = A.maxRXY_timeDelay(RXY, index_delays, minTimeDelay, maxTimeDelay)
#print(best_timeDelay, max_RXY)

# Plotting
plt.plot(index_delays / 2034.5, RXY, label="Cross-Correlation RXY")
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

velocity_vector, max_RXY = A.velocity(ele_1, ele_2, v_min, v_max, ind_shifts[peak_num])

print(ele_1, ele_2, velocity_vector, max_RXY, best_timeDelay)
#%%
L = LoadDataExcel(data)
time = L.time_data()
S = L.ele_signals()

#plt.plot(time, S[3], label = "electrode 3")
#plt.plot(time, S[2], label = "electrode 2")
#plt.plot(time, S[7], label = "electrode 7")
plt.plot(time, S[12], label = "electrode 12")
plt.plot(time, S[13], label = "electrode 13")
plt.legend()
plt.show()

#%%

L = LoadDataExcel(data)
time = L.time_data()
S = L.ele_signals()

plt.plot(time, S[2], label = "electrode 2")
plt.plot(time, S[6], label = "electrode 6")
plt.plot(time, S[10], label = "electrode 10")
plt.legend()
plt.show()


#%%
# Data and initialization
data = "ElectrogramData.xlsx"
A = AnalyseDataExcel(data)
peak_num = 3
num_vectors = 3
v_min = 0.4
v_max = 2

# List of e1 values to plot
e1_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Replace with the actual range or list of e1 values

# Set up the figure and color map
plt.figure(figsize=(12, 6))  # Adjust the figure size
cmap = plt.cm.viridis

# Electrode positions
electrode_x = [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]
electrode_y = [0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12]

# Initialize variables to track global min/max for axis limits
global_x_min, global_x_max = float('inf'), float('-inf')
global_y_min, global_y_max = float('inf'), float('-inf')

# Collect all max_RXY values for normalization
all_max_RXY = []

for e1 in e1_values:
    # Call the velocityMap function to get data
    _, _, max_RXY_arr = A.velocityMap(e1, v_min, v_max, peak_num, num_vectors)
    all_max_RXY.extend(max_RXY_arr)

# Define normalization for color mapping
norm = plt.Normalize(vmin=np.min(all_max_RXY), vmax=np.max(all_max_RXY))

# Loop through e1 values to plot vectors
for e1 in e1_values:
    # Call the velocityMap function
    vectors, origin, max_RXY_arr = A.velocityMap(e1, v_min, v_max, peak_num, num_vectors)

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
        cmap=cmap, norm=norm, alpha=0.6,  # Apply the normalization
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
e1 = 2
peak_num = 2
num_vector = 0.9
v_min = 0.4
v_max = 2
vectors, origin, max_RXY_arr = A.velocityMap(e1, v_min, v_max, peak_num, num_vector)
print(vectors, max_RXY_arr)

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