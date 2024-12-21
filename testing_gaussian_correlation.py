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
from test_cross_correlation_modules_MOVING import *
from dummy_gaussian_data import *
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

data = "test_gaussian_data_hori_1200_50Hz.xlsx"
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

data = "test_gaussian_data_10_1200_50Hz.xlsx"
peak_num = 0
v_min = 0.35
v_max = 8.5
corr_threshold = 0.75
window_length = 20
A = AnalyseDataExcel(data, v_min, v_max, window_length, corr_threshold)
C = A.coordinates()

ele_1 = 2
ele_2 = 7
e1 = S[ele_1]
e2 = S[ele_2]
peaks1, _ = sps.find_peaks(e1, height = 0.9)
peaks2, _ = sps.find_peaks(e2, height = 0.9)

peak_time_e1 = times[peaks1[0]]
peak_time_e2 = times[peaks2[0]]
time_diff = peak_time_e2 - peak_time_e1
print(peak_time_e1, peak_time_e2, time_diff)
#%%
plt.plot(times, e1, label = "electrode %i"%(ele_1))
plt.plot(times, e2, label = "electrode %i"%(ele_2))
plt.legend()
plt.show()

#%%

ind_shifts = A.findEGMPeak(e1)

window_offset = ind_shifts[peak_num]

e1_w, e2_w = A.windowSignal(e1, e2, window_offset)

plt.plot(times, e1, label = "electrode %i"%(ele_1))
plt.plot(times, e1_w, label = "electrode %i windowed"%(ele_1))
plt.plot(times, e2, label = "electrode %i"%(ele_2))
plt.plot(times, e2_w, label = "electrode %i windowed"%(ele_2))
#plt.plot(times, padded_kaiser, label = "kaiser window")
#plt.vlines((250 - 407/2) * 20/1000, 0, 1, color = 'black', label = "min time of window")
#plt.vlines((250 + 407/2) * 20/1000, 0, 1, color = 'black', label = "max time of window")
plt.legend()
plt.show()

#%%
# Perform correlation
RXY, index_delays = A.simpleCorrelate(e1_w, e2_w)
peaksRXY, _ = sps.find_peaks(RXY, height = 0.7)
print(peaksRXY)
peak_index_e1 = index_delays[peaksRXY[0]]
peakRXY_val = RXY[peaksRXY[0]]
print(peak_index_e1 / (2034.5), peakRXY_val)
plt.plot(index_delays / (2034.5), RXY, label = "Electrode 0 and 5")
plt.plot(peak_index_e1 / (2034.5), peakRXY_val, "o")


ind_shifts1 = A.findEGMPeak(S[2])

window_offset1 = ind_shifts1[peak_num]

e1_w1, e2_w1 = A.windowSignal(S[2], S[7], window_offset1)

RXY1, index_delays1 = A.simpleCorrelate(e1_w1, e2_w1)
peaksRXY1, _ = sps.find_peaks(RXY1, height = 0.7)
print(peaksRXY1)
peak_index_e11 = index_delays[peaksRXY1[0]]
peakRXY_val1 = RXY1[peaksRXY1[0]]
print(peak_index_e11 / (2034.5), peakRXY_val1)
plt.plot(index_delays1 / (2034.5), RXY1, label = "Electrode 2 and 7")
plt.plot(peak_index_e11 / (2034.5), peakRXY_val1, "o")

plt.xlabel("Time (s)")
plt.ylabel("RXY")
plt.legend()
plt.show()
#print("velocity guess", 4/(peak_index_e11 / (2034.5)) * 0.001)

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

best_timeDelay, max_RXY = A.maxRXY_timeDelay(RXY, index_delays, minTimeDelay, maxTimeDelay)
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

velocity_vector, _ = A.electrodePairVelocity_ORIGINAL(2, 3, ind_shifts[peak_num])
velocity_vector2, _ = A.electrodePairVelocity_ORIGINAL(2, 6, ind_shifts[peak_num])
print("CHECK", velocity_vector, velocity_vector2)
#%%
v_guess = A.guessVelocity_LSQ(0, 1, 5, peak_num)

#%%

data = "test_gaussian_data_20_400_50Hz.xlsx"

peak_num = 0
v_min = 0.35
v_max = 8.5
corr_threshold = 0.75
window_length = 814

A = AnalyseDataExcel(data, v_min, v_max, window_length, corr_threshold)
# Your existing code to compute vectors and origins
vectors, origins = A.velocityGuessMap(peak_num)

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

#1.5 m/s 50 Hz
v = [1.6276, 1.596, 1.5112, 1.3965, 1.4390, 1.3965, 1.5112, 1.596, 1.6276]
theta = [0, 11.309932474020227, 21.80140948635181, 30.96375653207351, 45.00000000000001, 59.03624346792648, 68.19859051364818, 78.69006752597979, 90]

v_ans = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
theta_ans = [0, 10, 20, 30, 45, 60, 70, 80, 90]

plt.plot(theta, v, "o", label = "data")
plt.plot(theta_ans, v_ans, label = "answer")
plt.xlabel("Angle (degrees)")
plt.ylabel("Velocity (m/s)")
plt.title("Gaussian wave travelling at 1.5 m/s with frequency of 50 Hz")
plt.legend()

#%%

#1.2 m/s 50 Hz
v1 = [1.1626, 1.1537, 1.2867, 1.2140, 1.151, 1.2140, 1.2867, 1.1537, 1.1626]
theta1 = [0, 8.130102354155916, 18.434948822922035, 26.565051177077994, 45.00000000000001, 63.43494882292201, 71.56505117707799, 81.86989764584402, 90]

v_ans = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
theta_ans = [0, 10, 20, 30, 45, 60, 70, 80, 90]

plt.plot(theta1, v1, "o", label = "data")
plt.plot(theta_ans, v_ans, label = "answer")
plt.xlabel("Angle (degrees)")
plt.ylabel("Velocity (m/s)")
plt.title("Gaussian wave travelling at 1.2 m/s with frequency of 50 Hz")
plt.legend()

#%%

#0.4 m/s 50 Hz
v1 = [0.4064, (0.4033+0.3992)/2, 0.3995, (0.4119+0.3945)/2, 0.4107, (0.4119+0.3945)/2, 0.3995, (0.4033+0.3992)/2, 0.4064]
theta1 = [2.8624052261117465, (11.309932474020227 + 8.530765609948139)/2, 20.224859431168078, (29.05460409907714 + 30.465544919459877)/2, 45.00000000000001, (60.94539590092285 + 59.53445508054011)/2,  69.77514056883193, (78.69006752597979 + 81.46923439005187)/2, 87.13759477388825]

v_ans = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
theta_ans = [0, 10, 20, 30, 45, 60, 70, 80, 90]

plt.plot(theta1, v1, "o", label = "data")
plt.plot(theta_ans, v_ans, label = "answer")
plt.xlabel("Angle (degrees)")
plt.ylabel("Velocity (m/s)")
plt.title("Gaussian wave travelling at 0.4 m/s with frequency of 50 Hz")
plt.legend()

#%%

#FUNCTIONS TO TEST GAUSSIAN WAVE WITH MOVING ELECTRODES

def test_single_wave_moving_electrode(angle, propagation_speed, max_variation, target_mag, target_theta, v_min = 0.35, v_max = 8.5, peak_num = 0, corr_threshold = 0.75, window_length = 50):
    data = generate_data_single_wave_moving_electrode(angle, propagation_speed, max_variation)
    A = AnalyseDataExcel_MOVING(data, v_min, v_max, window_length, corr_threshold)
    S = A.ele_signals()
    C = A.coordinates()
    
    # Your existing code to compute vectors and origins
    vectors, origins = A.velocityGuessMap(peak_num)
    
    # Calculate average velocity and angle
    tot_mag = 0
    tot_theta = 0
    var_mag = 0
    var_theta = 0
    N = len(vectors)
    for vx, vy in vectors:
        mag = np.sqrt(vx**2 + vy**2)
        theta = np.degrees(np.arctan2(vy, vx))
        tot_mag += mag
        tot_theta += theta
        var_mag += (mag - target_mag)**2 / N
        var_theta += (theta - target_theta)**2 / N
    
    avg_mag = tot_mag / N
    avg_theta = tot_theta / N
    stddev_mag = np.sqrt(var_mag)
    stddev_theta = np.sqrt(var_theta)
    print("avg", avg_mag, avg_theta)
    print("std dev.", stddev_mag, stddev_theta)
    
    # Vector plot code
    
    x_origins = [x[0] for x in origins]
    y_origins = [x[1] for x in origins]
    x_vectors = [x[0] for x in vectors]
    y_vectors = [x[1] for x in vectors]
    
    # Calculate vector magnitudes
    magnitudes = [np.sqrt(vx**2 + vy**2) for vx, vy in vectors]
    
    # Define electrode positions
    
    # Get the time index corresponding to the peak
    peak_index = A.findEGMPeak(S[0])[peak_num]
    
    # Extract the x and y positions at the time of the peak
    electrode_x = C[:, peak_index, 0]
    electrode_y = C[:, peak_index, 1]
    
    # Grid lines
    plt.plot(
        [electrode_x[0], electrode_x[1]],
        [electrode_y[0], electrode_y[1]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[0], electrode_x[4]],
        [electrode_y[0], electrode_y[4]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[1], electrode_x[2]],
        [electrode_y[1], electrode_y[2]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[1], electrode_x[5]],
        [electrode_y[1], electrode_y[5]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[2], electrode_x[3]],
        [electrode_y[2], electrode_y[3]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[2], electrode_x[6]],
        [electrode_y[2], electrode_y[6]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[4], electrode_x[5]],
        [electrode_y[4], electrode_y[5]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[4], electrode_x[8]],
        [electrode_y[4], electrode_y[8]],
        color="black",
        linewidth=0.5,
    )
    
    
    plt.plot(
        [electrode_x[5], electrode_x[6]],
        [electrode_y[5], electrode_y[6]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[5], electrode_x[9]],
        [electrode_y[5], electrode_y[9]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[6], electrode_x[7]],
        [electrode_y[6], electrode_y[7]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[6], electrode_x[10]],
        [electrode_y[6], electrode_y[10]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[8], electrode_x[9]],
        [electrode_y[8], electrode_y[9]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[8], electrode_x[12]],
        [electrode_y[8], electrode_y[12]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[9], electrode_x[10]],
        [electrode_y[9], electrode_y[10]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[9], electrode_x[13]],
        [electrode_y[9], electrode_y[13]],
        color="black",
        linewidth=0.5,
    )
    
    
    plt.plot(
        [electrode_x[10], electrode_x[11]],
        [electrode_y[10], electrode_y[11]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[10], electrode_x[14]],
        [electrode_y[10], electrode_y[14]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[3], electrode_x[7]],
        [electrode_y[3], electrode_y[7]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[7], electrode_x[11]],
        [electrode_y[7], electrode_y[11]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[11], electrode_x[15]],
        [electrode_y[11], electrode_y[15]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[12], electrode_x[13]],
        [electrode_y[12], electrode_y[13]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[13], electrode_x[14]],
        [electrode_y[13], electrode_y[14]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[14], electrode_x[15]],
        [electrode_y[14], electrode_y[15]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[1], electrode_x[4]],
        [electrode_y[1], electrode_y[4]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[2], electrode_x[5]],
        [electrode_y[2], electrode_y[5]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[5], electrode_x[8]],
        [electrode_y[5], electrode_y[8]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[3], electrode_x[6]],
        [electrode_y[3], electrode_y[6]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[6], electrode_x[9]],
        [electrode_y[6], electrode_y[9]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[9], electrode_x[12]],
        [electrode_y[9], electrode_y[12]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[7], electrode_x[10]],
        [electrode_y[7], electrode_y[10]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[10], electrode_x[13]],
        [electrode_y[10], electrode_y[13]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[11], electrode_x[14]],
        [electrode_y[11], electrode_y[14]],
        color="black",
        linewidth=0.5,
    )
    
    # Plot electrodes and vectors
    plt.plot(electrode_x, electrode_y, "o", label="Electrodes", color="red")
    quiver = plt.quiver(x_origins, y_origins, x_vectors, y_vectors, color="blue")
    
    # Annotate magnitudes near each vector
    for x, y, mag in zip(x_origins, y_origins, magnitudes):
        plt.text(x, y, f"{mag:.4f}", fontsize=8, color="black", ha='right', va='bottom')
    
    # Add grid and axes
    #plt.axhline(0, color='black', linewidth=0.5)
    #plt.axvline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.title("Vector plot (velocity vectors in m/s)")
    plt.xlabel("X position (mm)")
    plt.ylabel("Y position (mm)")
    plt.show()
    return vectors, avg_mag, avg_theta, stddev_mag, stddev_theta

def test_train_wave_moving_electrode(angle, propagation_speed, pulse_frequency, max_variation, target_mag, target_theta, v_min = 0.35, v_max = 8.5, peak_num = 0, corr_threshold = 0.75, window_length = 50):
    data = generate_data_train_wave_moving_electrode(angle, propagation_speed, pulse_frequency, max_variation)
    A = AnalyseDataExcel_MOVING(data, v_min, v_max, window_length, corr_threshold)
    S = A.ele_signals()
    C = A.coordinates()
    # Your existing code to compute vectors and origins
    vectors, origins = A.velocityGuessMap(peak_num)
    
    # Calculate average velocity and angle
    tot_mag = 0
    tot_theta = 0
    var_mag = 0
    var_theta = 0
    N = len(vectors)
    for vx, vy in vectors:
        mag = np.sqrt(vx**2 + vy**2)
        theta = np.degrees(np.arctan2(vy, vx))
        tot_mag += mag
        tot_theta += theta
        var_mag += (mag - target_mag)**2 / N
        var_theta += (theta - target_theta)**2 / N
    
    avg_mag = tot_mag / N
    avg_theta = tot_theta / N
    stddev_mag = np.sqrt(var_mag)
    stddev_theta = np.sqrt(var_theta)
    print("avg", avg_mag, avg_theta)
    print("std dev.", stddev_mag, stddev_theta)
    
    # Vector plot code
    
    x_origins = [x[0] for x in origins]
    y_origins = [x[1] for x in origins]
    x_vectors = [x[0] for x in vectors]
    y_vectors = [x[1] for x in vectors]
    
    # Calculate vector magnitudes
    magnitudes = [np.sqrt(vx**2 + vy**2) for vx, vy in vectors]
    
    # Define electrode positions
    
    # Get the time index corresponding to the peak
    peak_index = A.findEGMPeak(S[0])[peak_num]
    
    # Extract the x and y positions at the time of the peak
    electrode_x = C[:, peak_index, 0]
    electrode_y = C[:, peak_index, 1]
    
    # Grid lines
    plt.plot(
        [electrode_x[0], electrode_x[1]],
        [electrode_y[0], electrode_y[1]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[0], electrode_x[4]],
        [electrode_y[0], electrode_y[4]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[1], electrode_x[2]],
        [electrode_y[1], electrode_y[2]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[1], electrode_x[5]],
        [electrode_y[1], electrode_y[5]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[2], electrode_x[3]],
        [electrode_y[2], electrode_y[3]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[2], electrode_x[6]],
        [electrode_y[2], electrode_y[6]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[4], electrode_x[5]],
        [electrode_y[4], electrode_y[5]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[4], electrode_x[8]],
        [electrode_y[4], electrode_y[8]],
        color="black",
        linewidth=0.5,
    )
    
    
    plt.plot(
        [electrode_x[5], electrode_x[6]],
        [electrode_y[5], electrode_y[6]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[5], electrode_x[9]],
        [electrode_y[5], electrode_y[9]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[6], electrode_x[7]],
        [electrode_y[6], electrode_y[7]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[6], electrode_x[10]],
        [electrode_y[6], electrode_y[10]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[8], electrode_x[9]],
        [electrode_y[8], electrode_y[9]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[8], electrode_x[12]],
        [electrode_y[8], electrode_y[12]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[9], electrode_x[10]],
        [electrode_y[9], electrode_y[10]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[9], electrode_x[13]],
        [electrode_y[9], electrode_y[13]],
        color="black",
        linewidth=0.5,
    )
    
    
    plt.plot(
        [electrode_x[10], electrode_x[11]],
        [electrode_y[10], electrode_y[11]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[10], electrode_x[14]],
        [electrode_y[10], electrode_y[14]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[3], electrode_x[7]],
        [electrode_y[3], electrode_y[7]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[7], electrode_x[11]],
        [electrode_y[7], electrode_y[11]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[11], electrode_x[15]],
        [electrode_y[11], electrode_y[15]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[12], electrode_x[13]],
        [electrode_y[12], electrode_y[13]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[13], electrode_x[14]],
        [electrode_y[13], electrode_y[14]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[14], electrode_x[15]],
        [electrode_y[14], electrode_y[15]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[1], electrode_x[4]],
        [electrode_y[1], electrode_y[4]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[2], electrode_x[5]],
        [electrode_y[2], electrode_y[5]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[5], electrode_x[8]],
        [electrode_y[5], electrode_y[8]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[3], electrode_x[6]],
        [electrode_y[3], electrode_y[6]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[6], electrode_x[9]],
        [electrode_y[6], electrode_y[9]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[9], electrode_x[12]],
        [electrode_y[9], electrode_y[12]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[7], electrode_x[10]],
        [electrode_y[7], electrode_y[10]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[10], electrode_x[13]],
        [electrode_y[10], electrode_y[13]],
        color="black",
        linewidth=0.5,
    )
    
    plt.plot(
        [electrode_x[11], electrode_x[14]],
        [electrode_y[11], electrode_y[14]],
        color="black",
        linewidth=0.5,
    )
    
    # Plot electrodes and vectors
    plt.plot(electrode_x, electrode_y, "o", label="Electrodes", color="red")
    quiver = plt.quiver(x_origins, y_origins, x_vectors, y_vectors, color="blue")
    
    # Annotate magnitudes near each vector
    for x, y, mag in zip(x_origins, y_origins, magnitudes):
        plt.text(x, y, f"{mag:.4f}", fontsize=8, color="black", ha='right', va='bottom')
    
    # Add grid and axes
    #plt.axhline(0, color='black', linewidth=0.5)
    #plt.axvline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.title("Vector plot (velocity vectors in m/s)")
    plt.xlabel("X position (mm)")
    plt.ylabel("Y position (mm)")
    plt.show()
    return vectors, avg_mag, avg_theta, stddev_mag, stddev_theta

#%%

#TEST SINGLE GAUSSIAN WAVE WITH MOVING ELECTRODES
target_mag = 1.2
target_theta = 20
vectors, avg_mag, avg_theta, stddev_mag, stddev_theta = test_single_wave_moving_electrode(20, 1200, 0.5, target_mag, target_theta)


#%%

#TEST TRAIN GAUSSIAN WAVE WITH MOVING ELECTRODES
target_mag = 1.2
target_theta = 20
vectors, avg_mag, avg_theta, stddev_mag, stddev_theta = test_train_wave_moving_electrode(20, 1200, 50, 0.5, target_mag, target_theta)

