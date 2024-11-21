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

L = LoadDataExcel(data)
time = L.time_data()
S = L.ele_signals()
coord = L.coordinates()
print(S)
print(coord)

A = AnalyseDataExcel(data, [10, 100], [0, 2 * np.pi], -1)

e = 14
v = 133
theta = np.pi/2
d = A.electrodeDistance(0, e)
t = A.timeDelay(0, e, v, theta)
print(d, t)

shifted_e2_1 = A.shiftSignal(len(S[0]), e, t)
shifted_e2_2 = A.shiftSignal2(len(S[0]), e, t)
shifted_e2_3 = A.simpleShiftSignal(e, t)

e1 = S[0]
e2 = S[e]
plt.plot(time, e2, label = "original electrode 14")
plt.plot(time, shifted_e2_1, label = "integer shift")
#plt.plot(time, shifted_e2_2, label = "fraction - shiftSignal2")
plt.plot(time, shifted_e2_3, label = "fractional shift")
plt.show()
plt.legend()

RXY = A.simpleCorrelate(0, e, v, theta)

print(np.sum(RXY))

#%%

#TESTING CROSS-CORRELATION METHOD

data = "ElectrogramData.xlsx"

#A = AnalyseDataExcel(data, [0.1, 50], [0, 2 * np.pi], -1)
A = AnalyseDataExcel(data, [0.1, 50], [0, 2 * np.pi], -1)
#A = AnalyseDataExcel(data, [0.1, 65], [0, 2 * np.pi], -1)
#A = AnalyseDataExcel(data, [0.1, 100], [0, 2 * np.pi], -1)
#A = AnalyseDataExcel(data, [0.1, 120], [0, 2 * np.pi], -1) ############
#A = AnalyseDataExcel(data, [0.1, 150], [0, 2 * np.pi], -1)
#A = AnalyseDataExcel(data, [0.1, 175], [0, 2 * np.pi], -1)

#X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 1, 500, 500)
X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 4, 500, 500)
#X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 5, 500, 500)
#X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 6, 500, 500)
#X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 10, 500, 500) ###########
#X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 14, 500, 500)
#X, Y, VX, VY, RXY_matrix = A.crossCorrelationMatrix(0, 15, 500, 500)

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': None})

# **Contour Plot (left)**
contour = ax1.contourf(VX, VY, RXY_matrix, levels=50, cmap='viridis')
ax1.set_xlabel('Vx (mm/s)')
ax1.set_ylabel('Vy (mm/s)')
ax1.set_title('RXY vs Velocity')
plt.colorbar(contour, ax=ax1, label='RXY')

# Uncomment this line if `optimal_angle` and `optimal_velocity` are defined:
# ax1.scatter(np.degrees(optimal_angle), optimal_velocity, color='red', label='Max Correlation', zorder=10)
ax1.legend()

# **3D Surface Plot (right)**
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_zlim(np.min(RXY_matrix), np.max(RXY_matrix))
p = ax2.plot_surface(VX, VY, RXY_matrix, cmap=cm.Blues)
ax2.set_xlabel('Vx (mm/s)')
ax2.set_ylabel('Vy (mm/s)')
ax2.set_zlabel('RXY')
ax2.set_title('RXY Surface Plot')

# Add a colorbar for the 3D surface
sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=colors.Normalize(vmin=np.min(RXY_matrix), vmax=np.max(RXY_matrix)))
sm.set_array([])  # Set an empty array for the ScalarMappable
fig.colorbar(sm, ax=ax2, location='right', pad=0.1)

# Show the figure with both subplots
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()