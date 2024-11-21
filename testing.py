#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:02:20 2024

@author: candace_chung
"""

import time
import pandas as pd
import numpy as np
from scipy import integrate
from scipy.signal.windows import kaiser
from scipy.signal import correlate
import matplotlib.pyplot as plt
import modules_new
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib import cm
from matplotlib import colors


#%%
data = "testAF1data.mat"
#%%
L = loadData(data)

X = L.x_position()
Y = L.y_position()
Z = L.z_position()
signal = L.ele_signals()
posT = L.pos_time_arr()
sigT = L.sig_time_arr()
posF = L.pos_samp_freq()
sigF = L.sig_samp_freq()

#%%

# def egmcorr(eX, eY, sampleFreq, tWindowWidth, tMaxLag):
#     """
#     Cross-correlation function similar to MATLAB's egmcorr.

#     Parameters:
#     eX, eY: Input signals (1D arrays)
#     sampleFreq: Sampling frequency
#     tWindowWidth: Width of the time window in seconds
#     tMaxLag: Maximum lag in seconds

#     Returns:
#     RXY: Cross-correlation matrix
#     tShift: Time shifts corresponding to the lags
#     indShift: Lag indices
#     RXX, RYY: Auto-correlation values for normalization
#     """
#     # Convert to vectors
#     eX = np.asarray(eX).flatten()
#     eY = np.asarray(eY).flatten()

#     # Create window
#     nW = int(tWindowWidth * sampleFreq)  # Number of samples in the window
#     nHalfW = nW // 2
#     nW = 2 * nHalfW + 1
#     w = kaiser(nW, beta=2)  # Kaiser window (beta=2)
#     w = w / np.sum(w)  # Normalize window

#     # Max lag and lag indices
#     maxDelta = int(np.ceil(tMaxLag * sampleFreq))
#     indShift = np.arange(-maxDelta, maxDelta + 1)
#     indShiftZero = maxDelta

#     k = nHalfW + indShift[-1]

#     # Buffer for eY
#     nBuff = len(indShift)
#     eYwB = np.zeros((len(w), nBuff))  # Buffer for windowed eY
#     tFirst = k + 1

#     # Pre-fill buffer
#     for i in range(tFirst - maxDelta, tFirst + maxDelta + 1):
#         index = 1 + (i % nBuff)
#         eYwB[:, index - 1] = eY[(i - nHalfW):(i + nHalfW + 1)] * w

#     # Cross-correlation and auto-correlation calculations
#     RXY = np.zeros((len(eX), len(indShift)))
#     RXX = np.zeros(len(eX))
#     RYY = np.zeros(len(eX))

#     for t in range(tFirst, len(eX) - k):
#         # Add next eY segment to buffer
#         tNewBuff = t + maxDelta
#         index1 = 1 + (tNewBuff % nBuff)
#         eYwB[:, index1 - 1] = eY[(tNewBuff - nHalfW):(tNewBuff + nHalfW + 1)] * w

#         eXw = eX[(t - nHalfW):(t + nHalfW + 1)] * w
#         index2 = 1 + (t + indShift) % nBuff
#         eYw_shifted = eYwB[:, index2 - 1]

#         RXY[t, :] = np.dot(eXw, eYw_shifted)
#         RXX[t] = np.dot(eXw, eXw)

#         index3 = 1 + (t % nBuff)
#         eYw = eYwB[:, index3 - 1]
#         RYY[t] = np.dot(eYw, eYw)

#     # Convert shift indices to time values
#     tShift = indShift * 2 / sampleFreq

#     return RXY, tShift, indShift, RXX, RYY

#%%

#test code for Python version of egm_corr
# Parameters
sampleFreq = 2034.5
tS = 1 / sampleFreq

# Parameters for egmcorr
tWindowWidth = 20 / 1000
tMaxLag = 20 / 1000

# Generate test spike
sigma = 5 / 1000
t = np.arange(-4 * sigma, 4 * sigma + tS, tS)
testSpike = (1 - (t / sigma) ** 2) * np.exp(-t ** 2 / (2 * sigma ** 2))

# Spike timing and shifts
nSpike = 5
ind_Act = np.arange(1, nSpike + 1) * 200
diff_Act = ((np.arange(1, nSpike + 1) - np.ceil(nSpike / 2)) * tMaxLag / nSpike * sampleFreq).astype(int)

# Initialize signals
signal_length = ind_Act[-1] + diff_Act[-1] + 3 * round(sampleFreq * (tWindowWidth + tMaxLag))
e1 = np.zeros(signal_length)
e2 = np.zeros(signal_length)

# Add spikes
indSpike = np.arange(1, len(t) + 1) - np.ceil(len(t) / 2).astype(int)
for i in range(nSpike):
    ind = indSpike + ind_Act[i]
    e1[ind.astype(int)] = testSpike
    e2[(ind + diff_Act[i]).astype(int)] = testSpike

# Plot signals
plt.figure()
plt.plot(e1, label='e1')
plt.plot(e2, label='e2')
plt.legend()
plt.show()

R, tShift, _, _, _ = egmcorr(e1, e2, sampleFreq, tWindowWidth, tMaxLag)

# Find peaks in the maximum correlation
Rmax = np.max(R, axis=1)
iDelta = np.argmax(R, axis=1)
tDiff = tShift[iDelta]

# `find_peaks` returns a tuple; we need only the first element (indices of peaks)
pks, properties = find_peaks(Rmax, height= 0.5)  # MinPeakHeight equivalent
iDelta_locs = iDelta[pks]  # Use `pks` as indices
tShift_locs = tShift[iDelta_locs] * sampleFreq

# Print results
print("Index of peaks:", iDelta_locs)
print("Time shift of peaks (in samples):", tShift_locs)
print("Expected activity differences:", diff_Act)

#%%

A = AnalyseData(data, [0, 1], [0, 2 * np.pi])

d = A.electrodeDistance(0, 0, 1)

t = A.timeDelay(0, 0, 1, 0.5, np.pi/6)

print(d, t)

#%%
time = sigT  # Time axis
e1 = signal[0]          # First signal
e2 = signal[14]         # Second signal (to be shifted)

#shifted_array2, padded_array1 = A.shiftSignal(e1, e2, 1/dt, time_delay)
shifted_e2, paired_ind = A.shiftSignal2(e2, t)
    
plt.plot(time, e2, label = "original")
plt.plot(time, shifted_e2, label = "rolled")
plt.legend()
print("Padded Array 1:", e1.shape)
print("Shifted Array 2:", shifted_array2.shape)

#%%
eX = signal[0]
e2 = signal[1]
eY = A.shiftSignal2(eX, e2, t)
sampleFreq = sigF
tWindowWidth = 20/1000
tMaxLag = 20/1000

RXY, tShift, indShift, RXX, RYY = A.egmcorr(eX, eY, sampleFreq, tWindowWidth, tMaxLag)

#%%
print(RXY, tShift)

#%%
print(RXX, RYY)