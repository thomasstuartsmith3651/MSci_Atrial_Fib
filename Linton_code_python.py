#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:11:29 2024

@author: candace_chung
"""

import numpy as np
from scipy.io import loadmat
import pywt
import pandas as pd
from scipy.signal.windows import kaiser
import scipy.interpolate as spi
import scipy.signal as sps
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import pywt
from scipy.stats import zscore
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.optimize import minimize


def egmcorr(self, eX, eY, sampleFreq, tWindowWidth, tMaxLag):
    """
    Python version of Nick Linton's egm corr MATLAB code.

    Parameters:
    eX, eY: Input signals (1D arrays)
    sampleFreq: Sampling frequency
    tWindowWidth: Width of the time window in seconds
    tMaxLag: Maximum lag in seconds

    Returns:
    RXY: Cross-correlation matrix
    tShift: Time shifts corresponding to the lags
    indShift: Lag indices
    RXX, RYY: Auto-correlation values for normalization
    """
    # Convert to vectors
    eX = np.asarray(eX).flatten()
    eY = np.asarray(eY).flatten()

    # Create window
    nW = int(tWindowWidth * sampleFreq)  # Number of samples in the window
    nHalfW = nW // 2
    nW = 2 * nHalfW + 1
    w = kaiser(nW, beta = 2)  # Kaiser window (beta=2)
    w = w / np.sum(w)  # Normalize window

    # Max lag and lag indices
    maxDelta = int(np.ceil(tMaxLag * sampleFreq))
    indShift = np.arange(-maxDelta, maxDelta + 1)
    indShiftZero = maxDelta

    k = nHalfW + indShift[-1]

    # Buffer for eY
    nBuff = len(indShift)
    eYwB = np.zeros((len(w), nBuff))  # Buffer for windowed eY
    tFirst = k + 1

    # Pre-fill buffer
    for i in range(tFirst - maxDelta, tFirst + maxDelta + 1):
        index = 1 + (i % nBuff)
        eYwB[:, index - 1] = eY[(i - nHalfW):(i + nHalfW + 1)] * w

    # Cross-correlation and auto-correlation calculations
    RXY = np.zeros((len(eX), len(indShift)))
    RXX = np.zeros(len(eX))
    RYY = np.zeros(len(eX))

    for t in range(tFirst, len(eX) - k):
        # Add next eY segment to buffer
        tNewBuff = t + maxDelta
        index1 = 1 + (tNewBuff % nBuff)
        eYwB[:, index1 - 1] = eY[(tNewBuff - nHalfW):(tNewBuff + nHalfW + 1)] * w

        eXw = eX[(t - nHalfW):(t + nHalfW + 1)] * w
        index2 = 1 + (t + indShift) % nBuff
        eYw_shifted = eYwB[:, index2 - 1]

        RXY[t, :] = np.dot(eXw, eYw_shifted)
        RXX[t] = np.dot(eXw, eXw)

        index3 = 1 + (t % nBuff)
        eYw = eYwB[:, index3 - 1]
        RYY[t] = np.dot(eYw, eYw)

    # Convert shift indices to time values
    tShift = indShift * 2 / sampleFreq

    return RXY, tShift, indShift, RXX, RYY

