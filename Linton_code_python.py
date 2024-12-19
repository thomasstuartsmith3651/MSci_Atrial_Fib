#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:11:29 2024

@author: candace_chung
"""

import numpy as np
from scipy.signal import convolve2d
from scipy.signal.windows import kaiser

def sgolaymm(order, frameLen, observermatrix, weights):
    """
    SGOLAYMM Savitzky-Golay Filter Design incorporating Measurement Model.
    
    Parameters:
        order (int): Order of the polynomial fitting.
        frameLen (int): Number of samples in each frame (must be odd and > order).
        observermatrix (np.ndarray): Observer matrix (J states by K measurements).
        weights (np.ndarray): Weights for least squares, size = frameLen x nMeasurements.
    
    Returns:
        M3 (list): A list of matrices for each state, used for convolution with the signal.
        P (np.ndarray): Transformation matrix P.
        Q (np.ndarray): Transformation matrix Q.
    """
    debug = False
    
    # Validate inputs
    if (frameLen - 1) % 2 != 0:
        raise ValueError("frameLen must be odd.")
    if not isinstance(order, int) or order < 0 or order >= frameLen:
        raise ValueError("order must be a non-negative integer and less than frameLen.")
    if weights.shape != (frameLen, observermatrix.shape[1]):
        raise ValueError("weights must have shape (frameLen, nMeasurements).")
    
    H = observermatrix
    nJ, nK = H.shape  # Number of states (rows of H) and measurements (columns of H)
    
    # Create Vandermonde matrix
    z = np.arange(-(frameLen - 1) / 2, (frameLen - 1) / 2 + 1)
    V = np.vander(z, order + 1, increasing=True)
    
    if debug:
        Atest = np.random.rand(order + 1, nJ)
        Xtest = V @ Atest
        Ytest = Xtest @ H
    
    P = np.zeros(((order + 1) * nJ, frameLen * nK))
    Q = np.zeros(((order + 1) * nJ, (order + 1) * nJ))
    ZETAk = np.zeros((order + 1, (order + 1) * nJ))
    
    # Populate Q
    for k in range(nK):
        Wk = np.diag(weights[:, k])
        PHIk = V.T @ Wk @ V
        Hk = H[:, k]
        
        for ind in range(nJ):
            cols = slice(ind * (order + 1), (ind + 1) * (order + 1))
            ZETAk[:, cols] = Hk[ind] * PHIk
        
        for j in range(nJ):
            rows = slice(j * (order + 1), (j + 1) * (order + 1))
            Q[rows, :] += H[j, k] * ZETAk
    
    # Populate P
    for k in range(nK):
        Wk = np.diag(weights[:, k])
        PHIk = V.T @ Wk
        for j in range(nJ):
            cols = slice(k * frameLen, (k + 1) * frameLen)
            rows = slice(j * (order + 1), (j + 1) * (order + 1))
            P[rows, cols] = H[j, k] * PHIk
    
    # Solve for M
    M = np.linalg.solve(Q, P)
    
    if debug:
        diff1 = P @ Ytest.flatten() - Q @ Atest.flatten()
        diff2 = Atest.flatten() - M @ Ytest.flatten()
        print("Debug differences:", diff1, diff2)
    
    isRowNeeded = np.zeros((order + 1) * nJ, dtype=bool)
    isRowNeeded[::order + 1] = True
    M2 = M[isRowNeeded, :]
    
    # Generate M3
    M3 = []
    for j in range(nJ):
        M3J = np.zeros((frameLen, nK))
        for k in range(nK):
            kInd = slice(k * frameLen, (k + 1) * frameLen)
            M3J[:, k] = M2[j, kInd]
        M3.append(M3J)
    
    return M3, P, Q

def sgolaymmfilt(Y, order, frameLen, observermatrix, weights):
    """
    Implements Savitzky-Golay Filter with Measurement Model.
    
    Parameters:
        Y (ndarray): Data of shape (n, k), where there are k measurements.
        order (int): Order of the polynomial fitting.
        frameLen (int): Number of samples in each frame (must be odd and > order).
        observermatrix (ndarray): Measurement matrix, size (nJ, nK).
        weights (ndarray): Weighting for least squares, size (frameLen, nK).
        
    Returns:
        X (ndarray): Estimate of states such that Y_est = X @ observermatrix.
        Y_est (ndarray): Estimate of underlying 'measurements', Y_est = X @ observermatrix.
    """
    # Get the filter
    M = sgolaymm(order, frameLen, observermatrix, weights)
    
    # Initialize X
    n_samples, n_measurements = Y.shape
    n_states = observermatrix.shape[0]
    X = np.zeros((n_samples, n_states))
    
    # Apply convolution to estimate X
    for j in range(n_states):
        for k in range(n_measurements):
            X[:, j] += convolve2d(Y[:, [k]], M[j][:, [k]], mode='same').flatten()
    
    # If a second output is requested, calculate Y_est
    Y_est = None
    if observermatrix is not None:
        Y_est = X @ observermatrix
    
    return X, Y_est


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

