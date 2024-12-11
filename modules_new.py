#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:14:02 2024

@author: candace_chung
"""
import pickle
import numpy as np
from scipy.io import loadmat
import pywt
import pandas as pd
from scipy.signal.windows import kaiser
import scipy.interpolate as spi
from scipy.signal import correlate
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

#%%

#class

class loadData: #load matlab files into python
    def __init__(self, fileName): #input name of matlab file
        def electrodeData(fileName):
            """
            This function loads the matlab file and extracts the following data:
            - x coordinates of electrode positions is a 2D array
            - y coordinates of electrode positions is a 2D array
            - z coordinates of electrode positions is a 2D array
            - electrode signals is a 2D array
            - list of time for position data
            - list of time for signal data
            
            Data matlab files are stored in column-major format
            Numpy is based on C and uses row-major, so 2D arrays are stored with:
            - row as electrode number
            - column as measurements at certain times
            """
            #below arrays are all in column-major format
            data = loadmat(fileName)
            x_arr = data["X"]
            y_arr = data["Y"]
            z_arr = data["Z"]
            signals = data["S"]
            posTime_arr = data["pT"][0]
            sigTime_arr = data["sT"][0]
            posSampFreq = data["pF"][0, 0]
            sigSampFreq = data["sF"][0, 0]
            
            #transpose into row-major format
            x_arr = np.transpose(x_arr)
            y_arr = np.transpose(y_arr)
            z_arr = np.transpose(z_arr)
            signals = np.transpose(signals)
            
            return x_arr, y_arr, z_arr, signals, posTime_arr, sigTime_arr, posSampFreq, sigSampFreq
        
        self.x_pos, self.y_pos, self.z_pos, self.signals, self.pos_time, self.sig_time, self.posSampFreq, self.sigSampFreq = electrodeData(fileName)
    
    def x_position(self):
        """
        This function calls the 2D array of x coordinates of electrode positions outside of the class
        """
        return self.x_pos
    
    def y_position(self):
        """
        This function calls the 2D array of y coordinates of electrode positions outside of the class
        """
        return self.y_pos
    
    def z_position(self):
        """
        This function calls the 2D array of z coordinates of electrode positions outside of the class
        """
        return self.z_pos
    
    def ele_signals(self):
        """
        This function calls the 2D array of electrode signals outside of the class
        """
        return self.signals
    
    def pos_time_arr(self):
        """
        This function calls the 1D array of timestamps outside of the class
        """
        return self.pos_time
    
    def sig_time_arr(self):
        """
        This function calls the 1D array of timestamps outside of the class
        """
        return self.sig_time
    
    def pos_samp_freq(self):
        """
        This function calls the position sampling frequency outside of the class
        """
        return self.posSampFreq
    
    def sig_samp_freq(self):
        """
        This function calls the signal sampling frequency outside of the class
        """
        return self.sigSampFreq

#%%
class AnalyseData(loadData): #perform wavelet transform on data
    def __init__(self, fileName, velocityRange, angleRange): #angles in radians
        loadData.__init__(self, fileName)
        self.minVelocity, self.maxVelocity = velocityRange[0], velocityRange[1]
        self.minAngle, self.maxAngle = angleRange[0], angleRange[1]
    
    def electrodeDistance(self, t_ind, ele_num1, ele_num2):
        """
        This function creates a 1D vector for electrode distances
        Input:
        - t_ind = index in time array
        - ele_num1 = number for first electrode (0 - 15)
        - ele_num2 = number for second electrode (0 - 15)
        
        Output: 1D vector for electrode distances (difference)
        - Row 1 = x-coordinate distance at time with index t_ind
        - Row 2 = y-coordinate distance at time with index t_ind
        
        Can modify function to include z-coordinate later
        """
        posX = self.x_pos[ele_num1] - self.x_pos[ele_num2]
        posY = self.y_pos[ele_num1] - self.y_pos[ele_num2]
        d = np.array([posX[t_ind], posY[t_ind]])
        return d
    
    def timeDelay(self, t_ind, ele_num1, ele_num2, vel, ang):
        """
        This function calculates time delay at a certain time from distance, velocity, and angle
        Input:
        - t_ind = index in time array
        - ele_num1 = number for first electrode (0 - 15)
        - ele_num2 = number for second electrode (0 - 15)
        - vel = magnitude of velocity in m/s
        - ang = angle in radians
        
        Output: 
        - t_delay = time delay in seconds
        
        Can modify function to include z-coordinate later
        """
        d = self.electrodeDistance(t_ind, ele_num1, ele_num2)
        v_unit = np.array([np.cos(ang), np.sin(ang)])
        t_delay = np.dot(d, v_unit) / vel
        return t_delay
    
    def shiftSignal(self, e1, e2, t_delay):
        """
        This function shifts signal of electrode by a set time delay
        Signals are shifted by padding one end with the signal average
        Features at the end of the signal are cut off
        Fractional shifts use interpolation
        Input:
        - t_delay = time delay in seconds
        - eY = signal array of electrode eY
        
        Output: 
        - shifted_e2 = shifted signal array of electrode e2
        """
        index_offset = t_delay * self.sigSampFreq
        int_offset = int(np.floor(index_offset))
        frac_offset = index_offset - int_offset   # Fractional part
        average_e2 = np.average(e2)
        # Integer shift with average-padding
        if int_offset > 0:
            shifted_e2 = np.concatenate((np.full(int_offset, average_e2), e2[:-int_offset]))
        elif int_offset < 0:
            shifted_e2 = np.concatenate((e2[-int_offset:], np.full(-int_offset, average_e2)))
        else:
            shifted_e2 = e2.copy()
        
        #THIS PART IS USELESS
        # Fractional shift using interpolation
        if frac_offset != 0:
            shifted_e2 = (1 - frac_offset) * shifted_e2 + frac_offset * np.roll(shifted_e2, -1)
        
        # Truncate to original length
        shifted_e2 = shifted_e2[:len(e1)]
        
        return shifted_e2
    
    def shiftSignal2(self, e2, t_delay):
        """
        This function shifts signal of electrode by a set time delay
        Signals are shifted by shifting the indices and rolling
        Also generates tuple of indices mapping e1 to shifted e2 to read e2
        Input:
        - t_delay = time delay in seconds
        - e2 = signal array of electrode e2
        
        Output: 
        - shifted_e2 = shifted signal array of electrode eY
        - paired_ind = tuple of indices mapping e1 to shifted e2
          - format of paired_ind is (index of e1, index of e2)
        """
        N = len(e2)
        index_offset = t_delay * self.sigSampFreq
        indices = np.arange(N)

        #compute the fractional shift indices
        shifted_indices = (indices - index_offset) % N  #wrap around
        
        #read the shifted signal using the fractional indices
        shifted_e2 = np.interp(shifted_indices, indices, e2)
        
        #map the indices of e1 to e2 (sorted so start from index 0 for e1)
        #can remove sorting if it doesn't work
        paired_ind = sorted([(int(round(shifted_index)) % N, shifted_index) for shifted_index in shifted_indices])
        
        return shifted_e2, paired_ind
    
    """
    NEED TO WRITE SO IT READS THE SHIFTED INDICES FOR eY
    """
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
    
    # def egmcorr(self, eX, eY, sampleFreq, tWindowWidth, tMaxLag):
    #     """
    #     Python version of Nick Linton's egm corr MATLAB code.

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
    #     w = kaiser(nW, beta = 2)  # Kaiser window (beta=2)
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

