#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:12:38 2024

@author: candace_chung
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:14:02 2024

@author: candace_chung
"""
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

"""
For some reason my laptop can't find this file in my local directory so I need the line below to import the file.
"""
import sys 
sys.path.append('/Users/candace_chung/Desktop/Candace Chung Files/ICL/Academics/Year 4/MSci Project/code/MSci_Atrial_Fib')

#%%
class LoadDataExcel:
    def __init__(self, data): #input name of excel file
        def electrodeData(self, data):
            """
            This function loads the excel file and extracts the following data:
            - x coordinates of electrode positions is a 1D dataframe
            - y coordinates of electrode positions is a 1D dataframe
            - paired (x, y) coordinates is a 2D array
            - electrode signals is a 2D dataframe with row = electrode number and column = measurement at certain time
            - list of timestamps calculated from the sampling frequency is a 1D array
            """
            positions = pd.read_excel(data, sheet_name = 1)
            x = positions.iloc[0]
            y = positions.iloc[1]
            coord = positions.transpose().to_numpy()
            
            signals = pd.read_excel(data, sheet_name = 0)
            signals = signals.transpose().to_numpy()
            
            t_interval = 1/2034.5 #sampling frequency is 2034.5 Hz
            time = np.arange(0, signals.shape[1] * t_interval, t_interval)
            return x, y, coord, signals, time
        
        self.x_pos, self.y_pos, self.coord, self.signals, self.time = electrodeData(self, data)
    
    def x_position(self):
        """
        This function calls the 1D data frame of x coordinates of electrode positions outside of the class
        """
        return self.x_pos
    
    def y_position(self):
        """
        This function calls the 1D data frame of y coordinates of electrode positions outside of the class
        """
        return self.y_pos
    
    def coordinates(self):
        """
        This function calls the 2D array of (x, y) coordinates of electrode positions outside of the class
        """
        return self.coord
    
    def ele_signals(self):
        """
        This function calls the 2D array of electrode signals outside of the class
        
        Rows = electrode number 
        Columns = measurement at certain time
        """
        return self.signals
    
    def time_data(self):
        """
        This function calls the 1D array of timestamps outside of the class
        """
        return self.time

#%%

class AnalyseDataExcel(LoadDataExcel): #perform wavelet transform on data
    def __init__(self, fileName, velocityRange, angleRange, cpu_num): #velocity in mm/s angles in radians
        LoadDataExcel.__init__(self, fileName)
        self.minVelocity, self.maxVelocity = velocityRange[0], velocityRange[1]
        self.minAngle, self.maxAngle = angleRange[0], angleRange[1]
        self.sigSampFreq = 2034.5 #units of Hz
        self.cpu_num = cpu_num
    
    def electrodeDistance(self, ele_num1, ele_num2):
        """
        This function creates a 1D vector for electrode distances
        Input:
        - ele_num1 = number for first electrode (0 - 15)
        - ele_num2 = number for second electrode (0 - 15)
        
        Output: 1D vector for electrode distances (difference)
        - Column 1 = x-coordinate distance at time with index t_ind
        - Column 2 = y-coordinate distance at time with index t_ind
        
        Can modify function to include z-coordinate later
        """
        d = self.coord[ele_num1] - self.coord[ele_num2]
        return d
    
    def timeDelay(self, ele_num1, ele_num2, vel, ang):
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
        d = self.electrodeDistance(ele_num1, ele_num2)
        v_unit = np.array([np.cos(ang), np.sin(ang)])
        t_delay = np.dot(d, v_unit) / vel
        return t_delay
    
    def shiftSignal(self, N, ele_num2, t_delay):
        """
        This function shifts signal of electrode by a set time delay
        Signals are shifted by padding one end with the signal average
        Features at the end of the signal are cut off
        Input:
        - t_delay = time delay in seconds
        - N = length of first signal
        - ele_num2 = electrode number of second electrode
        
        Output: 
        - shifted_e2 = shifted signal array of electrode e2
        """
        e2 = self.signals[ele_num2]
        index_offset = t_delay * self.sigSampFreq
        int_offset = int(round(index_offset))
        average_e2 = np.average(e2)
        #integer shift with average-padding
        if int_offset > 0:
            shifted_e2 = np.concatenate((np.full(abs(int_offset), average_e2), e2[:-int_offset]))
        elif int_offset < 0:
            shifted_e2 = np.concatenate((e2[-int_offset:], np.full(abs(int_offset), average_e2)))
        else:
            shifted_e2 = e2.copy()
        #truncate to original length
        shifted_e2 = shifted_e2[:N]
        return shifted_e2
    
    def shiftSignal2(self, N, ele_num2, t_delay):
        """
        This function shifts signal of electrode by a set time delay
        Signals are shifted by padding one end with the signal average
        Features at the end of the signal are cut off
        Fractional shifts use interpolation
        Input:
        - t_delay = time delay in seconds
        - N = length of first signal
        - ele_num2 = electrode number of second electrode
        
        Output: 
        - shifted_e2 = shifted signal array of electrode e2
        """
        e2 = self.signals[ele_num2]
        index_offset = t_delay * self.sigSampFreq
        int_offset = int(np.floor(index_offset))
        frac_offset = index_offset - int_offset   # Fractional part
        average_e2 = np.average(e2)
        #integer shift with average-padding
        if int_offset > 0:
            shifted_e2 = np.concatenate((np.full(abs(int_offset), average_e2), e2[:-int_offset]))
        elif int_offset < 0:
            shifted_e2 = np.concatenate((e2[-int_offset:], np.full(abs(int_offset), average_e2)))
        else:
            shifted_e2 = e2.copy()
        """
        FRACTIONAL SHIFT CHANGES THE SIGNAL AMPLITUDE
        """    
        # Fractional shift using interpolation
        if frac_offset != 0:
            shifted_e2 = (1 - frac_offset) * shifted_e2 + frac_offset * np.roll(shifted_e2, -1)
        
        # Truncate to original length
        shifted_e2 = shifted_e2[:N]
        return shifted_e2
    
    def simpleShiftSignal(self, ele_num2, t_delay):
        """
        FRACTIONAL SHIFT CHANGES THE SIGNAL AMPLITUDE
        """
        N = len(self.signals[ele_num2])
        index_offset = t_delay * self.sigSampFreq
        original_ind = np.arange(N)
        shifted_ind = original_ind - index_offset
        
        #interpolate the signal at the shifted indices
        shifted_e2 = np.interp(shifted_ind, original_ind, self.signals[ele_num2], left = 0, right = 0)
        return shifted_e2
    
    def simpleCorrelate(self, ele_num1, ele_num2, vel, ang):
        e1 = self.signals[ele_num1]
        N = len(e1)
        t_delay = self.timeDelay(ele_num1, ele_num2, vel, ang)
        shifted_e2 = self.shiftSignal2(N, ele_num2, t_delay)
        #shifted_e2 = self.simpleShiftSignal(ele_num2, t_delay)
        e1 = np.asarray(e1).flatten()
        shifted_e2 = np.asarray(shifted_e2).flatten()
    
        #perform the correlation calculation directly using NumPy's outer product
        RXY = np.outer(e1, shifted_e2.conj())
        return np.sum(RXY)
    
    def crossCorrelationMatrix(self, ele_num1, ele_num2, num_vel, num_ang):
        velocities = np.linspace(self.minVelocity, self.maxVelocity, num_vel)
        angles = np.linspace(self.minAngle, self.maxAngle, num_ang)
        tasks = [(v, theta) for v in velocities for theta in angles]
        results = Parallel(n_jobs = self.cpu_num)(delayed(self.simpleCorrelate)(ele_num1, ele_num2, v, theta) for v, theta in tasks)
        RXY_matrix = np.array(results).reshape(len(velocities), len(angles)) #reshape the results into a 2D matrix
        X, Y = np.meshgrid(angles, velocities) #meshgrid of angles and velocities
        VX, VY = Y * np.cos(angles), Y * np.sin(angles)
        return X, Y, VX, VY, RXY_matrix
    
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
    