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
            """ FOR TESTING ONLY """
            #t_interval = 20/1000 #sampling frequency is 2034.5 Hz
            """ FOR TESTING ONLY """
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
    def __init__(self, fileName, minVelocity, maxVelocity, corr_threshold): #velocity in mm/s angles in radians
        LoadDataExcel.__init__(self, fileName)
        
        self.sigSampFreq = 2034.5 #units of Hz
        self.sigSampInterval = 1/self.sigSampFreq
        self.window_length = 814 #number of indices
        self.minVelocity = minVelocity
        self.maxVelocity = maxVelocity
        self.corr_threshold = corr_threshold
    
    def findEGMPeak(self, e1, height_threshold = 0.5, distance_threshold = 300):
        """
        This function finds the peaks in the EGM signal and outputs indices to apply the kaiser window
        Input:
        - e1 = time series for first electrode
        - height_threshold = minimum signal height to identify peak
        - distance_threshold = minimum peak separation for it to be counted
        
        Output: 
        - ind_shifts = indices to apply kaiser window so it's centred around the peaks
        
        Can modify function to include z-coordinate later
        """
        peaks, properties = sps.find_peaks(e1, height = height_threshold, distance = distance_threshold)
        ind_shifts = [peak_index - self.window_length // 2 for peak_index in peaks]
        return ind_shifts
    
    def windowSignal(self, e1, e2, ind_shift):
        """
        This function windows the signals by applying a kaiser window
        Input:
        - e1 = time series for first electrode
        - e2 = time series for second electrode
        
        Output: 
        - e1_windowed = 1D array of windowed time series for first electrode
        - e2_windowed = 1D array of windowed time series for second electrode
        
        Can modify function to include z-coordinate later
        """
        def shift_window(array_length, window, ind_shift):
            padded_window = np.zeros(array_length)
            w_length = len(window)
            start = max(0, ind_shift)
            end = min(array_length, ind_shift + w_length)
            w_start = max(0, -ind_shift)
            w_end = w_start + (end - start)
            if end > start and w_end > w_start:
                padded_window[start:end] = window[w_start:w_end]
            return padded_window
        N = len(e1)
        kaiser_window = kaiser(self.window_length, beta = 0)
        padded_kaiser = shift_window(N, kaiser_window, ind_shift)
        e1_windowed = e1 * padded_kaiser
        e2_windowed = e2 * padded_kaiser
        return e1_windowed, e2_windowed
    
    def simpleCorrelate(self, e1, e2):
        """
        This function cross-correlates the signals and outputs the index delays
        Input:
        - e1 = time series for first electrode
        - e2 = time series for second electrode
        
        Output: 
        - RXY = 1D array of discrete linear cross-correlations
        - index_delays = index shift for cross-corelation
        
        Can modify function to include z-coordinate later
        """
        norm_factor = np.sqrt(np.sum(e1**2) * np.sum(e2**2))
        RXY = sps.correlate(e2, e1, mode = "full", method = "direct")/norm_factor
        index_delays = sps.correlation_lags(len(e1), len(e2), mode = "full")
        return RXY, index_delays

    def maxRXY_timeDelay(self, RXY, index_delays, minTimeDelay, maxTimeDelay):
        """
        This function finds the time delay that maximises cross-correlation
        Non-sensical time delays are filtered out by setting it to infinity if the index shift is 0
        Input:
        - RXY = 1D array of discrete linear cross-correlations
        - index_delays = index shift for cross-corelation
        - minTimeDelay = minimum time delay threshold so only look at reasonable velocity values
        
        Output:
        - best_timeDelay = time delay in seconds
        - max_RXY = maximum cross-correlation value within the time window considered
        """
        # Set threshold for valid time delays
        min_index_shift = int(np.ceil(minTimeDelay * self.sigSampFreq))
        max_index_shift = int(np.floor(maxTimeDelay * self.sigSampFreq))
        
        max_possible_index_shift = max(index_delays)
        
        if max_index_shift > max_possible_index_shift:
            max_index_shift = max_possible_index_shift
        
        max_neg_index_threshold = np.where(index_delays == -min_index_shift)[0][0]
        min_neg_index_threshold = np.where(index_delays == -max_index_shift)[0][0]
        
        min_pos_index_threshold = np.where(index_delays == min_index_shift)[0][0]
        max_pos_index_threshold = np.where(index_delays == max_index_shift)[0][0]
        
        neg_max_index = RXY[min_neg_index_threshold:max_neg_index_threshold + 1].argmax() + min_neg_index_threshold
        pos_max_index = RXY[min_pos_index_threshold:max_pos_index_threshold].argmax() + min_pos_index_threshold
        # Only find maximum within window
        if RXY[pos_max_index] >= RXY[neg_max_index]:
            best_indexDelay = index_delays[pos_max_index]
            max_RXY = RXY[pos_max_index]
        elif RXY[pos_max_index] < RXY[neg_max_index]:
            best_indexDelay = index_delays[neg_max_index]
            max_RXY = RXY[neg_max_index]
            
        if max_RXY < self.corr_threshold:
            best_timeDelay = np.inf
        else:
            best_timeDelay = best_indexDelay/self.sigSampFreq
        return best_timeDelay, max_RXY
    
    def electrodePairVelocity(self, ele_num1, ele_num2, ind_shift):
        """
        This function creates a vector for wave velocity between pair of electrodes in m/s
        Input:
        - ele_num1 = number for first electrode (0 - 15)
        - ele_num2 = number for second electrode (0 - 15)
        - maxVelocity = maximum allowable velocity
        
        Output: velocity vector
        - Row 1 = x-component of velocity
        - Row 2 = y-component of velocity
        
        Can modify function to include z-coordinate later
        """
        d_vector = self.coord[ele_num2] - self.coord[ele_num1]
        d_mag = np.linalg.norm(d_vector)

        minTimeDelay = d_mag * 0.001 /self.maxVelocity
        maxTimeDelay = d_mag * 0.001 /self.minVelocity
        
        e1 = self.signals[ele_num1]
        e2 = self.signals[ele_num2]
        
        # NEED TO WINDOW SIGNAL
        e1_w, e2_w = self.windowSignal(e1, e2, ind_shift) # DO FIRST PEAK FOR NOW

        RXY, ind_delays = self.simpleCorrelate(e1_w, e2_w)
        best_t_delay, max_RXY = self.maxRXY_timeDelay(RXY, ind_delays, minTimeDelay, maxTimeDelay)
        
        speed = d_mag/best_t_delay
        direction_unit_vector = d_vector/d_mag

        velocity_vector = speed * direction_unit_vector * 0.001  #convert from mm/s to m/s

        return velocity_vector, max_RXY
    
    def guessVelocity_LSQ(self, ref_ele_num, ele_num1, ele_num2, peak_num):
        """
        This function combines two vectors measured from two electrodes with respect to a reference electrode
        NOTE: THIS FUNCTION ONLY WORKS IF ele_num2 IS ABOVE ele_num1
        Input:
        - ref_ele_num = number for reference electrode (0 - 15)
        - ele_num1 = number for first electrode
        - ele_num2 = number for second electrode (ele_num2 must be above ele_num1)
        - minVelocity = minimum allowable velocity
        - maxVelocity = maximum allowable velocity
        - peak_num = which peak to window the time-series signal around
        - corr_threshold = minimum correlation threshold for acceptable time delays
    
        Output: velocity vector estimate
        - Row 1 = x-component of velocity
        - Row 2 = y-component of velocity
        """
        e1 = self.signals[ref_ele_num]
        ind_shifts = self.findEGMPeak(e1)
        ind_shift = ind_shifts[peak_num]
        velocity1, max_RXY1 = self.electrodePairVelocity(ref_ele_num, ele_num1, ind_shift)
        velocity2, max_RXY2 = self.electrodePairVelocity(ref_ele_num, ele_num2, ind_shift)
        
        wavefront_vector = velocity2 - velocity1
        print(wavefront_vector, velocity2, velocity1)
        norm = np.linalg.norm(wavefront_vector)
        wavefront_unit_vector = (1 / norm) * wavefront_vector
    
        if wavefront_vector[0] == -velocity1[0] and wavefront_vector[1] == -velocity1[1]:
            v_guess = velocity1
        elif wavefront_vector[0] == velocity2[0] and wavefront_vector[1] == velocity2[1]:
            v_guess = velocity2
        else:
            rotation_matrix = np.array([[0, 1], [-1, 0]])
            guess_unit_vector = np.dot(rotation_matrix, wavefront_unit_vector)
    
            # Loss function
            def loss(v_mag_guess):
                cos_theta1 = np.dot(velocity1, guess_unit_vector) / np.linalg.norm(velocity1)
                cos_theta2 = np.dot(velocity2, guess_unit_vector) / np.linalg.norm(velocity2)
                ans = (v_mag_guess - magnitude1 * cos_theta1)**2 + (v_mag_guess - magnitude2 * cos_theta2)**2
                return ans
            
            magnitude1 = np.linalg.norm(velocity1)
            magnitude2 = np.linalg.norm(velocity2)
            initial_guess = (magnitude1 + magnitude2) / 2

            m = Minuit(loss, v_mag_guess=initial_guess)
            m.limits["v_mag_guess"] = (-self.maxVelocity, self.maxVelocity)  # Velocity limits (-v_max to v_max m/s)
            m.errordef = Minuit.LEAST_SQUARES  # Least-squares error definition
            m.migrad()

            v_mag_guess = m.values["v_mag_guess"]

            v_guess = v_mag_guess * guess_unit_vector
    
            print("angle", np.degrees(np.arccos(guess_unit_vector[0])), np.degrees(np.arcsin(guess_unit_vector[1])))
        return v_guess
    
    def velocityGuessMap(self, peak_num):
        origins = []
        velocity_vectors = []
        for ref_ele_num in range(3):
            for i in range(3):
                ref_origin = self.coord[ref_ele_num] + np.full(2, 1)
                origins.append(ref_origin)
                
                ele_1 = ref_ele_num + 4
                ele_2 = ref_ele_num + 1
                v_guess = self.guessVelocity_LSQ(ref_ele_num, ele_1, ele_2, peak_num)
                velocity_vectors.append(v_guess)
                
                ref_ele_num += 5
                ref_origin = self.coord[ref_ele_num] - np.full(2, 1)
                origins.append(ref_origin)
                
                ele_1 = ref_ele_num - 4
                ele_2 = ref_ele_num - 1 
                v_guess = self.guessVelocity_LSQ(ref_ele_num, ele_1, ele_2, peak_num)
                velocity_vectors.append(v_guess)
                ref_ele_num -= 1
        return velocity_vectors, origins

    