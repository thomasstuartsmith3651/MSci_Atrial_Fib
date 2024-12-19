#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:49:43 2024

@author: candace_chung
"""

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
class LoadDataExcel_MOVING:
    def __init__(self, data): #input name of excel file
        def electrodeData_MOVING(self, data):
            """
            This function loads the excel file and extracts the following data:
            - x coordinates of electrode positions is a 2D dataframe
            - y coordinates of electrode positions is a 2D dataframe
            - paired (x, y) coordinates stored as an array (row = electrode, column = time)
            - electrode signals is a 2D dataframe with row = electrode number and column = measurement at certain time
            - list of timestamps calculated from the sampling frequency is a 1D array
            
            Coord is created such that the position data is matched to the signal data according to the time of measurement
            This means that coord is expanded such that the number of columns matches that of signals
            
            Same thing for x and y
            """
            # Load the X, Y, and signal data
            x = pd.read_excel(data, sheet_name=1).transpose()
            y = pd.read_excel(data, sheet_name=2).transpose()
            signals = pd.read_excel(data, sheet_name=0).transpose().to_numpy()
        
            # Calculate time intervals for signals and positions
            sig_t_interval = 1 / 2034.5  # Sampling frequency is 2034.5 Hz
            sigTime = np.arange(0, signals.shape[1] * sig_t_interval, sig_t_interval)
        
            pos_t_interval = 1 / 101.725  # Sampling frequency is 101.725 Hz
            posTime = np.arange(0, x.shape[1] * pos_t_interval, pos_t_interval)
        
            # Align position indices to signal times using np.searchsorted (for np.searchsorted(a,v) side = right --> a[i-1] <= v < a[i])
            aligned_indices = np.searchsorted(posTime, sigTime, side='right') - 1
            aligned_indices[aligned_indices < 0] = 0  # Ensure no negative indices
        
            # Expand X and Y to match the signal time indices
            aligned_x = x.values[:, aligned_indices]
            aligned_y = y.values[:, aligned_indices]
        
            # Combine X and Y into coord (electrode, time, coordinate_pair)
            coord = np.dstack((aligned_x, aligned_y))
            return aligned_x, aligned_y, coord, signals, sigTime, posTime

        #MOVING ELECTRODE DATA
        self.x_pos, self.y_pos, self.coord, self.signals, self.sigTime, self.posTime = electrodeData_MOVING(self, data)
    
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
        - row = electrode, column = time
        """
        return self.coord
    
    def ele_signals(self):
        """
        This function calls the 2D array of electrode signals outside of the class
        
        Rows = electrode number 
        Columns = measurement at certain time
        """
        return self.signals
    
    def sig_time_data(self):
        """
        This function calls the 1D array of signal timestamps outside of the class
        """
        return self.sigTime
    
    def pos_time_data(self):
        """
        This function calls the 1D array of position timestamps outside of the class
        """
        return self.posTime

#%%

"""
THIS MAY NOT WORK, SO MAY NEED TO CHANGE LATER
"""

class AnalyseDataExcel_MOVING(LoadDataExcel_MOVING): #perform wavelet transform on data
    def __init__(self, fileName, minVelocity, maxVelocity, window_length, corr_threshold): #velocity in mm/s angles in radians
        LoadDataExcel_MOVING.__init__(self, fileName)
        
        self.sigSampFreq = 2034.5 #units of Hz
        self.posSampFreq = 101.725 #units of Hz
        self.sigSampInterval = 1/self.sigSampFreq
        self.posSampInterval = 1/self.posSampFreq
        self.window_length = window_length #number of indices
        self.minVelocity = minVelocity
        self.maxVelocity = maxVelocity
        self.corr_threshold = corr_threshold #percentage of maximum RXY that the RXY used to compute time delay after windowing must have for the time delay to be valid
        
    def findEGMPeak(self, e1, height_threshold = 0.7, distance_threshold = 300): #change to 300 for real data
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
        - ind_shift = indices to apply kaiser window so it's centred around the peaks
        
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
        RXY = sps.correlate(e2, e1, mode = "full", method = "direct")
        index_delays = sps.correlation_lags(len(e2), len(e1), mode = "full")
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
        
        RXY_threshold = self.corr_threshold * np.max(RXY) # FIND CROSS-CORRELATION THRESHOLD
        if max_RXY < RXY_threshold:
            best_timeDelay = np.inf
        else:
            best_timeDelay = best_indexDelay * self.sigSampInterval
        #print(best_timeDelay, best_indexDelay)
        return best_timeDelay, best_indexDelay, max_RXY
    
    def electrodePairVelocity(self, ele_num1, ele_num2, ind_shift):
        """
        This function creates a vector for wave velocity between pair of electrodes in m/s
        Input:
        - ele_num1 = number for first electrode (0 - 15)
        - ele_num2 = number for second electrode (0 - 15)
        - ind_shift = index for time at which the velocity is calculated
        
        Output: velocity vector
        - Row 1 = x-component of velocity
        - Row 2 = y-component of velocity
        
        Can modify function to include z-coordinate later
        """
        d_vector = self.coord[ele_num2, ind_shift, :] - self.coord[ele_num1, ind_shift, :]
        d_mag = np.linalg.norm(d_vector)
        #print("ind_shift", ind_shift, "old dist", d_mag, d_vector/d_mag)
        minTimeDelay = d_mag * 0.001 / self.maxVelocity
        maxTimeDelay = d_mag * 0.001 / self.minVelocity
        
        e1 = self.signals[ele_num1]
        e2 = self.signals[ele_num2]
        
        # NEED TO WINDOW SIGNAL
        e1_w, e2_w = self.windowSignal(e1, e2, ind_shift) # DO FIRST PEAK FOR NOW

        RXY, ind_delays = self.simpleCorrelate(e1_w, e2_w)
        best_t_delay, best_i_delay, max_RXY = self.maxRXY_timeDelay(RXY, ind_delays, minTimeDelay, maxTimeDelay)
        
        if best_i_delay > 0:
            new_d_vector = self.coord[ele_num2, ind_shift + best_i_delay, :] - self.coord[ele_num1, ind_shift, :]
            new_d_mag = np.linalg.norm(new_d_vector)
        elif best_i_delay < 0:
            new_d_vector = self.coord[ele_num2, ind_shift, :] - self.coord[ele_num1, ind_shift + abs(best_i_delay), :]
            new_d_mag = np.linalg.norm(new_d_vector)
        
        speed = new_d_mag/best_t_delay
        direction_unit_vector = new_d_vector/new_d_mag
        #print("dist", new_d_mag, direction_unit_vector)
        velocity_vector = speed * direction_unit_vector * 0.001  #convert from mm/s to m/s
        #print(velocity_vector)
        return velocity_vector, max_RXY
    
    def guessVelocity_LSQ(self, ref_ele_num, ele_num1, ele_num2, peak_num):
        """
        This function combines two vectors measured from two electrodes with respect to a reference electrode
        NOTE: THIS FUNCTION ONLY WORKS IF ele_num2 IS ABOVE ele_num1
        Input:
        - ref_ele_num = number for reference electrode (0 - 15)
        - ele_num1 = number for first electrode
        - ele_num2 = number for second electrode (ele_num2 must be above ele_num1)
        - peak_num = which peak to window the time-series signal around

        Output: velocity vector estimate
        - Row 1 = x-component of velocity
        - Row 2 = y-component of velocity
        """
        e1 = self.signals[ref_ele_num]
        ind_shifts = self.findEGMPeak(e1)
        ind_shift = ind_shifts[peak_num]
        # THIS IS NOT HERE FOR STATIONARY ELECTRODES
        ref_origin = self.coord[ref_ele_num, ind_shift, :]
        velocity1, max_RXY1 = self.electrodePairVelocity(ref_ele_num, ele_num1, ind_shift)
        velocity2, max_RXY2 = self.electrodePairVelocity(ref_ele_num, ele_num2, ind_shift)
        #print(velocity1, velocity2)
        wavefront_vector = velocity2 - velocity1
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
    
            print((np.degrees(np.arccos(guess_unit_vector[0])) + np.degrees(np.arcsin(guess_unit_vector[1])))/2)
        return v_guess, ref_origin #ref_origin IS NOT AN OUTPUT FOR STATIONARY ELECTRODES
    
    def velocityGuessMap(self, peak_num):
        origins = []
        velocity_vectors = []
        for ref_ele_num in range(3):
            for i in range(3):
                # FOR STATIONARY ELECTRODES
                #ref_origin = self.coord[ref_ele_num] + np.full(2, 1)
                #origins.append(ref_origin)
                
                ele_1 = ref_ele_num + 4
                ele_2 = ref_ele_num + 1
                
                # FOR STATIONARY ELECTRODES
                #v_guess = self.guessVelocity_LSQ(ref_ele_num, ele_1, ele_2, peak_num)
                
                #FOR MOVING ELECTRODES
                v_guess, ref_origin = self.guessVelocity_LSQ(ref_ele_num, ele_1, ele_2, peak_num)
                velocity_vectors.append(v_guess)
                vector_origin = ref_origin + np.full(2, 1)
                origins.append(vector_origin)
                
                ref_ele_num += 5
                
                # FOR STATIONARY ELECTRODES
                #ref_origin = self.coord[ref_ele_num] + np.full(2, 1)
                #origins.append(ref_origin)
                
                ele_1 = ref_ele_num - 4
                ele_2 = ref_ele_num - 1 

                # FOR STATIONARY ELECTRODES
                #v_guess = self.guessVelocity_LSQ(ref_ele_num, ele_1, ele_2, peak_num)
                
                #FOR MOVING ELECTRODES
                v_guess, ref_origin = self.guessVelocity_LSQ(ref_ele_num, ele_1, ele_2, peak_num)
                velocity_vectors.append(v_guess)
                vector_origin = ref_origin - np.full(2, 1)
                origins.append(vector_origin)
                
                ref_ele_num -= 1
        return velocity_vectors, origins
