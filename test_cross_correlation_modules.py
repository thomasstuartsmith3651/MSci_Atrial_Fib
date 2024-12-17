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
            m.limits["v_mag_guess"] = (-self.maxVelocity, self.maxVelocity)  # Velocity limits (-2 to 2 m/s)
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
    
    # def guessVelocity_LSQ(self, ref_ele_num, ele_num1, ele_num2, minVelocity, maxVelocity, peak_num, corr_threshold, alpha = 0.1, tolerance = 1e-6, max_iterations = 1000):
    #     """
    #     This function combines two vectors measured from two electrodes with respect to a reference electrode
    #     Velocity magnitude is guessed using least squares minimisation
    #     NOTE: THIS FUNCTION ONLY WORKS IF ele_num2 IS ABOVE ele_num1
    #     Input:
    #     - ref_ele_num = number for reference electrode (0 - 15)
    #     - ele_num1 = number for first electrode
    #     - ele_num2 = number for second electrode (ele_num2 must be above ele_num1)
    #     - minVelocity = minimum allowable velocity
    #     - maxVelocity = maximum allowable velocity
    #     - corr_threshold = minimum allowable cross-correlation for valid time delay
    #     - alpha = learning rate
    #     - tolerance = maximum allowable error
    #     - max_iterations = maximum number of iterations to compute least squares
        
    #     Output: velocity vector estimate
    #     - Row 1 = x-component of velocity
    #     - Row 2 = y-component of velocity
    #     """
    #     e1 = self.signals[ref_ele_num]
    #     ind_shifts = self.findEGMPeak(e1)
    #     ind_shift = ind_shifts[peak_num]
    #     velocity1, max_RXY1 = self.electrodePairVelocity(ref_ele_num, ele_num1, minVelocity, maxVelocity, ind_shift, corr_threshold)
    #     velocity2, max_RXY2 = self.electrodePairVelocity(ref_ele_num, ele_num2, minVelocity, maxVelocity, ind_shift, corr_threshold)
        
    #     wavefront_vector = velocity2 - velocity1
    #     norm = np.linalg.norm(wavefront_vector)
    #     wavefront_unit_vector = (1 / norm) * wavefront_vector
    
    #     if wavefront_vector[0] == -velocity1[0] and wavefront_vector[1] == -velocity1[1]:
    #         v_guess = velocity1
    #     elif wavefront_vector[0] == velocity2[0] and wavefront_vector[1] == velocity2[1]:
    #         v_guess = velocity2
    #     else:
    #         rotation_matrix = np.array([[0, 1], [-1, 0]])
    #         guess_unit_vector = np.dot(rotation_matrix, wavefront_unit_vector)

    #         cos_theta1 = np.dot(velocity1, guess_unit_vector) / np.linalg.norm(velocity1)
    #         cos_theta2 = np.dot(velocity2, guess_unit_vector) / np.linalg.norm(velocity2)
            
    #         magnitude1 = np.linalg.norm(velocity1)
    #         magnitude2 = np.linalg.norm(velocity2)
    
    #         # Gradient descent for minimization
    #         v_mag_guess = (magnitude1 + magnitude2) / 2  # Initial guess
            
    #         for iteration in range(max_iterations):
    #             # Compute gradient of the loss function
    #             gradient = 2 * (v_mag_guess - magnitude1 * cos_theta1) + 2 * (v_mag_guess - magnitude2 * cos_theta2)
                
    #             # Update velocity magnitude guess
    #             v_mag_guess -= alpha * gradient
                
    #             # Check for convergence
    #             if abs(gradient) < tolerance:
    #                 break
            
    #         # Compute final velocity vector
    #         v_guess = v_mag_guess * guess_unit_vector
    #         print("angle", np.degrees(np.arccos(guess_unit_vector[0])), np.degrees(np.arcsin(guess_unit_vector[1])))
    #     return v_guess
    
    # def guessVelocity_ORTHOGONAL(self, ref_ele_num, ele_num1, ele_num2, minVelocity, maxVelocity, peak_num, corr_threshold):
    #     """
    #     This function combines two vectors measured from two electrodes with respect to a reference electrode
    #     NOTE: THIS FUNCTION ONLY WORKS IF ele_num1 AND ele_num2 ARE ORTHOGONAL TO EACH OTHER
    #     Input:
    #     - ref_ele_num = number for reference electrode (0 - 15)
    #     - ele_num1 = number for first electrode (one above or to right of reference electrode)
    #     - ele_num2 = number for second electrode (one above or to right of reference electrode)
    #     - minVelocity = minimum allowable velocity
    #     - maxVelocity = maximum allowable velocity
        
    #     Output: velocity vector estimate
    #     - Row 1 = x-component of velocity
    #     - Row 2 = y-component of velocity
    #     """
    #     e1 = self.signals[ref_ele_num]
    #     ind_shifts = self.findEGMPeak(e1)
    #     ind_shift = ind_shifts[peak_num]
    #     velocity1, max_RXY1 = self.electrodePairVelocity(ref_ele_num, ele_num1, minVelocity, maxVelocity, ind_shift, corr_threshold)
    #     velocity2, max_RXY2 = self.electrodePairVelocity(ref_ele_num, ele_num2, minVelocity, maxVelocity, ind_shift, corr_threshold)
        
    #     sum_vector = velocity1 + velocity2
    #     norm = np.linalg.norm(sum_vector)
    #     sum_unit_vector = (1/norm) * sum_vector
    #     """
    #     FIRST IF: CHECK FOR HORIZONTAL VELOCITY VECTOR
    #     SECOND IF: CEHCK FOR VERTICAL VELOCITY VECTOR
    #     ELSE: USE SIMILAR TRIANGLE TRIGONOMETRY TO WORK OUT WHAT THE VELOCITY VECTOR IS (ASSUMING ELECTRODE MEASURES A PLANE WAVE)
    #     """
    #     if sum_vector[0] == velocity1[0] and sum_vector[1] == velocity1[1]:
    #         v_guess = velocity1
    #         print("v1")
    #     elif sum_vector[0] == velocity2[0] and sum_vector[1] == velocity2[1]:
    #         v_guess = velocity2
    #         print("v2")
    #     else:
    #         measured_ang1, measured_ang2 = np.arccos(sum_unit_vector[0]), np.arcsin(sum_unit_vector[1])
    #         propagation_ang1, propagation_ang2 = np.pi/2 - measured_ang1, np.pi/2 - measured_ang2
    #         guess_unit_vector = np.array([np.cos(propagation_ang1), np.sin(propagation_ang2)])
    #         # calculate velocity vector magnitude
    #         v_mag_guess = np.multiply(sum_vector, guess_unit_vector) # vector form of v = v1 * cos(alpha) and v = v2 * sin(alpha)
    #         print(np.degrees(propagation_ang1), np.degrees(propagation_ang2), v_mag_guess)
    #         # calculate velocity vector
    #         v_guess = np.multiply(v_mag_guess, guess_unit_vector) # calculate v * (cos(alpha), sin(alpha))
    #     return v_guess
    
    # def guessVelocity(self, ref_ele_num, ele_num1, ele_num2, minVelocity, maxVelocity, peak_num, corr_threshold):
    #     """
    #     This function combines two vectors measured from two electrodes with respect to a reference electrode
    #     NOTE: THIS FUNCTION ONLY WORKS IF ele_num2 IS ABOVE ele_num1
    #     Input:
    #     - ref_ele_num = number for reference electrode (0 - 15)
    #     - ele_num1 = number for first electrode
    #     - ele_num2 = number for second electrode (ele_num2 must be above ele_num1)
    #     - minVelocity = minimum allowable velocity
    #     - maxVelocity = maximum allowable velocity
        
    #     Output: velocity vector estimate
    #     - Row 1 = x-component of velocity
    #     - Row 2 = y-component of velocity
    #     """
    #     e1 = self.signals[ref_ele_num]
    #     ind_shifts = self.findEGMPeak(e1)
    #     ind_shift = ind_shifts[peak_num]
    #     velocity1, max_RXY1 = self.electrodePairVelocity(ref_ele_num, ele_num1, minVelocity, maxVelocity, ind_shift, corr_threshold)
    #     velocity2, max_RXY2 = self.electrodePairVelocity(ref_ele_num, ele_num2, minVelocity, maxVelocity, ind_shift, corr_threshold)
        
    #     wavefront_vector = velocity2 - velocity1
    #     norm = np.linalg.norm(wavefront_vector)
    #     wavefront_unit_vector = (1/norm) * wavefront_vector
    #     """
    #     FIRST IF: CHECK FOR HORIZONTAL VELOCITY VECTOR
    #     SECOND IF: CEHCK FOR VERTICAL VELOCITY VECTOR
    #     ELSE: USE VECTOR MATH TO CALCULATE VELOCITY VECTOR
    #     """
    #     if wavefront_vector[0] == -velocity1[0] and wavefront_vector[1] == -velocity1[1]:
    #         v_guess = velocity1
    #     elif wavefront_vector[0] == velocity2[0] and wavefront_vector[1] == velocity2[1]:
    #         v_guess = velocity2
    #     else:
    #         rotation_matrix = np.array([[0, 1], [-1, 0]])
    #         guess_unit_vector = np.dot(rotation_matrix, wavefront_unit_vector)
    #         velocity1_mag = np.linalg.norm(velocity1)
    #         velocity2_mag = np.linalg.norm(velocity2)

    #         v_mag_guess1 = np.dot(velocity1, guess_unit_vector)
    #         v_mag_guess2 = np.dot(velocity2, guess_unit_vector)
            
    #         #in case they are different, usually they are very similar
    #         v_mag_guess = (v_mag_guess1 + v_mag_guess2) / 2
    #         v_guess = v_mag_guess * guess_unit_vector
    #         print("angle", np.degrees(np.arccos(guess_unit_vector[0])), np.degrees(np.arcsin(guess_unit_vector[1])))
    #     return v_guess
    
    # def velocityMap(self, ref_ele_num, minVelocity, maxVelocity, peak_num, num_vector):
    #     """
    #     This function calculates wave velocities for each pair of electrodes
    #     Input:
    #     - ref_ele_num = reference electrode in the pair
    #     - maxVelocity = maximum allowable velocity
    #     - peak_num = which peak in intracardiac electrogram to calculate velocity
    #     - num_vector = top num_vector vectors with the highest cross-correlation are plotted
        
    #     Output: 
    #     - velocity_vectors = list of velocity vectors
    #     - origin = location of reference electrode
        
    #     Can modify function to include z-coordinate later
    #     """
    #     e1 = self.signals[ref_ele_num]
    #     origin = self.coord[ref_ele_num]
    #     ele_nums = np.arange(0, 16, 1)
    #     ele_nums = np.delete(ele_nums, ref_ele_num)
    #     velocity_vectors = []
    #     max_RXY_arr = []
    #     ind_shifts = self.findEGMPeak(e1)
    #     ind_shift = ind_shifts[peak_num]
    #     for ele_num in ele_nums:
    #         d_vector = self.coord[ele_num] - self.coord[ref_ele_num]
    #         d_mag = np.linalg.norm(d_vector)
    #         if d_mag <= 4 * np.sqrt(2):
    #             # FOR SPECIFIC PEAK
    #             ind_shifts = self.findEGMPeak(e1)
    #             ind_shift = ind_shifts[peak_num]
                
    #             velocity_vector, max_RXY = self.velocity(ref_ele_num, ele_num, minVelocity, maxVelocity, ind_shift)
    #             """"FOR CONVENIENCE SO I DON'T HAVE TO CONSTANTLY CHANGE THE FUNCTION INPUT"""
    #             corr_threshold = num_vector
                
    #             if max_RXY > corr_threshold:
    #                 velocity_vectors.append(velocity_vector)
    #                 print(ref_ele_num, ele_num, velocity_vector, max_RXY)
    #                 max_RXY_arr.append(max_RXY)
    #             else:
    #                 velocity_vectors.append(np.zeros(2))
    #                 max_RXY_arr.append(0)
    #     return velocity_vectors, origin, max_RXY_arr
    
    # def fractionalShift(self, max_index, best_indexDelay):
    #     """
    #     Find the fractional index shift that maximizes the cross-correlation
    #     between two signals. This approach uses linear interpolation.
    #     """            
    #     # Refine the result to find fractional lag using spline interpolation
    #     if max_index > 0 and max_index < len(RXY) - 1:
    #         # Create a spline interpolant over the region around the peak
    #         spline = spi.interp1d(index_delays[max_index-1:max_index+2], RXY[max_index-1:max_index+2], kind='linear', fill_value="extrapolate")
            
    #         # Find the fractional lag by evaluating the spline at higher resolution
    #         fine_lags = np.linspace(index_delays[max_index-1], index_delays[max_index+1], 1000)
    #         fine_corrs = spline(fine_lags)
            
    #         # Find the lag that gives the maximum correlation from the interpolated function
    #         fractional_indexDelay = fine_lags[np.argmax(fine_corrs)]
    #     else:
    #         # If interpolation is not possible, use the integer lag
    #         fractional_indexDelay = best_indexDelay
    
    #     # Convert the lag to fractional seconds using the sample frequency
    #     fractional_indexDelay = fractional_indexDelay
    #     return fractional_indexDelay
    
    # def timeDelay(self, ele_num1, ele_num2, vel, ang):
    #     """
    #     This function calculates time delay at a certain time from distance, velocity, and angle
    #     Input:
    #     - t_ind = index in time array
    #     - ele_num1 = number for first electrode (0 - 15)
    #     - ele_num2 = number for second electrode (0 - 15)
    #     - vel = magnitude of velocity in m/s
    #     - ang = angle in radians
        
    #     Output: 
    #     - t_delay = time delay in seconds
        
    #     Can modify function to include z-coordinate later
    #     """
    #     d = self.electrodeDistance(ele_num1, ele_num2)
    #     v_unit = np.array([np.cos(ang), np.sin(ang)])
    #     t_delay = np.dot(d, v_unit) / vel
    #     return t_delay
    
    # def shiftSignal(self, N, ele_num2, t_delay):
    #     """
    #     This function shifts signal of electrode by a set time delay
    #     Signals are shifted by padding one end with the signal average
    #     Features at the end of the signal are cut off
    #     Input:
    #     - t_delay = time delay in seconds
    #     - N = length of first signal
    #     - ele_num2 = electrode number of second electrode
        
    #     Output: 
    #     - shifted_e2 = shifted signal array of electrode e2
    #     """
    #     e2 = self.signals[ele_num2]
    #     index_offset = t_delay * self.sigSampFreq
    #     int_offset = int(round(index_offset))
    #     average_e2 = np.average(e2)
    #     #integer shift with average-padding
    #     if int_offset > 0:
    #         shifted_e2 = np.concatenate((np.full(abs(int_offset), average_e2), e2[:-int_offset]))
    #     elif int_offset < 0:
    #         shifted_e2 = np.concatenate((e2[-int_offset:], np.full(abs(int_offset), average_e2)))
    #     else:
    #         shifted_e2 = e2.copy()
    #     #truncate to original length
    #     shifted_e2 = shifted_e2[:N]
    #     return shifted_e2
    
    # def shiftSignal2(self, N, ele_num2, t_delay):
    #     """
    #     This function shifts signal of electrode by a set time delay
    #     Signals are shifted by padding one end with the signal average
    #     Features at the end of the signal are cut off
    #     Fractional shifts use interpolation
    #     Input:
    #     - t_delay = time delay in seconds
    #     - N = length of first signal
    #     - ele_num2 = electrode number of second electrode
        
    #     Output: 
    #     - shifted_e2 = shifted signal array of electrode e2
    #     """
    #     e2 = self.signals[ele_num2]
    #     index_offset = t_delay * self.sigSampFreq
    #     int_offset = int(np.floor(index_offset))
    #     frac_offset = index_offset - int_offset   # Fractional part
    #     average_e2 = np.average(e2)
    #     #integer shift with average-padding
    #     if int_offset > 0:
    #         shifted_e2 = np.concatenate((np.full(abs(int_offset), average_e2), e2[:-int_offset]))
    #     elif int_offset < 0:
    #         shifted_e2 = np.concatenate((e2[-int_offset:], np.full(abs(int_offset), average_e2)))
    #     else:
    #         shifted_e2 = e2.copy()
    #     """
    #     FRACTIONAL SHIFT CHANGES THE SIGNAL AMPLITUDE
    #     """    
    #     # Fractional shift using interpolation
    #     if frac_offset != 0:
    #         shifted_e2 = (1 - frac_offset) * shifted_e2 + frac_offset * np.roll(shifted_e2, -1)
        
    #     # Truncate to original length
    #     shifted_e2 = shifted_e2[:N]
    #     return shifted_e2
    
    # def simpleShiftSignal(self, ele_num2, delay_range):
    #     """
    #     FRACTIONAL SHIFT CHANGES THE SIGNAL AMPLITUDE
    #     """
    #     N = len(self.signals[ele_num2])
    #     original_ind = np.arange(N)
        
    #     # Convert time delays to index offsets
    #     index_offsets = np.array(delay_range) * self.sigSampFreq
    
    #     # Generate shifted indices for each delay
    #     shifted_signals = np.array([
    #         np.interp(original_ind - index_offset, original_ind, self.signals[ele_num2], left=0, right=0)
    #         for index_offset in index_offsets
    #     ])
    
    #     return shifted_signals
    
    # def simpleCorrelate(self, ele_num1, ele_num2, t_delay):
    #     e1 = self.signals[ele_num1]
    #     #shifted_e2 = self.shiftSignal2(N, ele_num2, t_delay)
    #     shifted_e2 = self.simpleShiftSignal(ele_num2, t_delay)
    #     e1 = np.asarray(e1).flatten()
    #     shifted_e2 = np.asarray(shifted_e2).flatten()
    
    #     #perform the correlation calculation directly using NumPy's outer product
    #     RXY = np.outer(e1, shifted_e2.conj())
    #     return np.sum(RXY)
    
    # def crossCorrelationMatrix(self, ele_num1, ele_num2, num_vel, num_ang):
    #     velocities = np.linspace(self.minVelocity, self.maxVelocity, num_vel)
    #     angles = np.linspace(self.minAngle, self.maxAngle, num_ang)
    #     tasks = [(v, theta) for v in velocities for theta in angles]
    #     results = Parallel(n_jobs = self.cpu_num)(delayed(self.simpleCorrelate)(ele_num1, ele_num2, v, theta) for v, theta in tasks)
    #     RXY_matrix = np.array(results).reshape(len(velocities), len(angles)) #reshape the results into a 2D matrix
    #     X, Y = np.meshgrid(angles, velocities) #meshgrid of angles and velocities
    #     VX, VY = Y * np.cos(angles), Y * np.sin(angles)
    #     return X, Y, VX, VY, RXY_matrix
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

    