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
  

