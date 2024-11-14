#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:14:02 2024

@author: candace_chung
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.interpolate as spi
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

#%%

#class

class loadData:
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
            posTime_arr = data["pT"]
            sigTime_arr = data["sT"]
            
            #transpose into row-major format
            x_arr = np.transpose(x_arr)
            y_arr = np.transpose(y_arr)
            z_arr = np.transpose(z_arr)
            signals = np.transpose(signals)
            
            return x_arr, y_arr, z_arr, signals, posTime_arr, sigTime_arr
        
        self.x_pos, self.y_pos, self.z_pos, self.signals, self.pos_time, self.sig_time = electrodeData(fileName)
    
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