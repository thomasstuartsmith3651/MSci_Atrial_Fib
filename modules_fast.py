#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:53:03 2024

@author: candace_chung
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:14:05 2024

@author: candace_chung
"""

import numpy as np
import pandas as pd
import scipy.interpolate as spi
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

#%%

#class

class loadData:
    def __init__(self, data, n): #input name of excel file
        def electrodeData(self, data):
            """
            This function loads the excel file and extracts the following data:
            - x coordinates of electrode positions is a 1D dataframe
            - y coordinates of electrode positions is a 1D dataframe
            - paired (x, y) coordinates is a 2D array
            - electrode signals is a 2D dataframe with columns = electrode number and row = measurement at certain time
            - list of timestamps calculated from the sampling frequency is a 1D array
            """
            positions = pd.read_excel(data, sheet_name = 1)
            x = positions.iloc[0]
            y = positions.iloc[1]
            coord = positions.transpose().to_numpy()
            
            signals = pd.read_excel(data, sheet_name = 0)
            
            t_interval = 1/2034.5 #sampling frequency is 2034.5 Hz
            time = np.arange(0, signals.shape[0] * t_interval, t_interval)
            return x, y, coord, signals, time

        self.n = n
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
        This function calls the data frame of electrode signals outside of the class
        
        Columns = electrode number 
        Rows = measurement at certain time
        """
        return self.signals
    
    def time_data(self):
        """
        This function calls the 1D array of timestamps outside of the class
        """
        return self.time
    
    def data_frame(self):
        def interpolatedDataFrame(self, x_df, y_df, coord_pairs, signal_df, time_arr):
            """
            This function generates the dataframe with the interpolated data
                
            The dataframe is composed of a series of concatenated sub-dataframes:
            - Dataframe with key "X": X coordinates of meshgrid grid points
            - Dataframe with key "Y": Y coordinates of meshgrid grid points
            - Dataframes with timestamps as keys: interpolated signal at each grid point
                
            The dimensions of the sub-dataframes correspond to the dimensions of the meshgrid
            """
            def interpolateSignal(signal_array, coordinate_pairs, X_array, Y_array, ind): #function to interpolate the signals
                signal = signal_array[ind, :]
                Zarr = spi.griddata(coordinate_pairs, signal, (X_array, Y_array), method='cubic') #cubic spline interpolation for electrical signals
                Zdf = pd.DataFrame(Zarr) #turn the interpolated signal array into dataframe
                return Zdf
            
            N = self.n * 1j
            Xarr, Yarr = np.mgrid[x_df.min():x_df.max():N, y_df.min():y_df.max():N] #create spatial meshgrid for interpolation
            Xdf, Ydf = pd.DataFrame(Xarr), pd.DataFrame(Yarr) #turn the arrays into dataframes
            
            signal_arr = signal_df.to_numpy() #convert the signal array into a dataframe for faster processing
            
            df_list = [Xdf, Ydf] #create list of dataframes for concatenation
            df_keys = ["X", "Y"] #create list of keys to call concatenated dataframes

            # Use parallelism to speed up interpolation across time steps
            results = Parallel(n_jobs = -1)(delayed(interpolateSignal)(signal_arr, coord_pairs, Xarr, Yarr, i) for i in range(len(time_arr))) #get the interpolated signal dataframes over all time

            df_list.extend(results) #add the signal dataframes to the dataframe list for concatenation
            df_keys.extend(time_arr) #add the times to the list of keys for concatenation

            df = pd.concat(df_list, axis = 0, keys = df_keys) #concatenate all the X, Y, and Z dataframes at different times and label them by the time
            return df
        
        data_frame = interpolatedDataFrame(self, self.x_pos, self.y_pos, self.coord, self.signals, self.time)
        return data_frame

class Animate(loadData):
    def __init__(self, data, n, dataframe, ind, ele_radius, animate = False):
        loadData.__init__(self, data, n)
        
        self.df = dataframe
        self.ind = ind
        self.ele_radius = ele_radius
        self.animate = animate
        
        self.X, self.Y = self.df.loc["X"].to_numpy(), self.df.loc["Y"].to_numpy()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_zlim(self.signals.min().min(), self.signals.max().max())

        self.electrodes = [] #create electrode patches
        for cX, cY in self.coord:
            ele = RegularPolygon((cX, cY), numVertices = 5, radius = self.ele_radius, color = 'blue')
            self.ax.add_patch(ele)
            art3d.pathpatch_2d_to_3d(ele, z = 0, zdir = 'z')
            self.electrodes.append(ele)

        self.surface_plot = None #initialise surface plot variable
        
        self.frame_data = self.precomputeFrames() #precompute frame data in parallel and assign it to self.frame_data

    def computeFrameData(self, ind):
        """ 
        Compute data for a specific frame
        """
        signal = self.signals.iloc[ind, :].to_numpy()
        Z = self.df.loc[self.time[ind]].to_numpy()
        return signal, Z

    def precomputeFrames(self):
        """ 
        Precompute all frame data in parallel using joblib to speed up animation process
        """
        frame_data = Parallel(n_jobs = -1)(delayed(self.computeFrameData)(i) for i in range(len(self.time)))
        return frame_data

    def plot_ith_frame(self, ind):
        if self.surface_plot is not None: #remove the previous surface plot if it exists
            self.surface_plot.remove()

        signal, Z = self.frame_data[ind] #get precomputed data

        self.surface_plot = self.ax.plot_surface(self.X, self.Y, Z, alpha=0.2, fc='w', ec='k', shade=False) #plot new signal

        for i, ele in enumerate(self.electrodes): #update vertical position of all electrodes
            ele._segment3d = [(x, y, signal[i]) for x, y, _ in ele._segment3d]

        self.ax.set_title(f't = {self.time[ind]:.3e} s') #graph title is the timestamp of the data
        self.ax.set_xlabel('x-position (mm)')
        self.ax.set_ylabel('y-position (mm)')
        self.ax.set_zlabel('voltage (mV)')

    def run(self):
        if self.animate: #run animation
            anim = FuncAnimation(self.fig, self.plot_ith_frame, frames = len(self.frame_data), interval = 1, repeat = False)
            #plt.show()
            f = r"/Users/candace_chung/Desktop/animation.gif" 
            writervideo = PillowWriter(fps=60) 
            anim.save(f, writer=writervideo)
        else: #plot single frame at specified index
            self.plot_ith_frame(self.ind)
            plt.show()
