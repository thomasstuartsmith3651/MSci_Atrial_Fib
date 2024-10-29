#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:02:20 2024

@author: candace_chung
"""

import time
import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from modules import *

data = "ElectrogramData.xlsx"

#%%
start = time.time()

L = loadData(data, n = 600)

df = L.data_frame()

end = time.time()

print("time take (new)", end - start)

#%%

# Loading data (assuming loadData is a defined function)
L = loadData(data, n=600)

# Getting time data and electrode signals
t = L.time_data()  # Assumes this returns a 1D array or Series for time values
Z = L.ele_signals().transpose().iloc[0].to_numpy()  # Extracts the first row as a 1D array

plt.plot(t, Z)
plt.title("Signal vs Time Graph of Electrode at (0,0)")
plt.xlabel("Time")
plt.ylabel("Electrode Signal")
plt.show()
#%%

start = time.time()

L = loadData(data, n = 600)

df = L.data_frame()

A = Animate(data, n = 600, dataframe = df, ind = 218, ele_radius = 0.5, animate = True)

A.run()

end = time.time()

print("time taken", end - start)