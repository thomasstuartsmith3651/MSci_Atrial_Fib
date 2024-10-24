#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:02:20 2024

@author: candace_chung
"""

import time

from modules import *

data = "ElectrogramData.xlsx"

#%%
start = time.time()

L = loadData(data, n = 600)

df = L.data_frame()

end = time.time()

print("time take (new)", end - start)

#%%

print(df)

#%%

start = time.time()

L = loadData(data, n = 600)

df = L.data_frame()

A = Animate(data, n = 600, dataframe = df, ind = 218, ele_radius = 0.5, animate = True)

A.run()

end = time.time()

print("time taken", end - start)