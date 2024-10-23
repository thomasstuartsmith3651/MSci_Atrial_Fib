#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:02:20 2024

@author: candace_chung
"""

from modules import *

data = "ElectrogramData.xlsx"

#%%

L = loadData(data, n = 600)

df = L.data_frame()
print(df)

#%%

A = Animate(data, n = 600, ind = 218, ele_radius = 0.5, animate = True)

A.run()