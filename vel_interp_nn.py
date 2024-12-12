'''Author: Hiroki Kozuki'''
#%%
'''Imports'''

import time
import numpy as np
import matplotlib.pyplot as plt
from modules import *
from numba import jit

import torch
import torch.nn as nn
import torch.optim as optim

#%%
'''Load data'''

data = "ElectrogramData.xlsx"

start = time.time()

L = loadData(data, n = 600) # Load electrogram data
print(L.coordinates().shape) # (16, 2) - x,y positions of electrodes 
print(L.ele_signals().shape) # (2201, 16) - 2201 voltage data at 16 electrodes.
print(L.ele_signals().iloc[0].shape) # (16, ) 
print(L.ele_signals().iloc[0])
print(L.time_data().shape)   # (2201, ) - 2201 time points. 

end = time.time()
print("time take (new)", end - start)

#%%
'''ML method: Learning the velocity map and voltage map for c(x,y)'''

# Define a neural network for c(x, y)
class VelocityNet(nn.Module):
    def __init__(self, grid_size):
        super(VelocityNet, self).__init__()
        self.velocity = nn.Parameter(torch.ones(grid_size, grid_size))  # Learnable parameter

    def forward(self):
        return self.velocity

# Parameters
grid_size = 600
dx = 12.0 / grid_size
dy = 12.0 / grid_size
timesteps = 100

# Observed voltage data (e.g., 16 electrode positions)
electrode_positions = [(i, j) for i in range(4) for j in range(4)]
observed_voltages = np.random.rand(16)  # Replace with real data

# Initialize the velocity network
velocity_net = VelocityNet(grid_size)
optimizer = optim.Adam(velocity_net.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Simulate the wave equation (discretized)
def wave_equation_step(u, velocity, dx, dy):
    u_new = u.clone()
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            if (i, j) not in electrode_positions:
                c_x_plus = (velocity[i+1, j] + velocity[i, j]) / 2
                c_x_minus = (velocity[i-1, j] + velocity[i, j]) / 2
                c_y_plus = (velocity[i, j+1] + velocity[i, j]) / 2
                c_y_minus = (velocity[i, j-1] + velocity[i, j]) / 2
            
                u_new[i, j] = (
                    c_x_plus * u[i+1, j] + c_x_minus * u[i-1, j]
                    + c_y_plus * u[i, j+1] + c_y_minus * u[i, j-1]
                ) / (c_x_plus + c_x_minus + c_y_plus + c_y_minus)
    return u_new

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    velocity = velocity_net()  # Learnable velocity field

    # Initialize wave equation solution
    u = torch.zeros((grid_size, grid_size))  # Replace with initial condition
    for t in range(timesteps):
        u = wave_equation_step(u, velocity, dx, dy)

    # Compute loss at electrode positions
    predicted_voltages = torch.tensor([u[x, y] for x, y in electrode_positions])
    loss = criterion(predicted_voltages, torch.tensor(observed_voltages))

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Final velocity field
final_velocity = velocity_net().detach().numpy()
print(final_velocity)


# %%
