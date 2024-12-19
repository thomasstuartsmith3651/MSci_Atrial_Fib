#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 23:19:18 2024

@author: candace_chung
"""

import pandas as pd
import numpy as np

# Load the Excel file
file_path = 'ElectrogramData_MOVING.xlsx'  # Replace with your file path
excel_data = pd.ExcelFile(file_path)

# Load the sheets into dataframes
electrograms = excel_data.parse('Electrograms')
x_positions = excel_data.parse('X')
y_positions = excel_data.parse('Y')

# Define sampling frequencies
signal_sampling_freq = 2034.5  # Hz
position_sampling_freq = 101.725  # Hz

# Calculate the time duration of the signal in seconds
time_duration = electrograms.shape[0] / signal_sampling_freq

# Calculate the number of rows needed for X and Y positions based on their sampling frequency
required_position_rows = int(time_duration * position_sampling_freq)

# Expand X and Y to match the required rows with random variation of ±0.5
expanded_x = np.tile(x_positions.values, (required_position_rows, 1)) + np.random.uniform(-0.5, 0.5, (required_position_rows, x_positions.shape[1]))
expanded_y = np.tile(y_positions.values, (required_position_rows, 1)) + np.random.uniform(-0.5, 0.5, (required_position_rows, y_positions.shape[1]))

# Convert back to DataFrame
expanded_x_df = pd.DataFrame(expanded_x, columns=x_positions.columns)
expanded_y_df = pd.DataFrame(expanded_y, columns=y_positions.columns)

# Save the updated data to a new Excel file
output_file_path = 'ElectrogramData_MOVING_Updated.xlsx'  # Specify the desired output path
with pd.ExcelWriter(output_file_path) as writer:
    expanded_x_df.to_excel(writer, index=False, sheet_name='X')
    expanded_y_df.to_excel(writer, index=False, sheet_name='Y')
    electrograms.to_excel(writer, index=False, sheet_name='Electrograms')

print(f"Updated file saved as {output_file_path}")
