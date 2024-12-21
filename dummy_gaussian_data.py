import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

#%%

# single wave

def gaussian_pulse_amplitude_SINGLE(x, y, t, centers, direction, speed, width):
    """
    Calculate the amplitude of multiple Gaussian pulses in 2D at given (x, y) coordinates.

    Parameters:
        x (float or np.ndarray): x-coordinate(s) where amplitude is calculated.
        y (float or np.ndarray): y-coordinate(s) where amplitude is calculated.
        t (float): Time at which the amplitude is calculated.
        centers (list of tuples): List of initial centers of the Gaussian pulses as [(x0, y0), ...].
        direction (tuple): Unit vector specifying the propagation direction as (dx, dy).
        speed (float): Speed of pulse propagation.
        width (float): Width of the Gaussian pulses.

    Returns:
        float or np.ndarray: Amplitude of the Gaussian pulses at the specified coordinates.
    """
    # Normalize the direction vector
    dx, dy = direction
    direction_norm = np.sqrt(dx**2 + dy**2)
    dx /= direction_norm
    dy /= direction_norm

    # Initialize the total amplitude
    total_amplitude = np.zeros_like(x, dtype=float)

    for x0, y0 in centers:
        # Calculate the current center of each pulse
        x_center = x0 + dx * speed * t
        y_center = y0 + dy * speed * t

        # Compute the squared distance from the pulse center
        distance_squared = (x - x_center)**2 + (y - y_center)**2

        # Add the Gaussian amplitude for this center
        total_amplitude += np.exp(-distance_squared / (2 * width**2))

    return total_amplitude

#train of waves

def gaussian_pulse_amplitude_TRAIN(x, y, t, centers, direction, speed, width, pulse_interval):
    """
    Calculate the amplitude of multiple Gaussian plane waves in 2D at a given (x, y) coordinate.

    Parameters:
        x (float or np.ndarray): x-coordinate(s) where amplitude is calculated.
        y (float or np.ndarray): y-coordinate(s) where amplitude is calculated.
        t (float): Time at which the amplitude is calculated.
        centers (list of tuples): List of initial centers of the Gaussian pulses [(x0, y0), ...].
        direction (tuple): Unit vector specifying the propagation direction (dx, dy).
        speed (float): Speed of pulse propagation.
        width (float): Width of the Gaussian pulses.
        pulse_interval (float): Time interval between consecutive pulses.

    Returns:
        float or np.ndarray: Amplitude of the Gaussian pulses at the specified coordinates.
    """
    # Normalize the direction vector
    dx, dy = direction
    direction_norm = np.sqrt(dx**2 + dy**2)
    dx /= direction_norm
    dy /= direction_norm

    # Initialize the total amplitude
    total_amplitude = np.zeros_like(x, dtype=float)

    # Compute the contribution of each active Gaussian pulse
    num_active_pulses = int(t // pulse_interval) + 1  # Number of pulses that have been generated up to time t
    for pulse_idx in range(num_active_pulses):
        pulse_time = pulse_idx * pulse_interval  # Time when this pulse was generated
        time_since_pulse = t - pulse_time  # Elapsed time for this pulse

        for x0, y0 in centers:
            # Calculate the current center of this pulse
            x_center = x0 + dx * speed * time_since_pulse
            y_center = y0 + dy * speed * time_since_pulse

            # Compute the squared distance from the pulse center
            distance_squared = (x - x_center)**2 + (y - y_center)**2

            # Add the Gaussian amplitude for this pulse
            total_amplitude += np.exp(-distance_squared / (2 * width**2))

    return total_amplitude

#%%

#SINGLE GAUSSIAN WAVE PULSE moving electrode positions

def generate_data_single_wave_moving_electrode(angle, propagation_speed, max_variation): # Propagation direction (degrees from the horizontal) and Speed of the pulse (mm/s)
    # Define pulse parameters 
    angle = np.radians(angle)
    propagation_direction = (np.cos(angle), np.sin(angle))  # Direction of propagation
    pulse_width = 0.5  # Width of the Gaussian pulses
    
    # Define specific coordinates to store data (mm)
    specific_coords = [
        (0, 0), (0, 4), (0, 8), (0, 12),
        (4, 0), (4, 4), (4, 8), (4, 12),
        (8, 0), (8, 4), (8, 8), (8, 12),
        (12, 0), (12, 4), (12, 8), (12, 12)
    ]
    
    # Generate Gaussian centers along a line perpendicular to the direction of propagation
    num_centers = 50  # Number of Gaussian centers
    screen_range = 12  # Range for the screen (x and y axis span)
    initial_centers = [
        (x * np.cos(angle + np.pi/2) - 2, x * np.sin(angle + np.pi/2) - 2)  # Place Gaussians in a line perpendicular to propagation
        for x in np.linspace(-screen_range - 3, screen_range + 3, num_centers)
    ]
    
    # Sampling frequencies
    signal_sampling_frequency = 2034.5  # Hz
    position_sampling_frequency = 101.725  # Hz
    position_update_interval = int(signal_sampling_frequency / position_sampling_frequency)  # Update positions every 20 time steps
    
    # Store the amplitude values for the specific coordinates
    stored_data = []
    random_coords_x = []
    random_coords_y = []
    
    # Time for which we want to store the data
    total_time = 1  # Total simulation time in seconds
    total_signal_samples = int(total_time * signal_sampling_frequency)
    
    # Initialize randomized coordinates
    randomized_coords = specific_coords
    
    for t in range(total_signal_samples):
        time = t / signal_sampling_frequency  # Time for this signal sample
    
        # Update randomized positions at the lower position sampling frequency
        if t % position_update_interval == 0:
            randomized_coords = [
                (
                    x + np.random.uniform(-max_variation, max_variation),  # Random variation for x-coordinate (scaled by 0.7)
                    y + np.random.uniform(-max_variation, max_variation)   # Random variation for y-coordinate (scaled by 0.3)
                )
                for x, y in specific_coords
            ]
            random_coords_x.append([coord[0] for coord in randomized_coords])
            random_coords_y.append([coord[1] for coord in randomized_coords])
    
        amplitude = gaussian_pulse_amplitude_SINGLE(
            np.array([coord[0] for coord in randomized_coords]),
            np.array([coord[1] for coord in randomized_coords]),
            time,
            initial_centers,
            propagation_direction,
            propagation_speed,
            pulse_width
        )
        # Store the data at this time step
        stored_data.append(amplitude)
    
    # Convert stored data to a data frame for easy manipulation
    stored_data = np.array(stored_data)
    data_df = pd.DataFrame(stored_data)
    
    # Convert randomized coordinates into separate DataFrames for x and y
    random_coords_x_df = pd.DataFrame(random_coords_x, columns=[f"Electrode_{i}_x" for i in range(len(specific_coords))])
    random_coords_y_df = pd.DataFrame(random_coords_y, columns=[f"Electrode_{i}_y" for i in range(len(specific_coords))])

    output_path = '/Users/candace_chung/Desktop/Candace Chung Files/ICL/Academics/Year 4/MSci Project/code/MSci_Atrial_Fib/test_gaussian_data_BACKUP.xlsx'
    
    # Save the DataFrames to the same Excel file with different sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        data_df.to_excel(writer, sheet_name="Signals", index=False)
        random_coords_x_df.to_excel(writer, sheet_name="X", index=False)
        random_coords_y_df.to_excel(writer, sheet_name="Y", index=False)
    return output_path

#%%

#TRAIN OF GAUSSIAN WAVES WITH MOVING ELECTRODE POSITIONS

def generate_data_train_wave_moving_electrode(angle, propagation_speed, pulse_frequency, max_variation):
    """
    Generate data for a train of Gaussian pulses with moving electrodes.
    """
    # Define pulse and propagation parameters
    angle = np.radians(angle)
    propagation_direction = (np.cos(angle), np.sin(angle))
    pulse_width = 0.5
    pulse_interval = 1 / pulse_frequency

    # Define specific coordinates to store data (mm)
    specific_coords = [
        (0, 0), (0, 4), (0, 8), (0, 12),
        (4, 0), (4, 4), (4, 8), (4, 12),
        (8, 0), (8, 4), (8, 8), (8, 12),
        (12, 0), (12, 4), (12, 8), (12, 12)
    ]

    # Generate Gaussian centers along a line perpendicular to the direction of propagation
    num_centers = 50
    screen_range = 12
    initial_centers = [
        (x * np.cos(angle + np.pi/2) - 2, x * np.sin(angle + np.pi/2) - 2)
        for x in np.linspace(-screen_range - 3, screen_range + 3, num_centers)
    ]

    # Sampling frequencies
    signal_sampling_frequency = 2034.5
    position_sampling_frequency = 101.725
    position_update_interval = int(signal_sampling_frequency / position_sampling_frequency)

    # Store the amplitude values for the specific coordinates
    stored_data = []
    random_coords_x = []
    random_coords_y = []

    # Time for which we want to store the data
    total_time = 1
    total_signal_samples = int(total_time * signal_sampling_frequency)

    # Initialize randomized coordinates
    randomized_coords = specific_coords

    for t in range(total_signal_samples):
        time = t / signal_sampling_frequency

        # Update randomized positions at the lower position sampling frequency
        if t % position_update_interval == 0:
            randomized_coords = [
                (
                    x + np.random.uniform(-max_variation, max_variation),
                    y + np.random.uniform(-max_variation, max_variation)
                )
                for x, y in specific_coords
            ]
            random_coords_x.append([coord[0] for coord in randomized_coords])
            random_coords_y.append([coord[1] for coord in randomized_coords])

        # Calculate amplitudes at current randomized positions
        amplitude = gaussian_pulse_amplitude_TRAIN(
            np.array([coord[0] for coord in randomized_coords]),
            np.array([coord[1] for coord in randomized_coords]),
            time,
            initial_centers,
            propagation_direction,
            propagation_speed,
            pulse_width,
            pulse_interval
        )
        # Store the data at this time step
        stored_data.append(amplitude)

    # Convert stored data to a data frame for easy manipulation
    stored_data = np.array(stored_data)
    data_df = pd.DataFrame(stored_data)

    # Convert randomized coordinates into separate DataFrames for x and y
    random_coords_x_df = pd.DataFrame(random_coords_x, columns=[f"Electrode_{i}_x" for i in range(len(specific_coords))])
    random_coords_y_df = pd.DataFrame(random_coords_y, columns=[f"Electrode_{i}_y" for i in range(len(specific_coords))])

    # Save the DataFrames to the same Excel file with different sheets
    output_path = '/Users/candace_chung/Desktop/Candace Chung Files/ICL/Academics/Year 4/MSci Project/code/MSci_Atrial_Fib/train_gaussian_data_moving.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        data_df.to_excel(writer, sheet_name="Signals", index=False)
        random_coords_x_df.to_excel(writer, sheet_name="X", index=False)
        random_coords_y_df.to_excel(writer, sheet_name="Y", index=False)

    return output_path

#%%

# def animate_gaussian_wave_train(angle, propagation_speed, pulse_frequency):
#     # Define pulse parameters
#     angle = np.radians(angle)  # Propagation direction (degrees from the horizontal)
#     propagation_direction = (np.cos(angle), np.sin(angle))  # Direction of propagation
#     pulse_width = 0.5  # Width of the Gaussian pulses
#     pulse_interval = 1 / pulse_frequency  # Time interval between pulses

#     # Generate Gaussian centers along a line perpendicular to the direction of propagation
#     num_centers = 50  # Number of Gaussian centers
#     screen_range = 12  # Range for the screen (x and y axis span)
#     initial_centers = [
#         (x * np.cos(angle + np.pi / 2) - 2, x * np.sin(angle + np.pi / 2) - 2)  # Centers in a line perpendicular to propagation
#         for x in np.linspace(-screen_range - 3, screen_range + 3, num_centers)
#     ]

#     # Define coordinates to evaluate the amplitude for animation
#     x_coords = np.linspace(0, 12, 200)  # Grid for x-coordinates
#     y_coords = np.linspace(0, 12, 200)  # Grid for y-coordinates
#     X, Y = np.meshgrid(x_coords, y_coords)

#     # Setup the figure and axis for animation
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.set_xlim(0, 12)
#     ax.set_ylim(0, 12)
#     ax.set_xlabel("x (mm)")
#     ax.set_ylabel("y (mm)")
#     ax.set_title("Gaussian Plane Wave Propagation")

#     # Compute the initial amplitude
#     amplitude = gaussian_pulse_amplitude_TRAIN(X, Y, 0, initial_centers, propagation_direction, propagation_speed, pulse_width, pulse_interval)
#     contour = ax.contourf(X, Y, amplitude, levels=50, cmap="viridis")
#     cbar = plt.colorbar(contour, ax=ax, label="Amplitude")

#     # Time settings
#     sampling_frequency = 2034.5  # Sampling frequency (Hz)
#     time_increment = 1 / sampling_frequency  # Time step size
#     time_steps = int(1 / time_increment)  # Simulate 1 second of wave propagation

#     # Update function for animation
#     def update(frame):
#         time = frame * time_increment  # Current time for animation
#         amplitude = gaussian_pulse_amplitude_TRAIN(X, Y, time, initial_centers, propagation_direction, propagation_speed, pulse_width, pulse_interval)
#         ax.clear()  # Clear previous plot
#         ax.set_xlim(0, 12)
#         ax.set_ylim(0, 12)
#         ax.set_xlabel("x (mm)")
#         ax.set_ylabel("y (mm)")
#         ax.set_title("Gaussian Plane Wave Propagation")
#         contour = ax.contourf(X, Y, amplitude, levels=50, cmap="viridis")
#         return contour.collections

#     # Create the animation
#     anim = FuncAnimation(fig, update, frames=time_steps, interval=1, blit=False)

#     # Prevent the animation object from being garbage collected
#     return anim

# if __name__ == "__main__":
#     # Define pulse parameters
#     angle = 90  # Propagation direction (degrees from the horizontal)
#     propagation_speed = 400  # Speed of the pulse (mm/s)
#     pulse_frequency = 50  # Frequency of the Gaussian pulses (Hz)

#     # Run the animation
#     anim = animate_gaussian_wave_train(angle, propagation_speed, pulse_frequency)  # Store the animation object
#     plt.show()  # Ensure the animation persists until shown

#%%


# if __name__ == "__main__":
#     # Define pulse parameters
#     angle = np.radians(80)  # Propagation direction (degrees from the horizontal)
#     propagation_direction = (np.cos(angle), np.sin(angle))  # Direction of propagation
#     propagation_speed = 500  # Speed of the pulse (mm/s)
#     pulse_width = 0.5  # Width of the Gaussian pulses

#     # Define specific coordinates to store data (mm)
#     specific_coords = [
#         (0, 0), (0, 4), (0, 8), (0, 12),
#         (4, 0), (4, 4), (4, 8), (4, 12),
#         (8, 0), (8, 4), (8, 8), (8, 12),
#         (12, 0), (12, 4), (12, 8), (12, 12)
#     ]

#     # Generate Gaussian centers along a line perpendicular to the direction of propagation
#     num_centers = 50  # Number of Gaussian centers
#     screen_range = 12  # Range for the screen (x and y axis span)
#     initial_centers = [
#         (x * np.cos(angle + np.pi/2)-2, x * np.sin(angle + np.pi/2)-2)  # Place Gaussians in a line perpendicular to propagation
#         for x in np.linspace(-screen_range-3, screen_range+3, num_centers)
#     ]

#     # Store the amplitude values for the specific coordinates
#     stored_data = []
    
#     # Time for which we want to store the data
#     time_steps = 2000  # Number of time steps to evaluate
#     for t in range(time_steps):
#         time = t * 1/2034.5  # Adjust time increment (SAMPLING FREQUENCY IS 2034.5 Hz)
#         amplitude = gaussian_pulse_amplitude_SINGLE(np.array([coord[0] for coord in specific_coords]),
#                                              np.array([coord[1] for coord in specific_coords]),
#                                              time,
#                                              initial_centers,
#                                              propagation_direction,
#                                              propagation_speed,
#                                              pulse_width)
#         # Store the data at this time step
#         stored_data.append(amplitude)

#     # Convert stored data to a data frame for easy manipulation
#     stored_data = np.array(stored_data)
#     data_df = pd.DataFrame(stored_data)

#     # # Define coordinates to evaluate the amplitude
#     # x_coords = np.linspace(0, screen_range, 100)
#     # y_coords = np.linspace(0, screen_range, 100)
#     # X, Y = np.meshgrid(x_coords, y_coords)

#     # # Setup the figure and axis for animation
#     # fig, ax = plt.subplots(figsize=(8, 6))
#     # amplitude = gaussian_pulse_amplitude_SINGLE(X, Y, 0, initial_centers, propagation_direction, propagation_speed, pulse_width)
#     # contour = ax.contourf(X, Y, amplitude, levels=50, cmap="viridis")
#     # cbar = plt.colorbar(contour, ax=ax, label="Amplitude")
#     # ax.set_xlabel("x")
#     # ax.set_ylabel("y")
#     # ax.set_title("Gaussian Plane Wave Amplitude")
    
#     # # Update function for animation
#     # def update(frame):
#     #     time = frame * 1/2034.5  # Adjust speed of propagation
#     #     amplitude = gaussian_pulse_amplitude(X, Y, time, initial_centers, propagation_direction, propagation_speed, pulse_width)
#     #     for coll in ax.collections:
#     #         coll.remove()  # Clear previous contours
#     #     ax.contourf(X, Y, amplitude, levels=50, cmap="viridis")

#     # # Create animation
#     # frames = time_steps  # Number of frames
#     # anim = FuncAnimation(fig, update, frames=frames, interval=1)
#     # plt.plot(x_coords, x_coords * np.tan(angle), label = "propagation direction")
#     # plt.xlim(0, max(x_coords))
#     # plt.ylim(0, max(x_coords))
#     # plt.legend()
#     # # Show the animation
#     # plt.show()


# if __name__ == "__main__":
#     # Define pulse parameters
#     angle = np.radians(90)  # Propagation direction (degrees from the horizontal)
#     propagation_direction = (np.cos(angle), np.sin(angle))  # Direction of propagation
#     propagation_speed = 400  # Speed of the pulse (mm/s)
#     pulse_width = 0.5  # Width of the Gaussian pulses
#     pulse_frequency = 50  # Frequency of the Gaussian pulses (Hz)
#     pulse_interval = 1 / pulse_frequency  # Time interval between pulses

#     # Generate Gaussian centers along a line perpendicular to the direction of propagation
#     num_centers = 50  # Number of Gaussian centers
#     screen_range = 12  # Range for the screen (x and y axis span)
#     initial_centers = [
#         (x * np.cos(angle + np.pi / 2)-2, x * np.sin(angle + np.pi / 2)-2)  # Centers in a line perpendicular to propagation
#         for x in np.linspace(-screen_range-3, screen_range+3, num_centers)
#     ]

#     # Define specific coordinates to store amplitude data
#     specific_coords = [
#         (0, 0), (0, 4), (0, 8), (0, 12),
#         (4, 0), (4, 4), (4, 8), (4, 12),
#         (8, 0), (8, 4), (8, 8), (8, 12),
#         (12, 0), (12, 4), (12, 8), (12, 12)
#     ]

#     # Initialize a list to store amplitude data
#     stored_data = []

#     # Define time steps
#     time_steps = 2000  # Number of time steps
#     sampling_frequency = 2034.5  # Sampling frequency (Hz)
#     time_increment = 1 / sampling_frequency  # Time step size

#     # Evaluate and store amplitudes at each time step
#     for t_step in range(time_steps):
#         time = t_step * time_increment  # Current time
#         x_coords = np.array([coord[0] for coord in specific_coords])
#         y_coords = np.array([coord[1] for coord in specific_coords])
#         amplitudes = gaussian_pulse_amplitude_TRAIN(x_coords, y_coords, time, initial_centers,
#                                               propagation_direction, propagation_speed, pulse_width, pulse_interval)
#         stored_data.append(list(amplitudes))

#     # Convert stored data into a DataFrame
#     column_names = [f"({x}, {y})" for x, y in specific_coords]
#     data_df = pd.DataFrame(stored_data, columns=column_names)

#     # Define coordinates to evaluate the amplitude for animation
#     x_coords = np.linspace(0, screen_range, 100)
#     y_coords = np.linspace(0, screen_range, 100)
#     X, Y = np.meshgrid(x_coords, y_coords)

#     # Setup the figure and axis for animation
#     fig, ax = plt.subplots(figsize=(8, 6))
#     amplitude = gaussian_pulse_amplitude_TRAIN(X, Y, 0, initial_centers, propagation_direction, propagation_speed, pulse_width, pulse_interval)
#     contour = ax.contourf(X, Y, amplitude, levels=50, cmap="viridis")
#     cbar = plt.colorbar(contour, ax=ax, label="Amplitude")
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_title("Gaussian Plane Wave Amplitude")

#     # Update function for animation
#     def update(frame):
#         time = frame * time_increment  # Current time for animation
#         amplitude = gaussian_pulse_amplitude_TRAIN(X, Y, time, initial_centers, propagation_direction, propagation_speed, pulse_width, pulse_interval)
#         for coll in ax.collections:
#             coll.remove()  # Clear previous contours
#         ax.contourf(X, Y, amplitude, levels=50, cmap="viridis")

#     # Create animation
#     anim = FuncAnimation(fig, update, frames=time_steps, interval=1)

#     # Show the animation
#     plt.show()


# # Simulation parameters
# num_points = 7  # Number of points in the line
# distance_between_electrodes = 4 # Distance (mm)
# #angle = np.radians(30)
# distance_between_points = distance_between_electrodes # Distance between each point
# time_steps = 4069  # Number of time steps
# time = np.linspace(0, 2, time_steps)  # Time array
# pulse_center = 0.3  # Center of the Gaussian pulse
# pulse_width = 0.0005  # Width of the Gaussian pulse
# speed = 700  # Propagation speed (mm/s)
# attenuation_factor = 1 # Amplitude attenuation per point

# # Create datasets
# datasets = []
# pulse_centers = []  # To track the pulse center positions in time
# for i in range(4):
#     delay = (i * distance_between_points) / speed  # Time delay for the ith point
#     amplitude = np.exp(-((time - (pulse_center + delay)) ** 2) / (2 * pulse_width**2))
#     amplitude *= attenuation_factor**i  # Apply attenuation
#     datasets.append(amplitude)
#     pulse_centers.append(pulse_center + delay)  # Store the pulse center position in time
    

# # Plotting the datasets
# plt.figure(figsize=(10, 6))
# for i, (dataset, center) in enumerate(zip(datasets, pulse_centers)):
#     plt.plot(time, dataset, label=f"Point {i+1}")
#     # Annotate pulse center on the plot
#     # plt.annotate(
#     #     f"Center: {center:.2f}", 
#     #     xy=(center, np.exp(-0.5 / pulse_width**2) * attenuation_factor**i),  # Approx. peak
#     #     xytext=(center, 1.1 * np.exp(-0.5 / pulse_width**2) * attenuation_factor**i),
#     #     arrowprops=dict(arrowstyle="->", color='black'),
#     #     fontsize=10
#     # )

# # Customize the plot
# plt.title("Gaussian Pulse Propagation with Pulse Centers")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid()
# plt.show()

# # diagonally propagating gaussian pulse plane wave (propagate from top left corner to bottom right corner)

# # electrode_3 = datasets[0]

# # electrode_2 = datasets[1]
# # electrode_7 = datasets[1]

# # electrode_1 = datasets[2]
# # electrode_6 = datasets[2]
# # electrode_11 = datasets[2]

# # electrode_0 = datasets[3]
# # electrode_5 = datasets[3]
# # electrode_10 = datasets[3]
# # electrode_15 = datasets[3]

# # electrode_4 = datasets[4]
# # electrode_9 = datasets[4]
# # electrode_14 = datasets[4]


# # electrode_8 = datasets[5]
# # electrode_13 = datasets[5]

# # electrode_12 = datasets[6]

# # horizontally propagating gaussian pulse plane wave (propagate from left to right)

# electrode_0 = datasets[0]
# electrode_1 = datasets[0]
# electrode_2 = datasets[0]
# electrode_3 = datasets[0]

# electrode_4 = datasets[1]
# electrode_5 = datasets[1]
# electrode_6 = datasets[1]
# electrode_7 = datasets[1]

# electrode_8 = datasets[2]
# electrode_9 = datasets[2]
# electrode_10 = datasets[2]
# electrode_11 = datasets[2]

# electrode_12 = datasets[3]
# electrode_13 = datasets[3]
# electrode_14 = datasets[3]
# electrode_15 = datasets[3]

# data_arr = np.stack([electrode_0, electrode_1, electrode_2, electrode_3, electrode_4, electrode_5, electrode_6, electrode_7, electrode_8, electrode_9, electrode_10, electrode_11, electrode_12, electrode_13, electrode_14, electrode_15], axis = 1)
# data_df = pd.DataFrame(data_arr)

# #%%

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Parameters
# alpha = 0.01  # Controls the width of the Gaussian envelope
# k = 2 * np.pi / 10  # Wave number (spatial frequency)
# omega = 2 * np.pi / 20  # Angular frequency
# default_theta = np.radians(30)  # Default propagation angle (30 degrees)

# # Grid
# x = np.linspace(0, 12, 500)
# y = np.linspace(0, 12, 500)
# x, y = np.meshgrid(x, y)

# # Time for animation
# time_steps = np.linspace(0, 40, 100)

# # Function to calculate the wave pulse
# def gaussian_plane_wave(x, y, t, theta):
#     # Components along x and y
#     x_comp = x * np.cos(theta)
#     y_comp = y * np.sin(theta)
    
#     delay = (i * distance_between_points) / speed  # Time delay for the ith point
#     # Gaussian envelope and wave
#     # envelope = np.exp(-alpha * (x**2 + y**2))
#     # wave = np.cos(k * (x_comp + y_comp) - omega * t)
#     amplitude = np.exp(-((time - (pulse_center + delay)) ** 2) / (2 * pulse_width**2))
#     return amplitude

# # Function to calculate amplitude at a specific coordinate
# def calculate_amplitude_at_point(x_coord, y_coord, t, theta=default_theta):
#     x_comp = x_coord * np.cos(theta)
#     y_comp = y_coord * np.sin(theta)
#     envelope = np.exp(-alpha * (x_coord**2 + y_coord**2))
#     wave = np.cos(k * (x_comp + y_comp) - omega * t)
#     return envelope * wave

# # Example usage of amplitude calculation
# x_coord, y_coord = 10, 10  # Example coordinates
# t = 10  # Example time step
# amplitude = calculate_amplitude_at_point(x_coord, y_coord, t)
# print(f"Amplitude at ({x_coord}, {y_coord}) at time {t}: {amplitude}")

# #%%
# # Plotting
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlim(-50, 50)
# ax.set_ylim(-50, 50)
# ax.set_title("Gaussian Plane Wave Pulse")
# ax.set_xlabel("x")
# ax.set_ylabel("y")

# # Initialize plot
# im = ax.imshow(np.zeros_like(x), extent=[-50, 50, -50, 50], cmap='viridis', origin='lower')

# # Update function for animation
# def update(frame):
#     t = frame  # Current time step
#     z = gaussian_plane_wave(x, y, t, default_theta)
#     im.set_data(z)
#     return [im]

# # Create animation
# ani = FuncAnimation(fig, update, frames=time_steps, interval=50, blit=True)

# plt.show()

