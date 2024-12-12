'''Author: Hiroki Kozuki'''
#%%
'''Imports'''
import time
import matplotlib.pyplot as plt
from modules import *

import jax.numpy as jnp
from jax import jit, lax, grad
from mpl_toolkits.mplot3d import Axes3D

#%%
'''Load data'''

data = "ElectrogramData.xlsx"

start = time.time()

L = loadData(data, n = 120) # Load electrogram data
print(L.coordinates().shape) # (16, 2) - x,y positions of electrodes 
print(L.ele_signals().shape) # (2201, 16) - 2201 voltage data at 16 electrodes.

time_indices = [1732, 1733, 1734, 1735, 1736, 1737, 1743, 
                1744, 1745, 1746, 1747, 1748, 1749, 1750]

for time_index in time_indices:

    # time_index = 1731 # row 1730 - 1740 - 1750 (1732 - 1742 - 1752 in excel)

    print(L.ele_signals().iloc[time_index].shape) # (16, ) 
    print(L.ele_signals().iloc[time_index])
    print(L.time_data().shape)   # (2201, ) - 2201 time points. 

    end = time.time()
    print("time take (new)", end - start)


    '''Traditional method: using velocity map to interpolate potential field using wave equation:
    Accelerated with JAX (runs on GPU/TPU), highly vectorised code'''


    # Parameters
    grid_size = 600  # 600 x 600 grid
    domain_length = 12.0  # 12 mm x 12 mm domain
    dx = domain_length / grid_size
    dy = domain_length / grid_size

    # Electrode positions (4x4 grid, separated by 4 mm)
    electrode_spacing = 4.0  # mm
    electrode_indices = [
        (int(i * grid_size / 3), int(j * grid_size / 3))
        for i in range(0, 4) for j in range(0, 4)
    ]
    print(electrode_indices) # [(0, 0), (0, 200), (0, 400), (0, 600), (200, 0), (200, 200), ..., (600, 400), (600, 600)]

    # Voltage values at electrode positions
    electrode_values = jnp.array(L.ele_signals().iloc[time_index])
    print(electrode_values)
    x_elec = jnp.array(L.x_position())
    y_elec = jnp.array(L.y_position())
    print(x_elec)
    print(y_elec)

    # Velocity profile and grid initialization
    c = 1000 * jnp.ones((grid_size + 1, grid_size + 1))  # Uniform velocity , 1 m/s = 1000 mm/s
    u = jnp.zeros((grid_size + 1, grid_size + 1))

    # Set electrode values
    for (i, j), value in zip(electrode_indices, electrode_values):
        u = u.at[i, j].set(value)


    import jax.numpy as jnp
    from jax import jit, lax

    @jit
    def jacobi_iteration(u, c, electrode_indices, electrode_values, tolerance, max_iterations):
        grid_size = u.shape[0] - 1
        c_x_plus = (c[1:, :] + c[:-1, :]) / 2
        c_x_minus = c_x_plus
        c_y_plus = (c[:, 1:] + c[:, :-1]) / 2
        c_y_minus = c_y_plus

        def body_func(val):
            u, iteration, converged = val
            u_new = u.at[1:-1, 1:-1].set(
                (
                    c_x_plus[1:, 1:-1] * u[2:, 1:-1] +
                    c_x_minus[:-1, 1:-1] * u[:-2, 1:-1] +
                    c_y_plus[1:-1, 1:] * u[1:-1, 2:] +
                    c_y_minus[1:-1, :-1] * u[1:-1, :-2]
                ) / (
                    c_x_plus[1:, 1:-1] + c_x_minus[:-1, 1:-1] +
                    c_y_plus[1:-1, 1:] + c_y_minus[1:-1, :-1]
                )
            )

            # Set electrode values
            for (i, j), value in zip(electrode_indices, electrode_values):
                u_new = u_new.at[i, j].set(value)

            # Check convergence
            diff = jnp.max(jnp.abs(u_new - u))
            converged = diff < tolerance
            return u_new, iteration + 1, converged

        def cond_func(val):
            _, iteration, converged = val
            return jnp.logical_and(iteration < max_iterations, jnp.logical_not(converged))

        # Initial state
        init_val = (u, 0, False)

        # Run the while loop
        final_u, final_iteration, final_converged = lax.while_loop(cond_func, body_func, init_val)
        return final_u, final_iteration, final_converged


    # Run Jacobi iteration
    tolerance = 1e-6
    max_iterations = 1000000
    start = time.time()
    u, iteration, converged = jacobi_iteration(u, c, electrode_indices, electrode_values, tolerance, max_iterations)
    end = time.time()

    print(f"Time taken (JAX): {end - start:.2f} seconds")
    if converged:
        print(f"Jacobi iteration converged in {iteration} iterations.")
    else:
        print(f"Jacobi iteration did not converge within the maximum of {max_iterations} iterations.")

    # Plot 2D results
    plt.figure(figsize=(8, 6))
    plt.imshow(u, extent=[0, domain_length, 0, domain_length], origin='lower', cmap='viridis')
    plt.colorbar(label='Voltage (mV)')
    plt.scatter(x_elec, y_elec, c='red', label='Electrodes')
    # plt.title(f'Interpolated Voltage Field (tol={tolerance}, max_iter={max_iterations})')
    plt.title(f'2D Interpolated Voltage Field (t_index={time_index})', fontsize=16)
    plt.xlim(-1.0, 13.0)
    plt.ylim(-1.0, 13.0)
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.legend()
    plt.show()

    # Plot 3D results
    # Generate 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Create grid for 3D plot
    x = jnp.linspace(0, domain_length, grid_size + 1)
    y = jnp.linspace(0, domain_length, grid_size + 1)
    X, Y = jnp.meshgrid(x, y)
    # Plot the surface
    surf = ax.plot_surface(X, Y, u, cmap='viridis', edgecolor='none')
    # Scatter electrode points
    ax.scatter(x_elec, y_elec, electrode_values.T, c='red', s=50, label='Electrodes')
    print(x_elec, y_elec)
    # Add vertical bars indicating the electrode voltages
    bar_width = 0.1  # Width of the bars
    for x, y, voltage in zip(x_elec, y_elec, electrode_values):
        ax.bar3d(
            x, y, 0,  # Start at the base plane (z=0)
            bar_width, bar_width, voltage,  # Bar width in x and y, height is the voltage
            color='red', alpha=0.8
        )
    # Add labels and color bar
    # ax.set_title('3D Interpolated Voltage Field', fontsize=16)
    ax.set_title(f'3D Interpolated Voltage Field (t_index={time_index})', fontsize=16)
    ax.set_xlabel('X-axis [mm]', fontsize=12)
    ax.set_ylabel('Y-axis [mm]', fontsize=12)
    ax.set_zlabel('Voltage (mV)', fontsize=12)
    fig.colorbar(surf, ax=ax, label='Voltage (mV)')
    # Show the legend
    ax.legend()
    # Show the plot
    plt.show()
# %%



