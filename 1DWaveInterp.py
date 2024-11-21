#%%
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

#%%
# 1.1) Propagation of 1D wave equation with constant speed c.
# Euler method
#'''
# Wave equation parameters
L = 1.0            # Length of the domain
nx = 100           # Number of spatial points
dx = L / nx        # Spatial step
c = 1.0            # Wave speed
t_max = 2.0        # Maximum simulation time
dt = 0.005         # Time step
alpha = c * dt / dx  # CFL number

# CFL Stability condition
if alpha > 1:
    raise ValueError("Stability condition not satisfied. Reduce dt or increase dx.")

# Initialize displacement field
u = np.zeros(nx)       # Current displacements over x. (t)
u_new = np.zeros(nx)   # Displacements over x for next time step. (t+1)
u_old = np.zeros(nx)   # Displacements over x from previous time step. (t-1) 

# Initial condition: Gaussian pulse in the center
x = np.linspace(0, L, nx)
u = np.exp(-100 * (x - L / 4)**2)
u_old[:] = u  # Set initial state

# Prepare animation
fig, ax = plt.subplots()
line, = ax.plot(x, u, lw=2)
ax.set_xlim(0, L)
ax.set_ylim(-1.1, 1.1)
ax.set_title("1D Wave Equation Propagation")
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")

# Update function for animation
def update(frame):
    global u, u_new, u_old
    for i in range(1, nx - 1):
        u_new[i] = 2 * u[i] - u_old[i] + alpha**2 * (u[i+1] - 2 * u[i] + u[i-1])
    
    # Swap arrays
    u_old[:] = u
    u[:] = u_new
    
    line.set_ydata(u)
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=int(t_max / dt), blit=True, interval=20)
plt.show()
#'''
# %%
# 1.2) Propagation of 1D wave equation with constant speed varying in x -> c(x).
# Euler method
#'''
# Wave equation parameters
L = 5.0            # Length of the domain # 5.0
nx = 500           # Number of spatial points # 200 (increasing nx solves most instability issues like noise and wave breaking, but make sure to increase L too).
dx = L / nx        # Spatial step
t_max = 2.0        # Maximum simulation time
dt = 0.005         # Time step # 0.005

# Spatially variable wave speed: c(x)
x = np.linspace(-L/2, L/2, nx)
#c = 1.0 + 0.5 * np.sin(2 * np.pi * x)  # Example: wave speed varies sinusoidally
c = np.heaviside(x, 1) + 1 # np.heaviside gives 0 if x < 0 and 1 if x => 0.
# half = int(nx/2)
# c1 = np.ones(half)
# c2 = np.ones(half) + 1
# c = np.concatenate((c1,c2))
alpha = c * dt / dx  # CFL number (array because c is variable)

# Ensure CFL stability condition
if np.any(alpha > 1):
    raise ValueError("Stability condition not satisfied. Reduce dt or increase dx.")

# Initialize displacement field
u = np.zeros(nx); u_new = np.zeros(nx); u_old = np.zeros(nx)

# Initial condition: Gaussian pulse in the center
u = np.exp(-100 * (x - L / 5)**2); u_old[:] = u  # Set initial state

# Prepare animation
fig, ax = plt.subplots(); line, = ax.plot(x, u, lw=2)
ax.set_xlim(-L/2, L/2); ax.set_ylim(-1.1, 1.1)
ax.set_title("1D Wave Equation with Variable Wave Speed c(x)")
ax.set_xlabel("x"); ax.set_ylabel("u(x, t)")

# Update function for animation
def update(frame, periodic_BC):
    global u, u_new, u_old

    if periodic_BC == True:
        # Apply periodic boundary conditions
        c_left = c[0]  # wave speed at the left boundary
        c_right = c[-1]  # wave speed at the right boundary
        alpha_left = c_left * dt / dx
        alpha_right = c_right * dt / dx
        
        # Left boundary (wrap around to the right)
        u_new[0] = (
            2 * u[0]
            - u_old[0]
            + alpha_left**2 * (u[1] - 2 * u[0] + u[-1])
        )
        # Right boundary (wrap around to the left)
        u_new[-1] = (
            2 * u[-1]
            - u_old[-1]
            + alpha_right**2 * (u[0] - 2 * u[-1] + u[-2])
        )

    for i in range(1, nx - 1):
        # Use variable wave speed in finite difference scheme
        c_i = c[i]  # wave speed at position i
        alpha_i = c_i * dt / dx
        u_new[i] = (
            2 * u[i]
            - u_old[i]
            + alpha_i**2 * (u[i+1] - 2 * u[i] + u[i-1])
        )
    # Swap arrays
    u_old[:] = u; u[:] = u_new
    line.set_ydata(u)
    return line,

# Create animation
periodic_BC = True
ani = FuncAnimation(fig, update, frames=int(t_max / dt), fargs=(periodic_BC,), blit=True, interval=20)
plt.show()
#'''
# %%
# 1.3) Propagation of 1D wave equation with constant speed varying in x -> c(t,x).
# Euler method

# Wave equation parameters
L = 5.0            # Length of the domain # 5.0
nx = 300           # Number of spatial points # 300
dx = L / nx        # Spatial step 
t_max = 2.0        # Maximum simulation time # 2.0
dt = 0.005         # Time step # 0.001 # 0.005

# Spatial grid
x = np.linspace(-L/2, L/2, nx)

# Define wave speed c(t, x)
def wave_speed(t, x):
    """Spatially and temporally varying wave speed."""
    
    #return 1.0 + np.abs(np.sin(np.pi * x) * np.sin(2 * np.pi * t))
    
    if t > 1:
        return 1.5 * (np.heaviside(x, 1) + 1) # np.heaviside gives 0 if x < 0 and 1 if x => 0.
    else:
        return np.heaviside(x, 1) + 1

# Initial wave speed at t=0
c = wave_speed(0, x)

alpha = c * dt / dx  # CFL number (array because c is variable)

# Initialize wave field
u = np.zeros(nx) ; u_new = np.zeros(nx) ; u_old = np.zeros(nx)

# Initial condition: Gaussian pulse in the center
u = np.exp(-100 * (x - L / 5)**2) ; u_old[:] = u  # Set initial state

# Prepare animation
fig, ax = plt.subplots() ; line, = ax.plot(x, u, lw=2)
ax.set_xlim(-L/2, L/2) ; ax.set_ylim(-1.1, 1.1)
ax.set_title("1D Wave Equation with Variable Wave Speed c(t,x)")
ax.set_xlabel("x") ; ax.set_ylabel("u(x, t)")
ax.axvspan(-L/2, 0, color='lightgreen', alpha=0.3, label="slower conduction")

# Time tracking
#time = [0]  # Use a list to track mutable time across frames
t = 0.0
# Update function for animation
def update(frame, periodic_BC):
    global u, u_new, u_old, c, t, alpha
    #t = time[0]  # Current time
    c = wave_speed(t, x)  # Update wave speed based on current time

    # Ensure CFL stability condition
    if np.any(alpha > 1):
        raise ValueError("Stability condition not satisfied. Reduce dt or increase dx.")
    if np.any(u > 2):
        raise ValueError("Wave amplitude exploding. Reduce dt or increase dx.")
    
    if periodic_BC == True:
        # Apply periodic boundary conditions
        # wave speeds at the left and right boundaries.
        c_left = c[0] ; c_right = c[-1]
        alpha_left = c_left * dt / dx ; alpha_right = c_right * dt / dx
        
        # Left boundary (wrap around to the right)
        u_new[0] = (
            2 * u[0]
            - u_old[0]
            + alpha_left**2 * (u[1] - 2 * u[0] + u[-1])
        )
        # Right boundary (wrap around to the left)
        u_new[-1] = (
            2 * u[-1]
            - u_old[-1]
            + alpha_right**2 * (u[0] - 2 * u[-1] + u[-2])
        )

    for i in range(1, nx-1):
        # Use variable wave speed in finite difference scheme
        c_i = c[i]  # wave speed at position i
        alpha_i = c_i * dt / dx
        u_new[i] = (
            2 * u[i]
            - u_old[i]
            + alpha_i**2 * (u[i+1] - 2 * u[i] + u[i-1])
        )
    
    # Swap arrays
    u_old[:] = u ; u[:] = u_new

    # Increment time
    #time[0] += dt
    t += dt

    line.set_ydata(u)
    return line,

# Create animation
periodic_BC = True
ani = FuncAnimation(
    fig, update, frames=int(t_max / dt),
    fargs=(periodic_BC,),  
    blit=True, interval=20
)
writer = FFMpegWriter(fps = 30, bitrate = 500, codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
        # writer = animation.FFMpegWriter(fps = 24, bitrate = 10000, codec = "libx264", extra_args = ["-pix_fmt", "yuv420p"])
ani.save('ani_Euler_1D_ctx.mp4', writer=writer)
plt.show()

# %%
###### SEVERE INSTABILITY, BUNCH OF OVERFLOW ERRORS WITHIN RK4 METHOD. --> NEEDS FIXING. 

# 1.4) Propagation of 1D wave equation with constant speed varying in x -> c(t,x).
# RK4 method
'''
# Wave equation parameters
L = 5.0            # Length of the domain
nx = 300           # Number of spatial points
dx = L / nx        # Spatial step
t_max = 2.0        # Maximum simulation time
dt = 0.0001        # Reduced time step for stability

# Spatial grid
x = np.linspace(-L/2, L/2, nx)

# Define wave speed c(t, x)
def wave_speed(t, x):
    """Spatially and temporally varying wave speed."""
    return np.clip((np.heaviside(x, 1) + 1) + np.heaviside(t - 1, 1), 0, 3)

# Initialize wave field
u = np.zeros(nx)  # Current wave field
u_old = np.zeros(nx)  # Wave field at the previous time step

# Initial condition: Smoothed Gaussian pulse in the center
u[:] = np.exp(-25 * (x - L / 5)**2)
u_old[:] = u  # Initialize `u_old` with the same initial condition

# Prepare figure for animation
fig, ax = plt.subplots()
line, = ax.plot(x, u, lw=2)
ax.set_xlim(-L/2, L/2)
ax.set_ylim(-1.1, 1.1)
ax.set_title("1D Wave Equation with RK4 and Variable Wave Speed")
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")

# Time tracking
time = [0.0]  # Time as a mutable object

# RK4 helper function
def compute_rhs(u, c, periodic_BC):
    """Compute the right-hand side of the wave equation."""
    rhs = np.zeros_like(u)
    
    for i in range(1, nx - 1):
        rhs[i] = (c[i] / dx)**2 * (u[i+1] - 2*u[i] + u[i-1])

    if periodic_BC:
        # Periodic boundary conditions
        rhs[0] = (c[0] / dx)**2 * (u[1] - 2*u[0] + u[-1])
        rhs[-1] = (c[-1] / dx)**2 * (u[0] - 2*u[-1] + u[-2])
    
    return rhs

# Update function using RK4
def update(frame, periodic_BC):
    global u, u_old

    # Current time
    t = time[0]

    # Compute wave speed
    c = wave_speed(t, x)

    # Ensure CFL stability
    if np.any(c * dt / dx > 0.8):  # Stricter CFL condition
        raise ValueError("CFL condition violated: reduce dt or increase dx.")

    # RK4 steps
    k1 = compute_rhs(u, c, periodic_BC)
    k2 = compute_rhs(u + 0.5 * dt * k1, c, periodic_BC)
    k3 = compute_rhs(u + 0.5 * dt * k2, c, periodic_BC)
    k4 = compute_rhs(u + dt * k3, c, periodic_BC)

    # Check for numerical instability
    if np.isnan(k1).any() or np.isnan(k2).any() or np.isnan(k3).any() or np.isnan(k4).any():
        print("Debugging Info:")
        print(f"k1: {k1}")
        print(f"k2: {k2}")
        print(f"k3: {k3}")
        print(f"k4: {k4}")
        raise ValueError("Numerical instability detected: NaN values in RK4 steps.")

    # RK4 update for wave field
    u_new = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Update old and current wave fields
    u_old[:] = u
    u[:] = u_new

    # Increment time
    time[0] += dt

    # Update the line data for animation
    line.set_ydata(u)
    return line,

# Create animation
periodic_BC = True
ani = FuncAnimation(fig, update, frames=int(t_max / dt), fargs=(periodic_BC,), blit=True, interval=20)

# Save animation as MP4
writer = FFMpegWriter(fps=15, bitrate=500)
ani.save('wave_equation_RK4_stabilized.mp4', writer=writer)

# Show the animation
plt.show()

'''

#%%

# 1.3) Propagation of 2D wave equation with constant speed varying in x -> c(t,x,y).
# Euler method

# Wave equation parameters
L = 5.0             # Length of the domain
nx, ny = 100, 100   # Number of spatial points in x and y directions
dx = L / nx         # Spatial step in x
dy = L / ny         # Spatial step in y
t_max = 2.0         # Maximum simulation time
dt = 0.005          # Time step

# Spatial grid
x = np.linspace(-L/2, L/2, nx)
y = np.linspace(-L/2, L/2, ny)
X, Y = np.meshgrid(x, y)

# Define wave speed c(x, y, t)
def wave_speed(t, x, y):
    """Spatially and temporally varying wave speed."""
    
    #return 2.0 + 3.0 * np.abs(np.sin(np.pi * (x-0.8)) * np.sin(2 * np.pi * (y-0.2)) * np.sin(3 * np.pi * t))
    
    if t > 1:
        return 2 * (1.5*np.heaviside(x, 1) * 1.5*np.heaviside(y, 1)) + 2 # np.heaviside gives 0 if x < 0 and 1 if x => 0.
    else:
        return (1.5*np.heaviside(x, 1) * 1.5*np.heaviside(y, 1)) + 2

# Initial wave speed at t=0
c = wave_speed(0, X, Y)
alpha_x = c * dt / dx  # CFL number in x
alpha_y = c * dt / dy  # CFL number in y

# Initialize wave field
u = np.zeros((nx, ny))
u_new = np.zeros((nx, ny))
u_old = np.zeros((nx, ny))

# Initial condition: Gaussian pulse in the center
u = np.exp(-50 * ((X - L / 5)**2 + (Y - L / 5)**2))
u_old[:] = u

# Ensure CFL stability condition
if np.any(alpha_x > 1) or np.any(alpha_y > 1):
    raise ValueError("Stability condition not satisfied. Reduce dt or increase dx/dy.")

# Prepare 3D visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, u, cmap="viridis")
ax.set_zlim(-1.1, 1.1)
ax.set_title("2D Wave Equation with Variable Wave Speed c(x, y, t)")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Time tracking
t = 0.0

# Update function for animation
def update(frame):
    global u, u_new, u_old, c, t, alpha_x, alpha_y

    # Current time
    t += dt
    c = wave_speed(t, X, Y)  # Update wave speed
    alpha_x = c * dt / dx
    alpha_y = c * dt / dy

    # Finite difference scheme for 2D wave equation
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u_new[i, j] = (
                2 * u[i, j]
                - u_old[i, j]
                + alpha_x[i, j]**2 * (u[i+1, j] - 2 * u[i, j] + u[i-1, j])
                + alpha_y[i, j]**2 * (u[i, j+1] - 2 * u[i, j] + u[i, j-1])
            )

    # Apply periodic boundary conditions
    u_new[0, :] = u_new[-2, :]
    u_new[-1, :] = u_new[1, :]
    u_new[:, 0] = u_new[:, -2]
    u_new[:, -1] = u_new[:, 1]

    # Swap arrays
    u_old[:, :] = u
    u[:, :] = u_new

    # Update the surface plot
    ax.clear()
    ax.set_zlim(-0.5, 0.5)
    ax.set_title("2D Wave Equation with Variable Wave Speed c(x, y, t)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    surface = ax.plot_surface(X, Y, u, cmap="viridis")
    return surface,

# Create animation
ani = FuncAnimation(fig, update, frames=int(t_max / dt), interval=20, blit=False)

# Save animation as a video
writer = FFMpegWriter(fps=30, bitrate=500, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
ani.save('ani_Euler_2D_ctx.mp4', writer=writer)

plt.show()

