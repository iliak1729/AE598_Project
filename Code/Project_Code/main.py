# This file is a part of the AE598 - Multiphase Flow project of Ilia Kheirkhah, Parin Trivedi, and Conrad Delgado.
# This main file is the file through with the simulation is run and results are compiled.

# Imports
import numpy as np
import matplotlib.pyplot as plt
# Particle Sim Function Import
from particle_simulation import *
from tqdm import tqdm

# Main Function
def main():
    # Important Inputs
    work_file_path = '/home/iliak2/Desktop/Homework/MultiphaseFlow/AE598_Project/'
    data_file_path = work_file_path + 'Code/Flow_Data/hit/ae598-mf-hw3-data.npz'
    # This will install the packages needed for the HW (if these are not installed yet)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams.update({'font.size': 18})
    # The following lines import LaTeX packages to make plots nicer.
    UseLaTeX = False # This can be set to "True" if LaTeX is available on your computer
    if UseLaTeX:
        plt.rcParams['text.usetex'] = True
        plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{color}')
    # ================================================ Load Data ==========================================================
    loaded_data = np.load(data_file_path) # If your data file is located in a separate folder, you need to provide the full path to the file
    L                            = float(loaded_data['L']) # Domain length
    nx                           = int(loaded_data['nx']) # Number of cells per direction
    dx                           = L/float(nx) # Mesh spacing
    nu                           = float(loaded_data['nu']) # Kinematic viscosity
    vel                          = np.array(loaded_data['velocity'].reshape((nx,nx,nx,3),order='F'),dtype=np.float32) # Load velocity from data file
    ng                           = 3 # Number of ghost layers for interpolation and gradient calculation
    velg                         = np.pad(vel, ((ng,ng),(ng,ng),(ng,ng),(0,0)), 'wrap') # Padding domain with 'ng' periodic ghost layers
    u, v, w                      = vel[:,:,:,0],  vel[:,:,:,1],  vel[:,:,:,2] # Velocity components
    vel_mag                      = np.sqrt(u**2+v**2+w**2) # Velocity magnitude
    ug, vg, wg                   = velg[:,:,:,0], velg[:,:,:,1], velg[:,:,:,2] # Velocity components with ghost layer padding (needed for gradient)
    # Compute velocity gradient
    grad_u_x, grad_u_y, grad_u_z = np.array(np.gradient(ug, dx))[:,ng:-ng,ng:-ng,ng:-ng]
    grad_v_x, grad_v_y, grad_v_z = np.array(np.gradient(vg, dx))[:,ng:-ng,ng:-ng,ng:-ng]
    grad_w_x, grad_w_y, grad_w_z = np.array(np.gradient(wg, dx))[:,ng:-ng,ng:-ng,ng:-ng]
    # Compute turbulent kinetic energy and dissipation rate
    tke                          = 0.5*np.mean(u**2+v**2+w**2)
    eps                          = 2.0*nu*np.mean(grad_u_x**2+grad_v_y**2+grad_w_z**2+0.25*(grad_u_y+grad_v_x)**2+0.25*(grad_u_z+grad_w_x)**2+0.25*(grad_v_z+grad_w_y)**2)
    # Compute Taylor-scale Reynolds number and flow length- and time-scales
    u_rms                        = np.sqrt(2.0*tke/3.0) # RMS turbulent velocity [m/s]
    Taylor_lambda                = np.sqrt(15.0*nu*u_rms**2/eps)  # Taylor microscale [m]
    Re_lambda                    = u_rms*Taylor_lambda/nu # Taylor-scale Reynolds number [-]
    eta                          = (nu**3/eps)**0.25 # Kolmogorov length scale [m]
    tau_eta                      = (nu/eps)**0.5 # Kolmogorov time scale [s]
    u_eta                        = (nu*eps)**0.25 # Kolmogorov velocity scale [m/s]
    L_integral                   = u_rms**3/eps # Integral length scale [m]
    tau_L                        = L_integral/u_rms # Large-eddy turnover time [s]


    # ================================================ Print Features ==========================================================
    print("Number of computational cells = %i^3    [-]" % nx)
    print("                 Domain width = %.2e [m]" % L)
    print("          Kinematic viscosity = %.2e [m^2/s]" % nu)
    print("                 RMS velocity = %.2e [m/s]" % u_rms)
    print(" Taylor-scale Reynolds number = %.2e [-]" % Re_lambda)
    print("      Kolmogorov length scale = %.2e [m]" % eta)
    print("        Kolmogorov time scale = %.2e [s]" % tau_eta)
    print("    Kolmogorov velocity scale = %.2e [m/s]" % u_eta)
    print("        Integral length scale = %.2e [m]" % L_integral)
    print("           Eddy turnover time = %.2e [s]" % tau_L)


    # ================================================ Plot Data Example ==========================================================
    # Plot and save slice of the simulations snapshot
    plt.imshow(vel_mag[:,:,0], interpolation='spline36', cmap='PuOr_r', origin='lower', extent=[0, L, 0, L])
    x_coords = 0.5*dx + np.linspace(0, L, nx, endpoint=False); X, Y = np.meshgrid(x_coords, x_coords)
    plt.quiver(X[::3,::3], Y[::3,::3], u[::3,::3,-1],v[::3,::3,-1],color='black'); plt.axis('off')
    plt.savefig(work_file_path+"Code/Results/hit.pdf", dpi=600, bbox_inches='tight', pad_inches=0) 

# Run only here
if __name__ == '__main__':
    main()