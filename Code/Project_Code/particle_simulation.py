# This file is a part of the AE598 - Multiphase Flow project of Conrad Delgado, Ilia Kheirkhah, Parin Trivedi
# This particle_simulation file holds the functions that are used by the main file. It will include anything that is needed like
# functions for each force, classes for particle data storage (if needed), etc.
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Gravity Force - TO DO 
def get_gravity_force(m,g):
    print("Gravity")
    return 1

# Drag Force - Update if needed
def get_drag_force(x,v,v_interp,tau_p):
    return (v_interp-v)/tau_p

# Torque
def get_torque(wf_interp, wp, tau_r):
   return (0.5 * wf_interp - wp) / tau_r

# Get Saffman Lift
def get_saffman_lift(rho,u,R):
    print("Saffman")
    return 3

# Magnus Lift
def get_magnus_lift(rho,u,R,omega):
    print("Gravity")
    return 4

# Contact Forces
def get_contact_force(u,R,v,omega):
    print("Contact")
    return 5

# Linear velocity interpolator -> return fluid velocity interpolated at locations x
def fluid_velocity_interpolator(x,L,dx,ug,vg,wg,ng): # Shape of input "x" is (N,3)
  # x = Locations
  # L = domain size
  # dx = Mesh Spacing
  # ug,vg,wg = Velocity Components with Ghost Cells
  # ng = Number of ghost layers for interpolation and gradient calculation
  xmod    = np.mod(x,L)
  indices = np.int_(np.floor(xmod/dx))
  phi     = (xmod - indices*dx)/dx
  up      = phi[:,0]*(ug[ng+indices[:,0]+1,ng+indices[:,1],ng+indices[:,2]]) +(1.0-phi[:,0])*(ug[ng+indices[:,0],ng+indices[:,1],ng+indices[:,2]])
  vp      = phi[:,1]*(vg[ng+indices[:,0],ng+indices[:,1]+1,ng+indices[:,2]]) +(1.0-phi[:,1])*(vg[ng+indices[:,0],ng+indices[:,1],ng+indices[:,2]])
  wp      = phi[:,2]*(wg[ng+indices[:,0],ng+indices[:,1],ng+indices[:,2]+1]) +(1.0-phi[:,2])*(wg[ng+indices[:,0],ng+indices[:,1],ng+indices[:,2]])
  return np.swapaxes(np.array([up,vp,wp]),0,1)

def fluid_vorticity_interpolator(x, L, dx, omega, ng):
  # x = Locations
  # L = domain size
  # dx = Mesh Spacing
  # omega = Vorticity with Ghost Cells
  # w1,w2,w3 = Vorticity Components with Ghost Cells
  # ng = Number of ghost layers for interpolation and gradient calculation
  w1 = omega[..., 0]; w2 = omega[..., 1]; w3 = omega[..., 2]
  xmod    = np.mod(x,L)
  indices = np.int_(np.floor(xmod/dx))
  phi     = (xmod - indices*dx)/dx
  w1p     = phi[:,0]*(w1[ng+indices[:,0]+1,ng+indices[:,1],ng+indices[:,2]]) +(1.0-phi[:,0])*(w1[ng+indices[:,0],ng+indices[:,1],ng+indices[:,2]])
  w2p     = phi[:,1]*(w2[ng+indices[:,0],ng+indices[:,1]+1,ng+indices[:,2]]) +(1.0-phi[:,1])*(w2[ng+indices[:,0],ng+indices[:,1],ng+indices[:,2]])
  w3p     = phi[:,2]*(w3[ng+indices[:,0],ng+indices[:,1],ng+indices[:,2]+1]) +(1.0-phi[:,2])*(w3[ng+indices[:,0],ng+indices[:,1],ng+indices[:,2]])   

  return np.swapaxes(np.array([w1p,w2p,w3p]),0,1)


def rk4_integrator(x0,v0,w0,dt,Nt,derivativeFunction): 
  x = np.copy(x0) # position
  v = np.copy(v0) # translational velocity
  w = np.copy(w0) # angular velocity
  t = 0.0
  for i in tqdm(range(Nt)): 
    # First Step
    (k1x,k1v,k1w) = derivativeFunction(t,x,v,w)
    # Second Step
    (k2x,k2v,k2w) = derivativeFunction(t,x+k1x*dt/2,
                                         v+k1v*dt/2,
                                         w+k1w*dt/2)
    # Third Step
    (k3x,k3v,k3w) = derivativeFunction(t,x+k2x*dt/2,
                                         v+k2v*dt/2,
                                         w+k2w*dt/2)
    # Fourth Step
    (k4x,k4v,k4w) = derivativeFunction(t,x+k3x*dt,
                                         v+k3v*dt,
                                         w+k3w*dt)
    # Final Step
    x = x + dt*(k1x+2*k2x+2*k3x+k4x)/6
    v = v + dt*(k1v+2*k2v+2*k3v+k4v)/6
    w = w + dt*(k1w+2*k2w+2*k3w+k4w)/6
    t = t + dt

    # Update this for storage of x,v,w
  return (x,v,w)

# For Plotting
def plot_particles(ax, position, z_slice_location, z_slice_thickness,L, title=''):
  # Compute particle locations modulo L (so that position is inside [0,L]^3)
  x = np.mod(position[:,0],L); y = np.mod(position[:,1],L); z = np.mod(position[:,2],L)
  # Create mask for particles inside the slice with provided thickness
  mask = np.abs(z-z_slice_location) > 0.5*z_slice_thickness
  # Plot all particles in slice
  ax.scatter(np.ma.MaskedArray(x, mask),np.ma.MaskedArray(y, mask),s=0.125,color='k')
  # Set plot title
  ax.set_title(title, fontsize=18); ax.margins(0); ax.set_xticks([]); ax.set_yticks([])