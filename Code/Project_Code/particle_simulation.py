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

# Fluid Force - TO DO 
def get_drag_force(rho,u,v,w,R):
    print("Drag")
    return 2

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


def rk4_integrator(x0,dt,Nt,derivativeFunction): 
  x = np.copy(x0)
  t = 0.0
  for i in range(Nt): 
     # RK4 time integrator, which uses the derivative function to add everything together.
     # x should store position and velocity and the derivative function should return the acceleration and velocity of each particle
     # in an array that is the same size as x0.
     k1 = derivativeFunction(t,x)
     k2 = derivativeFunction(t+dt/2,x+dt*k1/2)
     k3 = derivativeFunction(t+dt/2,x+dt*k2/2)
     k4 = derivativeFunction(t+dt,x+dt*k3)

     x = x + dt*(k1+2*k2+2*k3+k4)/6
  return x