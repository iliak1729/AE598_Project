# This file is a part of the AE598 - Multiphase Flow project of Conrad Delgado, Ilia Kheirkhah, Parin Trivedi
# This particle_simulation file holds the functions that are used by the main file. It will include anything that is needed like
# functions for each force, classes for particle data storage (if needed), etc.
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Gravity Force - TO DO 
def get_gravity_force(g):
    return g

# Stokes Drag Term 
def get_stokes_drag(v, u_interp, tau_p):
  return (u_interp-v)/tau_p

# Faxen Drag Term
def get_faxen_correction(mu,rho_particle,lap_u):
   return 3*mu*lap_u/(4*rho_particle)

# Torque
def get_torque(wf_interp, wp, tau_r):
   return (0.5 * wf_interp - wp) / tau_r

# Get Saffman Lift
def get_saffman_lift(R,rho_p,rho,mu,up,u_interp,wf_interp):
    K = 6.46
    coeff = 3*K/(4*np.pi * R* rho_p)
    mag_vorticity = np.linalg.norm(wf_interp, axis=1, keepdims=True)

    inside_root = mu * rho / mag_vorticity

    relative_velocity = u_interp - up
    cross_product = np.zeros(relative_velocity.shape)
    cross_product[:,0] = relative_velocity[:,1]*wf_interp[:,2] - relative_velocity[:,2]*wf_interp[:,1]
    cross_product[:,1] = relative_velocity[:,2]*wf_interp[:,0] - relative_velocity[:,0]*wf_interp[:,2] 
    cross_product[:,2] = relative_velocity[:,0]*wf_interp[:,1] - relative_velocity[:,1]*wf_interp[:,0]

    return coeff * np.sqrt(inside_root) * cross_product

# Magnus Lift
def get_magnus_lift(lambda_rho,u_p,omega_p,u_interp,wf_interp):
    coeff = 3*lambda_rho/4

    relative_rotation = 0.5*wf_interp - omega_p

    relative_velocity = u_interp - u_p

    cross_product = np.zeros(relative_velocity.shape)
    cross_product[:,0] = relative_rotation[:,1]*relative_velocity[:,2] - relative_rotation[:,2]*relative_velocity[:,1]
    cross_product[:,1] = relative_rotation[:,2]*relative_velocity[:,0] - relative_rotation[:,0]*relative_velocity[:,2] 
    cross_product[:,2] = relative_rotation[:,0]*relative_velocity[:,1] - relative_rotation[:,1]*relative_velocity[:,0]

    return coeff * cross_product

# surface (contact) velocity
def surface_velocity_3d(v, omega, r, n, sign):
    """
    Surface velocity at contact point.
    v      : linear velocity 
    omega  : angular velocity
    r      : radius
    n      : unit normal from particle 1 -> 2
    sign   : +1 for particle 1 (contact at +r n)
             -1 for particle 2 (contact at -r n)
    """
    r_c = sign * r * n            # center -> contact point
    v_rot = np.cross(omega, r_c)  # rotational velocity at contact
    return v + v_rot, r_c

# Contact Forces
def get_contact_force(
    x1, v1, omega1, m1, r1,
    x2, v2, omega2, m2, r2,
    k, eta, mu, kt, FtOld, dt
):
    """
    Inputs:
      x1, x2    : positions
      v1, v2    : linear velocities
      omega1, omega2 : angular velocities
      m1, m2    : masses
      r1, r2    : radii
      k, eta    : normal stiffness and damping
      mu        : friction coefficient
      kt        : tangential stiffness
      FtOld     : previous tangential force
      dt        : time step

    Returns:
      a1, a2        : linear accelerations
      alpha1, alpha2: angular accelerations
      Ft_new        : updated tangential force to store for next step
    """
    # Solid sphere moment of inertia
    I1 = 0.4 * m1 * r1**2
    I2 = 0.4 * m2 * r2**2

    # Vector from 1 to 2
    dx = x2 - x1
    dist = np.linalg.norm(dx)

    # No contact -> zero everything and reset Ft to zero
    if dist >= r1 + r2 or dist == 0.0:
        a1 = np.zeros(3)
        a2 = np.zeros(3)
        alpha1 = np.zeros(3)
        alpha2 = np.zeros(3)
        Ft_new = np.zeros(3)
        return a1, a2, alpha1, alpha2, Ft_new

    # Unit normal
    n = dx / dist

    # Overlap
    delta = (r1 + r2) - dist

    # Surface velocities at contact
    v_surf1, r_c1 = surface_velocity_3d(v1, omega1, r1, n, +1)
    v_surf2, r_c2 = surface_velocity_3d(v2, omega2, r2, n, -1)

    # Relative contact velocity
    v_rel = v_surf1 - v_surf2

    # Normal and tangential components
    vn = np.dot(v_rel, n)
    v_n = vn * n
    v_t = v_rel - v_n
    vt_mag = np.linalg.norm(v_t)

    # Normal force (spring + dashpot)
    F_n = -k * delta * n - eta * v_n

    # tangential force
    if vt_mag > 1e-12:
        # Trial tangential spring update
        FtTrial = FtOld - kt * v_t * dt
        phi_trial = np.linalg.norm(FtTrial) - mu * np.linalg.norm(F_n)

        if phi_trial <= 0.0:
            F_t = FtTrial
        else:
            F_t = mu * np.linalg.norm(F_n) * FtTrial / np.linalg.norm(FtTrial)

        Ft_new = F_t.copy()
    else:
        F_t = np.zeros(3)
        Ft_new = np.zeros(3)

    # Total forces on particles
    F1 = F_n + F_t
    F2 = -F1

    # Torques from tangential force
    tau1 = np.cross(r_c1, F_t)
    tau2 = np.cross(r_c2, -F_t)

    # Linear accelerations
    a1 = F1 / m1
    a2 = F2 / m2

    # Angular accelerations
    alpha1 = tau1 / I1
    alpha2 = tau2 / I2

    return a1, a2, alpha1, alpha2, Ft_new


# Virtual Mass
def get_virtual_mass_force(mat_der_u,lambda_p):
   # The tricky part with this is that it has a dUp/dt in the equation, which is also what we are solving for. This means that when we add it to the LHS and then divide
   # that term basically acts as a (1+0.5*lambda_p) scale factor to other terms. We will return that scaling factor
   a_virt = 0.5*lambda_p*mat_der_u
   return a_virt,1+0.5*lambda_p

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

def fluid_vector_interpolator(x, L, dx, omega, ng):
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

# Take Laplacian Function. Includes Ghost Cells
def laplacian_scalar_field(f,dx):
   lap = (
      np.gradient(np.gradient(f ,dx, axis=0), dx, axis=0) + 
      np.gradient(np.gradient(f, dx, axis=1), dx, axis=1) +
      np.gradient(np.gradient(f, dx, axis=2), dx, axis=2)
      )
   return lap

def check_particles_periodic(x, L):
  # domain beg
  x = np.where(x < 0, x+L, x)
  # domain end
  x = np.where(x > L, x-L, x)
  return x

def rk4_integrator(x0,v0,w0,dt,Nt,L,derivativeFunction): 
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

    # check if particles have left domain - remap into domain
    x = check_particles_periodic(x, L)

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