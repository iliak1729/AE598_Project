# This file is a part of the AE598 - Multiphase Flow project of Ilia Kheirkhah, Parin Trivedi, and Conrad Delgado.
# This main file is the file through with the simulation is run and results are compiled.

# Imports
import numpy as np
import matplotlib.pyplot as plt
# Particle Sim Function Import
from particle_simulation import *
from tqdm import tqdm
import os

# Main Function
def main():
    # Important Inputs
    work_file_path = os.getcwd()  
    data_file_path = work_file_path + '/Code/Flow_Data/hit/ae598-mf-hw3-data.npz'
    # This will install the packages needed for the HW (if these are not installed yet)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams.update({'font.size': 18})
    # The following lines import LaTeX packages to make plots nicer.
    UseLaTeX = True # This can be set to "True" if LaTeX is available on your computer
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
    gradU = np.zeros(grad_u_x.shape + (3,3))
    gradU[..., 0, 0] = grad_u_x
    gradU[..., 0, 1] = grad_u_y
    gradU[..., 0, 2] = grad_u_z
    gradU[..., 1, 0] = grad_v_x
    gradU[..., 1, 1] = grad_v_y
    gradU[..., 1, 2] = grad_v_z
    gradU[..., 2, 0] = grad_w_x
    gradU[..., 2, 1] = grad_w_y
    gradU[..., 2, 2] = grad_w_z

    # Compute material derivative - assuming no d/dt
    mat_der_u = u*grad_u_x + v*grad_u_y + w*grad_u_z
    mat_der_v = u*grad_v_x + v*grad_v_y + w*grad_v_z
    mat_der_w = u*grad_w_x + v*grad_w_y + w*grad_w_z
    mat_der_vel  = np.stack((mat_der_u, mat_der_v, mat_der_w), -1)
    mat_der_vel_g = np.pad(mat_der_vel, ((ng,ng),(ng,ng),(ng,ng),(0,0)), 'wrap') # Padding domain with 'ng' periodic ghost layers
    # Strain rate and rotation rate tensors
    strain_rate = 0.5 * (gradU + np.swapaxes(gradU, -1, -2))  # symmetric
    rotation_rate = 0.5 * (gradU - np.swapaxes(gradU, -1, -2))  # antisymmetric
    Sg = np.pad(strain_rate,
            ((ng, ng), (ng, ng), (ng, ng), (0, 0), (0, 0)),
            mode='wrap')
    Rg = np.pad(rotation_rate,
                ((ng, ng), (ng, ng), (ng, ng), (0, 0), (0, 0)),
                mode='wrap')


    # Laplacians
    lap_u = laplacian_scalar_field(ug,dx)
    lap_v = laplacian_scalar_field(vg,dx)
    lap_w = laplacian_scalar_field(wg,dx)
    lap_vel  = np.stack((lap_u, lap_v, lap_w), -1)
    # Compute fluid vorticity
    omega_f = np.zeros((nx, nx, nx, 3))
    omega_f[:, :, :, 0] = grad_w_y - grad_v_z
    omega_f[:, :, :, 1] = grad_u_z - grad_w_x
    omega_f[:, :, :, 2] = grad_v_x - grad_u_y
    omega_fg = np.pad(omega_f, ((ng,ng),(ng,ng),(ng,ng),(0,0)), 'wrap')

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

    g = 9.81 # gravitational acceleration


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
    plt.savefig(work_file_path+"/Code/Results/hit.pdf", dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close('all')
    # ================================================ Derivative Functions ==========================================================
    # For tracers, since the velocity updates instantly, there is no ODE for it. 
    # The resulting velocity is then meaningless, but the positions are correct.
    def tracerDerivative(t,x,v):
        v_interp = fluid_velocity_interpolator(x,L,dx,ug,vg,wg,ng)
        dxdt = v_interp
        dvdt = 0
        return (dxdt,dvdt)
    # This takes in stokes number as an input, so lambda functions are needed to pass it into the f(t,x,v) form that is assumed by my rk4 integrator
    def inertialDerivative(t,x,v,St):
        tau_p = St*tau_eta
        v_interp = fluid_velocity_interpolator(x,L,dx,ug,vg,wg,ng)
        dxdt = v
        dvdt = (v_interp - v)/tau_p
        return(dxdt,dvdt)
    
    def particle_RHS(t,x,v,w,St,Re_p,rho_f):
        # parameter setup
        R = nu * Re_p / u_rms
        tau_p = St * tau_eta
        lambda_rho = 2 * R**2 / (9 * nu * tau_p)
        rho_p = rho_f / lambda_rho
        tau_r = R**2 / (15 * nu * lambda_rho)   
        mu = nu * rho_f
            
        # interpolate quantities
        u_interp = fluid_velocity_interpolator(x,L,dx,ug,vg,wg,ng)
        wf_interp = fluid_vector_interpolator(x, L, dx, omega_fg, ng)
        lap_u_interp = fluid_vector_interpolator(x, L, dx, lap_vel, ng)
        
        # form RHS
        dxdt = v
        dvdt = ( get_stokes_drag(v, u_interp, tau_p) + 
                 get_faxen_correction(mu, rho_p, lap_u_interp) + 
                 get_magnus_lift(lambda_rho, v, w, u_interp, wf_interp) + 
                 get_saffman_lift(R, rho_p, rho_f, mu, v, u_interp, wf_interp) ) 
        dwdt = get_torque(wf_interp,w,tau_r)

        return(dxdt,dvdt,dwdt)
    

    def particle_RHS_with_history(t,x,v,w,St,Re_p,rho_f,histories_array,i,j):
        # parameter setup
        R = nu * Re_p / u_rms
        tau_p = St * tau_eta
        lambda_rho = 2 * R**2 / (9 * nu * tau_p)
        rho_p = rho_f / lambda_rho
        tau_r = R**2 / (15 * nu * lambda_rho)   
        mu = nu * rho_f
            
        # interpolate quantities
        u_interp = fluid_velocity_interpolator(x,L,dx,ug,vg,wg,ng)
        wf_interp = fluid_vector_interpolator(x, L, dx, omega_fg, ng)
        lap_u_interp = fluid_vector_interpolator(x, L, dx, lap_vel, ng)
        mat_der_interp = fluid_vector_interpolator(x,L,dx,mat_der_vel_g,ng)

        # Get accelerations
        a_drag = get_stokes_drag(v, u_interp, tau_p)
        a_faxen = get_faxen_correction(mu,rho_p,lap_u_interp)
        a_undisturbed = get_undisturbed_force(lambda_rho,mat_der_interp,g)
        a_magnus = get_magnus_lift(lambda_rho, v, w, u_interp, wf_interp)
        a_saffman = get_saffman_lift(R, rho_p, rho_f, mu, v, u_interp, wf_interp)
        # Update History
        if(j[0] == 3):
            # Calculate Mean values
            a_drag_mag = np.linalg.norm(a_drag,axis=1)
            a_drag_average = np.sum(a_drag_mag)/len(a_drag_mag)

            a_faxen_mag = np.linalg.norm(a_faxen,axis=1)
            a_faxen_average = np.sum(a_faxen_mag)/len(a_faxen_mag)

            a_undisturbed_mag = np.linalg.norm(a_undisturbed,axis=1)
            a_undisturbed_average = np.sum(a_undisturbed_mag)/len(a_undisturbed_mag)

            a_magnus_mag = np.linalg.norm(a_magnus,axis=1)
            a_magnus_average = np.sum(a_magnus_mag)/len(a_magnus_mag)

            a_saffman_mag = np.linalg.norm(a_saffman,axis=1)
            a_saffman_average = np.sum(a_saffman_mag)/len(a_saffman_mag)

            histories_array[i[0],0] = a_drag_average
            histories_array[i[0],1] = a_faxen_average
            histories_array[i[0],2] = a_magnus_average
            histories_array[i[0],3] = a_saffman_average
            histories_array[i[0],4] = a_undisturbed_average
            i[0] += 1
            j[0] = -1
        j[0] += 1
        
        # form RHS
        dxdt = v
        dvdt = (a_drag + a_faxen + a_magnus + a_saffman)
        dwdt = get_torque(wf_interp,w,tau_r)

        return(dxdt,dvdt,dwdt)
    
    # The Associated ODE for this should be on slide 76 of the Chapter 2 notes. 
    # Since we also need to solve the rotation equation for this, an updated rk4 solver will be needed
    # We will also need to calculate certain values (vorticity,material derivative) of the flow field and particle
    # These should all be additional functions in the paticle simulation files.
    def generalBBO(t,x,v,St,CM,lambda_rho,CH,CLR,CLS):
        return (0,0)

    def fluid_tensor_interpolator(x, L, dx, Tg, ng):
        """
        Trilinear interpolation of a 3x3 tensor field Tg at points x.
        Tg has shape (Nxg, Nyg, Nzg, 3, 3) with ng ghost cells.
        x has shape (Npart, 3).
        Returns array of shape (Npart, 3, 3).
        """
        xmod = np.mod(x, L)
        idx  = np.floor(xmod / dx).astype(int)
        phi  = (xmod - idx * dx) / dx  # in [0,1)

        i = ng + idx[:, 0]
        j = ng + idx[:, 1]
        k = ng + idx[:, 2]

        wx0 = 1.0 - phi[:, 0]; wx1 = phi[:, 0]
        wy0 = 1.0 - phi[:, 1]; wy1 = phi[:, 1]
        wz0 = 1.0 - phi[:, 2]; wz1 = phi[:, 2]

        # weights for 8 corners
        w000 = (wx0 * wy0 * wz0)[:, None, None]
        w100 = (wx1 * wy0 * wz0)[:, None, None]
        w010 = (wx0 * wy1 * wz0)[:, None, None]
        w110 = (wx1 * wy1 * wz0)[:, None, None]
        w001 = (wx0 * wy0 * wz1)[:, None, None]
        w101 = (wx1 * wy0 * wz1)[:, None, None]
        w011 = (wx0 * wy1 * wz1)[:, None, None]
        w111 = (wx1 * wy1 * wz1)[:, None, None]

        Tp = (
            w000 * Tg[i    , j    , k    ] +
            w100 * Tg[i + 1, j    , k    ] +
            w010 * Tg[i    , j + 1, k    ] +
            w110 * Tg[i + 1, j + 1, k    ] +
            w001 * Tg[i    , j    , k + 1] +
            w101 * Tg[i + 1, j    , k + 1] +
            w011 * Tg[i    , j + 1, k + 1] +
            w111 * Tg[i + 1, j + 1, k + 1]
        )
        return Tp

    # # ============================ Reproduce Inertial Particles from Homework ===========================================================
    # Nparticle = 500
    # T = 2*tau_L
    # scaling_dt = 1/20
    # dt = scaling_dt*tau_eta
    # Nt = int(np.ceil(T/dt))

    # # particle simulation parameters
    # St = 1
    # Re_p = 0.1
    # rho_f = 1
    # # Physical Properties
    # R = nu * Re_p / u_rms
    # tau_p = St * tau_eta
    # lambda_rho = 2 * R**2 / (9 * nu * tau_p)
    # rho_p = rho_f / lambda_rho
    # tau_r = R**2 / (15 * nu * lambda_rho)   
    # mu = nu * rho_f
    # print("                       Radius = %.2e [m]" % R)
    # print("                        tau_r = %.2e [s]" % tau_r)
    # print("                           mu = %.2e [Pa*s]" % mu)
    # print("                   lambda_rho = %.2e [N/A]" % lambda_rho)
    # # Initial random fluid tracer locations in [0,L]^3
    # v0 = np.zeros((Nparticle,3))
    # w0 = np.zeros((Nparticle,3))
    # x0 = np.reshape(L*np.random.rand(3*Nparticle),(Nparticle,3))

    # (x, v, w) = rk4_integrator(x0,v0,w0,dt,Nt,L,lambda t,x,v,w : particle_RHS(t,x,v,w,St,Re_p,rho_f))    
    # fig = plt.figure(layout='constrained', figsize=(10, 5)); subfigs = fig.subplots(1, 2)
    # # Initialize slice paramters
    # slice_loc = 0.25*L
    # slice_thickness = 5.0*eta
    # # Print initial particle distributionbbb
    # if UseLaTeX:
    #     time_init_string = r'$t = 0$'
    #     time_final_string = r'$t = 2 \tau_L$'
    # else:
    #     time_init_string = 't = 0'
    #     time_final_string = 't = 2 τ_L'
    # plot_particles(subfigs[0], x0, slice_loc, slice_thickness,L, r'Particle distribution at %s' % time_init_string)
    # # Print final particle distribution
    # plot_particles(subfigs[1], x, slice_loc, slice_thickness,L, r'Particle distribution at %s' % time_final_string)
    # plt.close('all')

    # ========================== Force History Graph - No Collisions =======================================
    Nparticle = 5000
    T = 2*tau_L
    scaling_dt = 1/15
    dt = scaling_dt*tau_eta
    Nt = int(np.ceil(T/dt))
    plt.show()
    # particle simulation parameters
    St = 10
    Re_p = 0.1
    rho_f = 100
    # Physical Properties
    R = nu * Re_p / u_rms
    tau_p = St * tau_eta
    lambda_rho = 2 * R**2 / (9 * nu * tau_p)
    rho_p = rho_f / lambda_rho
    tau_r = R**2 / (15 * nu * lambda_rho)   
    mu = nu * rho_f
    print("                       Radius = %.2e [m]" % R)
    print("                        tau_r = %.2e [s]" % tau_r)
    print("                           mu = %.2e [Pa*s]" % mu)
    print("                   lambda_rho = %.2e [N/A]" % lambda_rho)
    # Initial random fluid tracer locations in [0,L]^3
    v0 = np.zeros((Nparticle,3))
    w0 = np.zeros((Nparticle,3))
    x0 = np.reshape(L*np.random.rand(3*Nparticle),(Nparticle,3))
    history_store = np.zeros((Nt,5))
    count = [0]
    innerCount = [0]
    (x, v, w,t,a) = rk4_integrator(x0,v0,w0,dt,Nt,L,lambda t,x,v,w : particle_RHS_with_history(t,x,v,w,St,Re_p,rho_f,history_store,count,innerCount),True)    
    print(x.shape)
    x = x[:,:,-1]
    v = v[:,:,-1]
    w = w[:,:,-1]

    plt.figure()
    plt.semilogy(t[0:Nt-1]/tau_L,history_store[0:Nt-1,0],label ="drag")
    export_to_csv(t[0:Nt-1]/tau_L,history_store[0:Nt-1,0],"NormT","Drag",work_file_path+"/csv_data/DragForceHistory.csv")
    plt.semilogy(t[0:Nt-1]/tau_L,history_store[0:Nt-1,1],label ="Faxen")
    export_to_csv(t[0:Nt-1]/tau_L,history_store[0:Nt-1,1],"NormT","Faxen",work_file_path+"/csv_data/FaxenForceHistory.csv")
    plt.semilogy(t[0:Nt-1]/tau_L,history_store[0:Nt-1,2],label ="magnus")
    export_to_csv(t[0:Nt-1]/tau_L,history_store[0:Nt-1,2],"NormT","magnus",work_file_path+"/csv_data/MagnusForceHistory.csv")
    plt.semilogy(t[0:Nt-1]/tau_L,history_store[0:Nt-1,3],label ="saffman")
    export_to_csv(t[0:Nt-1]/tau_L,history_store[0:Nt-1,3],"NormT","saffman",work_file_path+"/csv_data/SaffmanForceHistory.csv")
    plt.semilogy(t[0:Nt-1]/tau_L,history_store[0:Nt-1,4],label ="undisturbed")
    export_to_csv(t[0:Nt-1]/tau_L,history_store[0:Nt-1,4],"NormT","undisturbed",work_file_path+"/csv_data/UndisturbedForceHistory.csv")
    plt.legend()
    plt.show()

    # def particle_RHS_Loth(t,x,v,w,St):
    #     # parameter setup
    #     tau_p = St * tau_eta

    #     # interpolate quantities
    #     u_interp = fluid_velocity_interpolator(x,L,dx,ug,vg,wg,ng)
        
    #     # form RHS
    #     dxdt = v
    #     dvdt = get_stokes_drag_with_correlation(v, u_interp, tau_p, Re_p) 
    #     dwdt = np.zeros_like(w)

    #     return(dxdt,dvdt,dwdt)

    # def particle_RHS_Loth_collision(t,x,v,w,St):
    #     # parameter setup
    #     tau_p = St * tau_eta

    #     # interpolate quantities
    #     u_interp = fluid_velocity_interpolator(x,L,dx,ug,vg,wg,ng)
        
    #     # Get collision acceleration
        
    #     dxdt_C,dvdt_C,dwdt_C = deriv(t,x.T,v.T,w.T)

    #     # form RHS
    #     dxdt = v+dxdt_C.T
    #     dvdt = get_stokes_drag_with_correlation(v, u_interp, tau_p,Re_p) + dvdt_C.T
    #     dwdt = np.zeros_like(w)+dwdt_C.T

    #     return(dxdt,dvdt,dwdt)
    # # ========================== Force History Graph - Drag vs Stokes =======================================
    # St_list = np.logspace(-1,1,10)
    # diff_array = np.zeros_like(St_list)
    # diff_array_collision = np.zeros_like(St_list)

    
    
    # for i in range(len(St_list)):
    #     St = St_list[i]
    #     Nparticle = 10000
    #     T = 2*tau_L
    #     scaling_dt = 1/15
    #     dt = scaling_dt*tau_eta
    #     # particle simulation parameters
    #     Re_p = 13
    #     rho_f = 1

    #     # Physical Properties
    #     R = nu * Re_p / u_rms
    #     tau_p = St * tau_eta
    #     lambda_rho = 2 * R**2 / (9 * nu * tau_p)
    #     rho_p = rho_f / lambda_rho
    #     tau_r = R**2 / (15 * nu * lambda_rho)   
    #     mu = nu * rho_f
    #     V_p = (4/3)*np.pi*R**3
    #     m_p = rho_p * V_p

    #     t_D = 2*R/u_rms
    #     dt = min(scaling_dt*tau_eta,t_D/10)
    #     Nt = int(np.ceil(T/dt))

    #     k,eta = get_spring_damping_params(1,m_p,m_p,10*dt)
    #     mu = 0.8
    #     V_p_tot = Nparticle*V_p

    #     phi_p = V_p_tot/(L**3)
        
    #     deriv = collision_derivative(L, R, m_p, k, eta, mu, k, dt)

        
    #     # Initial random fluid tracer locations in [0,L]^3
    #     x0 = initialize_particles((0,0,0),L,2*R,phi_p,np.random.default_rng(seed=42))[0]
    #     v0 = np.zeros_like(x0)
    #     w0 = np.zeros_like(x0)
    #     print("SIMULATION INFORMATION =======================")
    #     print("                       Radius = %.2e [m]" % R)
    #     print("                        tau_r = %.2e [s]" % tau_r)
    #     print("                           dt = %.2e [s]" % dt)
    #     print("                           mu = %.2e [Pa*s]" % mu)
    #     print("                   lambda_rho = %.2e [N/A]" % lambda_rho)
    #     print("                        phi_p = %.2e [N/A]" % phi_p)
    #     print("                           Np = %.2e [N/A]" % len(x0))
    #     print("                           St = %.2e [N/A]" % St)

    #     # Collisions
    #     (x, v, w) = rk4_integrator(x0,v0,w0,dt,Nt,L,lambda t,x,v,w : particle_RHS_Loth_collision(t,x,v,w,St))  
                
    #     Sp = fluid_tensor_interpolator(x, L, dx, Sg, ng)  # shape (Npart, 3,3)
    #     Rp = fluid_tensor_interpolator(x, L, dx, Rg, ng)  # shape (Npart, 3,3)

    #     S2_p = np.einsum('...ij,...ij->...', Sp, Sp)  # length Npart
    #     R2_p = np.einsum('...ij,...ij->...', Rp, Rp)  # length Npart

    #     S2_mean = np.mean(S2_p)
    #     R2_mean = np.mean(R2_p)  

    #     diff = tau_eta**2 * (S2_mean - R2_mean)
    #     diff_array_collision[i] = diff
    #     print(f"St = {St}, Drag Difference (with collisions) = {diff_array_collision[i]}")
    #     # No Collisions
    #     (x, v, w) = rk4_integrator(x0,v0,w0,dt,Nt,L,lambda t,x,v,w : particle_RHS_Loth(t,x,v,w,St))  
                
    #     Sp = fluid_tensor_interpolator(x, L, dx, Sg, ng)  # shape (Npart, 3,3)
    #     Rp = fluid_tensor_interpolator(x, L, dx, Rg, ng)  # shape (Npart, 3,3)

    #     S2_p = np.einsum('...ij,...ij->...', Sp, Sp)  # length Npart
    #     R2_p = np.einsum('...ij,...ij->...', Rp, Rp)  # length Npart

    #     S2_mean = np.mean(S2_p)
    #     R2_mean = np.mean(R2_p)  

    #     diff = tau_eta**2 * (S2_mean - R2_mean)
    #     diff_array[i] = diff

    #     print(f"St = {St}, Drag Difference = {diff_array[i]}")
    # plt.figure()
    # # plt.scatter(St_list, diff_array, s=50, color='black')
    # plt.loglog(St_list, diff_array, "o", color='black')
    # plt.loglog(St_list, diff_array_collision, "o", color='red')

    # # plt.xscale('log')
    # plt.xlabel(r"St")
    # plt.ylabel(r"$\tau_\eta^2(\langle S^2\rangle^p - \langle R^2\rangle^p)$")

    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(work_file_path+"/Code/Results/drag_difference_vs_stokes.pdf", dpi=600, bbox_inches='tight', pad_inches=0)
    # plt.show()
    # ========================== Inertial particles in HIT without collisions =======================================
    # Stokes numbers
    # St_list = np.logspace(-1, 1, 10)
    # St_list = [1]
    # Npart = 200000
    # T = 2.0 * tau_L

    # S2_mean_array = np.zeros_like(St_list)
    # R2_mean_array = np.zeros_like(St_list)

    # for i, St in enumerate(St_list):
    #     print(f"Running simulation for St = {St}")
        
    #     # particle response time
    #     tau_p = St * tau_eta
    #     dt = (1.0 / 5.0) * min(tau_p, tau_eta)
    #     Nt = int(np.ceil(T / dt))

    #     # Initial positions: random in [0, L]^3
    #     part_x0 = np.reshape(L * np.random.rand(3 * Npart), (Npart, 3))

    #     # Initial velocities: start with fluid velocity at particle positions
    #     part_v0 = fluid_velocity_interpolator(part_x0, L, dx, ug, vg, wg, ng)

    #     # advance inertial particles
    #     part_x, part_v, part_w = rk4_integrator(part_x0, part_v0, np.zeros((Npart, 3)), dt, Nt, L, lambda t, x, v, w: particle_RHS_Loth(t, x, v, w, St))
        
    #     Sp = fluid_tensor_interpolator(part_x, L, dx, Sg, ng)  # shape (Npart, 3,3)
    #     Rp = fluid_tensor_interpolator(part_x, L, dx, Rg, ng)  # shape (Npart, 3,3)

    #     S2_p = np.einsum('...ij,...ij->...', Sp, Sp)  # length Npart
    #     R2_p = np.einsum('...ij,...ij->...', Rp, Rp)  # length Npart

    #     S2_mean = np.mean(S2_p)
    #     R2_mean = np.mean(R2_p)
    #     diff_array = tau_eta**2 * (S2_mean - R2_mean)
    #     print(f"St = {St}, Drag Difference = {diff_array}")

        # S2_mean_array[i] = S2_mean
        # R2_mean_array[i] = R2_mean
    
    # # Plot results
    # diff_array = tau_eta**2 * (S2_mean_array - R2_mean_array)
    # plt.figure()
    # # plt.scatter(St_list, diff_array, s=50, color='black')
    # plt.loglog(St_list, diff_array, "o", color='black')

    # # plt.xscale('log')
    # plt.xlabel(r"St")
    # plt.ylabel(r"$\tau_\eta^2(\langle S^2\rangle^p - \langle R^2\rangle^p)$")

    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()




# Run only here
if __name__ == '__main__':
    main()