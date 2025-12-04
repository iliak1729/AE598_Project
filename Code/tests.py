import unittest
import numpy as np
import matplotlib.pyplot as plt
from Project_Code.particle_simulation import *
import os

# To run tests from vscode, use the testing tab on the left. Set up testing to look in the Code folder.
# To run from terminal, run: python tests.py
# for all tests
# For specific tests, run: python -m unittest tests.ForceTest.test_magnus tests.ForceTest.test_saffman
# or simiilar for a specific subset
class ForceTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Important Inputs
        cls.work_file_path = os.getcwd()  
        data_file_path = cls.work_file_path + '/Code/Flow_Data/hit/ae598-mf-hw3-data.npz'
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
        cls.L                            = float(loaded_data['L']) # Domain length
        cls.nx                           = int(loaded_data['nx']) # Number of cells per direction
        cls.dx                           = cls.L/float(cls.nx) # Mesh spacing
        cls.nu                           = float(loaded_data['nu']) # Kinematic viscosity
        cls.vel                          = np.array(loaded_data['velocity'].reshape((cls.nx,cls.nx,cls.nx,3),order='F'),dtype=np.float32) # Load velocity from data file
        cls.ng                           = 3 # Number of ghost layers for interpolation and gradient calculation
        cls.velg                         = np.pad(cls.vel, ((cls.ng,cls.ng),(cls.ng,cls.ng),(cls.ng,cls.ng),(0,0)), 'wrap') # Padding domain with 'ng' periodic ghost layers
        cls.u, cls.v, cls.w                      = cls.vel[:,:,:,0],  cls.vel[:,:,:,1],  cls.vel[:,:,:,2] # Velocity components
        cls.vel_mag                      = np.sqrt(cls.u**2+cls.v**2+cls.w**2) # Velocity magnitude
        cls.ug, cls.vg, cls.wg                   = cls.velg[:,:,:,0], cls.velg[:,:,:,1], cls.velg[:,:,:,2] # Velocity components with ghost layer padding (needed for gradient)

        # Compute velocity gradient
        cls.grad_u_x, cls.grad_u_y, cls.grad_u_z = np.array(np.gradient(cls.ug, cls.dx))[:,cls.ng:-cls.ng,cls.ng:-cls.ng,cls.ng:-cls.ng]
        cls.grad_v_x, cls.grad_v_y, cls.grad_v_z = np.array(np.gradient(cls.vg, cls.dx))[:,cls.ng:-cls.ng,cls.ng:-cls.ng,cls.ng:-cls.ng]
        cls.grad_w_x, cls.grad_w_y, cls.grad_w_z = np.array(np.gradient(cls.wg, cls.dx))[:,cls.ng:-cls.ng,cls.ng:-cls.ng,cls.ng:-cls.ng]
        # Compute material derivative - assuming no d/dt
        cls.mat_der_u = cls.u*cls.grad_u_x + cls.v*cls.grad_u_y + cls.w*cls.grad_u_z
        cls.mat_der_v = cls.u*cls.grad_v_x + cls.v*cls.grad_v_y + cls.w*cls.grad_v_z
        cls.mat_der_w = cls.u*cls.grad_w_x + cls.v*cls.grad_w_y + cls.w*cls.grad_w_z
        cls.mat_der_vel  = np.stack((cls.mat_der_u, cls.mat_der_v, cls.mat_der_w), -1)
        # Laplacians
        cls.lap_u = laplacian_scalar_field(cls.ug,cls.dx)
        cls.lap_v = laplacian_scalar_field(cls.vg,cls.dx)
        cls.lap_w = laplacian_scalar_field(cls.wg,cls.dx)
        cls.lap_vel  = np.stack((cls.lap_u, cls.lap_v, cls.lap_w), -1)
        # Compute fluid vorticity
        cls.omega_f = np.zeros((cls.nx, cls.nx, cls.nx, 3))
        cls.omega_f[:, :, :, 0] = cls.grad_w_y - cls.grad_v_z
        cls.omega_f[:, :, :, 1] = cls.grad_u_z - cls.grad_w_x
        cls.omega_f[:, :, :, 2] = cls.grad_v_x - cls.grad_u_y
        cls.omega_fg = np.pad(cls.omega_f, ((cls.ng,cls.ng),(cls.ng,cls.ng),(cls.ng,cls.ng),(0,0)), 'wrap')

        # Compute turbulent kinetic energy and dissipation rate
        cls.tke                          = 0.5*np.mean(cls.u**2+cls.v**2+cls.w**2)
        cls.eps                          = 2.0*cls.nu*np.mean(cls.grad_u_x**2+cls.grad_v_y**2+cls.grad_w_z**2+0.25*(cls.grad_u_y+cls.grad_v_x)**2+0.25*(cls.grad_u_z+cls.grad_w_x)**2+0.25*(cls.grad_v_z+cls.grad_w_y)**2)
        # Compute Taylor-scale Reynolds number and flow length- and time-scales
        cls.u_rms                        = np.sqrt(2.0*cls.tke/3.0) # RMS turbulent velocity [m/s]
        cls.Taylor_lambda                = np.sqrt(15.0*cls.nu*cls.u_rms**2/cls.eps)  # Taylor microscale [m]
        cls.Re_lambda                    = cls.u_rms*cls.Taylor_lambda/cls.nu # Taylor-scale Reynolds number [-]
        cls.eta                          = (cls.nu**3/cls.eps)**0.25 # Kolmogorov length scale [m]
        cls.tau_eta                      = (cls.nu/cls.eps)**0.5 # Kolmogorov time scale [s]
        cls.u_eta                        = (cls.nu*cls.eps)**0.25 # Kolmogorov velocity scale [m/s]
        cls.L_integral                   = cls.u_rms**3/cls.eps # Integral length scale [m]
        cls.tau_L                        = cls.L_integral/cls.u_rms # Large-eddy turnover time [s]

        cls.g = 9.81 # gravitational acceleration


        
    def test_magnus(self):
        print("Running Magnus")
        self.assertEqual(1+1,2)

    def test_saffman(self):
        print("Running Saff")
        self.assertTrue(5 > 1)

    def test_head_on_collision(self):
        v = 10
        rho = 8000
        E = 200e9
        alpha = 0.04
        R = 10e-3

        V = 4*np.pi*R**3/3
        m = V*rho

        mu = 0 # Tangential component does not matter
        kn = 0.5*alpha*np.pi*E*R
        Ccr = 2*np.sqrt(m*kn/2)

        
        # Initial Conditions in get_contact_force form
        # Position
        x1 = np.array([2*R,0,0])
        x2 = np.array([4*R,0,0])
        x = np.stack((x1,x2),-1)
        # Velocity
        v1 = np.array([+v,0,0])
        v2 = np.array([-v,0,0])
        v = np.stack((v1,v2),-1)
        # Omega
        omega1 = omega2 = np.array([0,0,0])
        w = np.stack((omega1,omega2),-1)
        # Mass
        m1 = m2 = m
        # Radii
        r1 = r2 = R
        # Model Paramters
        kt = kn
        global FtOld
        FtOld = np.array([0,0,0])
        dt = 1e-7
        Tmax = 50e-6
        Nt = round(Tmax/dt)
        # Run integration and get values
        def collision_derivative(t,xi,vi,wi,e):
            zeta = -np.log(e)/(np.sqrt(np.pi**2 + (np.log(e))**2)) # Changes with chaning e
            eta = Ccr*zeta
            global FtOld
            x1 = xi[:,0]
            x2 = xi[:,1]
            v1 = vi[:,0]
            v2 = vi[:,1]
            omega1 = wi[:,0]
            omega2 = wi[:,1]
            
            a1,a2,alpha1,alpha2,FtOld = get_contact_force(x1,v1,omega1,m1,r1,
                                                          x2,v2,omega2,m2,r2,
                                                          kn,eta,mu,kt,FtOld,dt)

            dxdt = vi
            dvdt = np.stack((a1,a2),-1)
            dwdt = np.stack((alpha1,alpha2),-1)
            
            return(dxdt,dvdt,dwdt)
        eSet = [0.1,0.5,0.8,1.0]
        plt.figure()
        for i in range(len(eSet)):
            xp,vp,wp,tp = rk4_integrator(x,v,w,dt,Nt,self.L,lambda tl,xl,vl,wl : collision_derivative(tl,xl,vl,wl,eSet[i]),True)
            # Right now I need to make FtOld global. This makes me unhappy. Is there a better way to do this?
            plt.plot(tp,vp[0,0,:],label=fr'COR = {eSet[i]:.1f}')
        plt.legend()
        plt.savefig(self.work_file_path +"/Code/Results/collision_test_velocty.png", dpi=300)
        
        
        self.assertAlmostEqual(kn,1.26e8,-6) # kn value given in book
        



if __name__ == "__main__":
    unittest.main()

