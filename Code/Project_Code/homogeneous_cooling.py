import numpy as np
import matplotlib.pyplot as plt
import math
from particle_simulation import *
from collections import defaultdict

# collision derivatives 
def collision_derivative(L, r, m, k, eta, mu, kt, dt):
    """    
    :param L: Domain length
    :param r: particle radius
    :param m: particle mass
    :param k: spring stiffness
    :param eta: damping coefficient
    :param mu: friction coefficient
    :param kt: tangential stiffness
    :param dt: time step (for FtTrial update)
    """

    Ft_dict = {}  # To store tangential forces between particle pairs

    # particle based mesh
    cell_size = 2.0 * r
    n_cells = int(L / cell_size)

    def derivative(t, x, v, w):
        nonlocal Ft_dict

        N = x.shape[1]          # number of particles
        dxdt = v.copy()         # dx/dt = v
        dvdt = np.zeros_like(v) # accleration
        dwdt = np.zeros_like(w) # angular acceleration

        # cell linked list
        cell_list = defaultdict(list)
        for i in range(N):
            xi = x[:, i] # RK4 integrator already wraps position
            idx = np.floor(xi / cell_size).astype(int)
            idx = np.clip(idx, 0, n_cells - 1)
            cell_idx = (idx[0], idx[1], idx[2])
            cell_list[cell_idx].append(i) # add particle i to cell i,j,k

        # cell id
        def cell_id(c):
            return c[0] + n_cells * (c[1] + n_cells * c[2])
        
        # looping over cells and neighbors
        for cell, p_indices in cell_list.items():
            cx, cy, cz = cell
            cid = cell_id(cell)

            # looping over stencil
            for ii in (-1, 0, 1):
                for jj in (-1, 0, 1):
                    for kk in (-1, 0, 1):
                        # neighor cell indices
                        nb = ( (cx + ii) % n_cells,
                               (cy + jj) % n_cells,
                               (cz + kk) % n_cells )
                        nb_id = cell_id(nb)

                        # to avoid double counting only do >= id
                        if nb_id < cid:
                            continue

                        # particle indices in a neighbor cell
                        nb_p_indices = cell_list.get(nb, [])
                        if not nb_p_indices:
                            continue

                        # looping over particle pairs
                        for i in p_indices:         # target cell particles
                            for j in nb_p_indices:  # neighbor cell particles
                                if nb_id == cid and j <= i:
                                    # for same cell only do j>i
                                    continue

                                # particle pair
                                xi = x[:, i]
                                xj = x[:, j]

                                # periodic min dist
                                dx_vec = min_dx_periodic(xi, xj, L)
                                dist = np.linalg.norm(dx_vec)
                                if dist >= 2.0 * r or dist == 0.0:
                                    # no contact
                                    pair_key = (i, j) if i < j else (j, i)
                                    if pair_key in Ft_dict:
                                        Ft_dict[pair_key] = np.zeros(3)
                                    continue

                                # Ftold
                                pair_key = (i, j) if i < j else (j, i)
                                Ft_old = Ft_dict.get(pair_key, np.zeros(3))

                                # linear and angular velocity
                                vi = v[:, i]
                                vj = v[:, j]
                                wi = w[:, i]
                                wj = w[:, j]

                                # contact force accleration
                                x1_rel = np.zeros(3)
                                x2_rel = dx_vec
                                ai, aj, alpha_i, alpha_j, Ft_new = get_contact_force(
                                    x1_rel, vi, wi, m, r,
                                    x2_rel, vj, wj, m, r,
                                    k, eta, mu, kt, Ft_old, dt)
                                
                                # update accelerations
                                dvdt[:, i] += ai
                                dvdt[:, j] += aj
                                dwdt[:, i] += alpha_i
                                dwdt[:, j] += alpha_j

                                # update Ft_dict
                                Ft_dict[pair_key] = Ft_new
        return dxdt, dvdt, dwdt
    return derivative


# running homoegeneous cooling simulation

def run_homogeneous_cooling():
    dp = 1.0e-3     # particle diameter
    r = 0.5 * dp
    rho_p = 2400.0  # particle density
    phi_v = 0.2     # volume fraction
    L = 0.02        # domain length
    e = 0.6         # restitution coefficient
    mu_f = 0.1      # friction coefficient

    # number of particles
    Vp = (4.0 / 3.0) * math.pi * r**3
    m = rho_p * Vp
    N_est = phi_v * L**3 / Vp
    N = int(N_est)
    print(f"Number of particles: {N}")

    # initializing particle positions
    origin = (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed=42)
    x0, r_vec = initialize_particles(origin, L, dp, phi_v, rng)
    x0 = x0.T  # shape (3, N)
    r = r_vec[0]
    print(f"Initialized {x0.shape[1]} particles.")
    print(x0.shape)
    
    # plotting initial configurations
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x0[0,:], x0[1,:], x0[2,:], s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Initial Particle Configuration')
    plt.show()

    # initializing velocities and angular velocities
    rng = np.random.default_rng(seed=24)
    u0_max = 0.1
    v0 = rng.uniform(-u0_max, u0_max, size=(3, N))
    w0 = np.zeros((3, N))

    # time stepping parameters
    dt = 1e-4
    t_col = 20.0 * dt
    T_end = 0.5
    Nt = int(T_end / dt)

    # spring and damping constants
    kn, eta_n = get_spring_damping_params(e, m, m, t_col)
    kt = kn

    # collision derivatives
    deriv = collision_derivative(L, r, m, kn, eta_n, mu_f, kt, dt)

    # rk4 integration
    x_hist, v_hist, w_hist, t_hist = rk4_integrator(x0, v0, w0, dt, Nt, L, deriv, True)

    # computing kinetic energy history
    KE = np.zeros(Nt)
    KE_trans = np.zeros(Nt)
    KE_rot = np.zeros(Nt)

    I = 0.4 * m * r**2
    
    for k in range(Nt):
        v_k = v_hist[:, :, k]
        w_k = w_hist[:, :, k]
        v2 = np.sum(v_k**2, axis=0)
        w2 = np.sum(w_k**2, axis=0)
        KE_trans[k] = 0.5 * m * np.sum(v2)
        KE_rot[k] = 0.5 * I * np.sum(w2)
        KE[k] = KE_trans[k] + KE_rot[k]
        # normalize by initial KE
        # KE[k] /= KE[0]
        # KE_trans[k] /= KE_trans[0]
        # KE_rot[k] /= KE_rot[0]
    
    # normalization
    KE /= KE[0]
    KE_trans /= KE_trans[0]
    
    # plotting kinetic energy decay
    plt.figure()
    plt.plot(t_hist, KE)
    plt.xlabel('Time')
    plt.ylabel('Kinetic Energy')
    plt.grid(True)
    plt.show()

    # plotting translational and rotational kinetic energy decay
    plt.figure()
    plt.plot(t_hist, KE_trans, label='Translational KE')
    plt.xlabel('Time')
    plt.ylabel('Kinetic Energy')
    plt.grid(True)
    plt.show()

    # plotting particle positions at final time
    x_final = x_hist[:, :, -1]
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x_final[0,:], x_final[1,:], x_final[2,:], s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Final Particle Configuration')
    plt.show()

if __name__ == "__main__":
    run_homogeneous_cooling()


