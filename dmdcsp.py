import numpy as np
import scipy as sp
from scipy import signal
from scipy.signal import StateSpace
import time
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=24)
from matplotlib import animation
from matplotlib import patches
import data_loader
import cvxpy as cp
import copy
import control


class stateSpace(object):

    """
    Custom class for defining a simple discrete-time state-space model of the form:
        x(k+1) = A x(k) + B u(k)
          y(k) = C x(k)

    - Initialization: 
        Simply give matrices A, B, C

    """

    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.ny = C.shape[0]


    def lsim(self, x0, u):
        """
        lsim(x0, u):
            Simulates the state and output for a given sequence of inputs u = [u(0), u(1), ...] and initial condition x(0) = x0.

            Input: 
                x0: initial condition
                u: control input [u(0), u(1), ..., u(N)]

            Output:
                x: state [x(0), x(1), ..., x(N+1)]
                y: output [y(0), y(1), ..., y(N+1)]
        """
        p = u.shape[1]

        if u.shape[0] != self.nu:
            raise ValueError("u.shape[0] should be equal to the number of inputs")
        if x0.shape[0] != self.nx:
            raise ValueError("The size of x0 should be equal to the number of states nx")

        x = np.zeros((self.nx, p), dtype=complex)
        y = np.zeros((self.ny, p))
        x[:,0] = x0
        y[:,0] = np.real(self.C @ x0)
        
        for k in range(p-1):
            x[:,k+1] = self.A @ x[:,k] + self.B @ u[:,k]
            y[:,k+1] = np.real(self.C @ x[:,k+1])

        return x, y


    def error(self, Y0, Y1, U0):
        """
        error(Y, U):
            One-step forward reconstruction error for the output:

                          ||Y_actual - Y_simulated||_F
                error  =  ----------------------------
                                 ||Y_actual||_F
        """
        Cp = np.linalg.pinv(self.C)

        error = np.linalg.norm((Y1 - self.C @ (self.A @ (Cp @ Y0) + self.B @ U0)), ord='fro')/ \
                  np.linalg.norm(Y1, ord='fro')*100
        print('    Model error = ', error, '%')



class dmdcsp(object):
    """
    Sparsity-Promoting Dynamic Mode Decomposition with Control Class

    Initialization: Simply provide data matrices Y0, Y1, U0

    """

    def __init__(self, Y0, Y1, U0, q=10, u_nominal=1, dt=1):

        if Y0.shape[1] != U0.shape[1]:
            raise ValueError("The number of snapshot pairs Y0, U0 should be equal to the number of training inputs.")

        if Y1.shape[1] != U0.shape[1]:
            raise ValueError("The number of snapshot pairs Y1, U0 should be equal to the number of training inputs.")

        # Parameters
        self.ny = Y0.shape[0]   # Number of outputs
        self.nu = U0.shape[0]   # Number of inputs
        self.q = q              # Number of POD modes kept
        self.p = Y0.shape[1]    # Number of training snapshots available
        self.dt = dt            # Time step

        # Save snapshots
        self.Y0 = Y0
        self.Y1 = Y1
        self.U0 = U0

        # Compute POD Modes
        self.Uq = self.compute_POD_basis(Y0, self.q)

        # Project snapshots
        H0 = self.Uq.conj().T @ Y0
        H1 = self.Uq.conj().T @ Y1

        # POD projection error
        pod_error = np.linalg.norm((Y0 - self.Uq @ H0), ord='fro')/ \
                    np.linalg.norm(Y0, ord='fro')*100
        print('    POD projection error = ', pod_error, '%')

        # Run DMDc
        self.F, self.G = self.DMDc(H0, H1, U0)
        
        # Eigendecomposition of A
        lamb, self.W = np.linalg.eig(self.F)
        self.Lambda = np.diag(lamb)
        self.Gamma = np.linalg.inv(self.W) @ self.G

        # DMD modes
        self.Phi = self.Uq @ self.W

        # Setup matrices for sparsity-promoting optimization
       #self.R = np.zeros((self.q, self.p), dtype=np.complex)
        R = self.Lambda @ (np.linalg.pinv(self.Phi) @ self.Y0) + self.Gamma @ U0
        L = self.Phi
        #Y1 = self.Y[:,1:]

        # Now cast into quadratic form
        self.P = np.multiply(L.conj().T @ L, (R @ R.conj().T).conj())
        self.d = np.diag(R @ Y1.conj().T @ L).conj()
        self.s = np.trace(Y1.conj().T @ Y1)
        print('s.shape = ', self.s.shape)

        self.sys_pod = stateSpace(self.F, self.G, self.Uq)
        self.sys_dmd = stateSpace(self.Lambda, self.Gamma, self.Phi)

        print('DMD model error:')
        self.sys_dmd.error(Y0, Y1, U0)



    def DMDc(self, H0, H1, U0):
        """
        Inputs: 
            H0 : first snapshot matrix
            H1 : second snapshot matrix (after one time step)
            U0 : corresponding input

        Outputs:
            A, B : state and output matrices fitted to the reduced-order data

        """

        q = H0.shape[0]
        nu = U0.shape[0]
        p  = U0.shape[1]

        U, Sig, VT = np.linalg.svd(np.vstack((H0, U0)), full_matrices=False)
        thres = 1.0e-10
        rtil = np.min((np.sum(np.diag(Sig) > thres), q))
        print('    rtil = ', rtil)
        Util = U[:,:rtil]
        Sigtil = np.diag(Sig)[:rtil,:rtil]
        Vtil = VT.T[:,:rtil]

        U_F = Util[:q,:]
        U_G = Util[q:q+nu,:]

        F = H1 @ Vtil @ np.linalg.inv(Sigtil) @ U_F.conj().T
        G = H1 @ Vtil @ np.linalg.inv(Sigtil) @ U_G.conj().T

        dmdc_error = np.linalg.norm((H1 - (F @ H0 + G @ U0)), ord='fro')/ \
                  np.linalg.norm(H1, ord='fro')*100
        print('    DMDc error = ', dmdc_error, '%')

        print('    Maximum eigenvalue: ', np.max(np.abs(np.linalg.eig(F)[0])))

        return F, G



    def sparse(self, gamma, niter):

        zero_thres = 1.e-6

        print("gamma = %e | number of modes =         " % (gamma))

        # Define and solve the sparsity-promoting optimization problem
        # Weighted L1 norm is updated iteratively
        alpha = cp.Variable(self.q)
        weights = np.ones(self.q)
        for i in range(niter):
            objective_sparse = cp.Minimize(cp.quad_form(alpha, self.P) 
                                         - 2.*cp.real(self.d.conj().T @ alpha)
                                         + self.s
                                         + gamma * cp.pnorm(np.diag(weights)*alpha, p=1))
            prob_sparse = cp.Problem(objective_sparse)
            sol_sparse = prob_sparse.solve(verbose=False, solver=cp.SCS)
        
            alpha_sp = alpha.value # Sparse solution
            if alpha_sp is None:
                alpha_sp = np.ones(self.q)

            # Update weights
            weights = 1.0/(np.abs(alpha_sp) + np.finfo(float).eps)

            nonzero = np.abs(alpha_sp) > zero_thres # Nonzero modes
            print("                               %d of %d" % (np.sum(nonzero), self.q))

        J_sp = np.real(alpha_sp.conj().T @ self.P @ alpha_sp - 2*np.real(self.d.conj().T @ alpha_sp) + self.s)  # Square error
        nx = np.sum(nonzero)    # Number of nonzero modes - order of the sparse/reduced model

        Ez = np.eye(self.q)[:,~nonzero]

        # Define and solve the refinement optimization problem
       #alpha = cp.Variable(self.q)
        objective_refine = cp.Minimize(cp.quad_form(alpha, self.P) 
                                     - 2.*cp.real(self.d.conj().T @ alpha)
                                     + self.s)
        if np.sum(~nonzero):
            constraint_refine = [Ez.T @ alpha == 0]
            prob_refine = cp.Problem(objective_refine, constraint_refine)
        else:
            prob_refine = cp.Problem(objective_refine)

        sol_refine = prob_refine.solve()

        alpha_ref = alpha.value
        J_ref = np.real(alpha_ref.conj().T @ self.P @ alpha_ref - 2*np.real(self.d.conj().T @ alpha_ref) + self.s)  # Square error

        P_loss = 100*np.sqrt(J_ref/self.s)

        # Truncate modes
        E = np.eye(self.q)[:,nonzero]
        Lambda_bar = E.T @ self.Lambda @ E
        Beta_bar = E.T @ self.Gamma
        Phi_bar = self.Phi @ np.diag(alpha_ref) @ E

        # Save data
        stats = {}
        stats["nx"]     = nx
        stats["alpha_sp"]   = alpha_sp
        stats["alpha_ref"]  = alpha_ref
        stats["z_0"]    = (np.linalg.pinv(self.Phi) @ self.Y0[:,0])[nonzero]
        stats["E"]      = E
        stats["J_sp"]   = J_sp
        stats["J_ref"]  = J_ref
        stats["P_loss"] = P_loss

        if nx != 0:
            print("Rank of controllability matrix: %d of %d" % (np.linalg.matrix_rank(control.ctrb(Lambda_bar, Beta_bar)), nx))

        # Convert system from complex modal form to real block-modal form
        lambda_bar = np.diag(Lambda_bar)
        A = np.zeros((nx, nx))
        B = np.zeros((nx, self.nu))
        Theta = np.zeros((self.ny, nx))

        i = 0
        while i < nx:
            if np.isreal(lambda_bar[i]):
                A[i,i] = lambda_bar[i]
                B[i] = Beta_bar[i]
                Theta[:,i] = Phi_bar[:,i]
                i += 1
            elif i == nx-1:
                # In case only one of the conjugate pairs is truncated - this rarely happens
                A[i,i] = np.real(lambda_bar[i])
                B[i] = np.real(Beta_bar[i])
                Theta[:,i] = np.real(Phi_bar[:,i])
                i += 1
            elif np.isreal(np.isreal(lambda_bar[i] + lambda_bar[i+1])):
                A[i:i+2,i:i+2] = np.array([[np.real(lambda_bar[i]), np.imag(lambda_bar[i])],
                                           [-np.imag(lambda_bar[i]), np.real(lambda_bar[i])]])
                B[i] = np.real(Beta_bar[i])
                B[i+1] = -np.imag(Beta_bar[i])
                Theta[:,i] = 2*np.real(Phi_bar[:,i])
                Theta[:,i+1] = 2*np.imag(Phi_bar[:,i])
                i += 2
            else:
                raise ValueError("Eigenvalues are not grouped in conjugate pairs")

       #return stateSpace(Lambda_bar, Beta_bar, Phi_bar), lambda_bar, stats
        return stateSpace(A, B, Theta), lambda_bar, stats



    def sparse_batch(self, gamma, niter):
        num = gamma.shape[0]

        self.rsys = num*[None]
        self.sys_eig = num*[None]

        stats = {}
        stats["alpha_sp"]   = num*[None]
        stats["alpha_ref"]  = num*[None]
        stats["z_0"]    = num*[None]
        stats["E"]      = num*[None]
        stats['nx']     = np.zeros(num)
        stats["J_sp"]   = np.zeros(num)
        stats["J_ref"]  = np.zeros(num)
        stats["P_loss"] = np.zeros(num)

        for i in range(num):
            print('Model # %d' % i)
            self.rsys[i], self.sys_eig[i], stats_tmp = self.sparse(gamma[i], niter)

            # Save stats
            stats["alpha_sp"][i]   = stats_tmp["alpha_sp"]  
            stats["alpha_ref"][i]  = stats_tmp["alpha_ref"] 
            stats["z_0"][i]    = stats_tmp["z_0"] 
            stats["E"][i]      = stats_tmp["E"]     
            stats['nx'][i]     = stats_tmp['nx']    
            stats["J_sp"][i]   = stats_tmp["J_sp"]  
            stats["J_ref"][i]  = stats_tmp["J_ref"] 
            stats["P_loss"][i] = stats_tmp["P_loss"]

        self.sp_stats = stats
        return stats



    def compute_noise_cov(self, sys_i, sens):

        # Measurement output matrix
        C = self.rsys[sys_i].C[sens,:]

        ThetaInv = np.linalg.pinv(self.rsys[sys_i].C)

        Qe = np.cov(ThetaInv @ self.Y1 
                  - self.rsys[sys_i].A @ (ThetaInv @ self.Y0 
                  - self.rsys[sys_i].B @ self.U0))
        Re = np.cov(self.Y1[sens] - C @ (ThetaInv @ self.Y1))

        return C, Qe, Re



    def compute_POD_basis(self, Y, q):
        return np.linalg.svd(Y, full_matrices=False)[0][:,:q]


    """
    Plot functions
    """
    def plot_dmd_modes(self, grid, E=None):

        if E is None:
            E = np.eye(self.q)

        nlevels = 41
        wymin = -0.1
        wymax = 0.1

        nx = E.shape[1]

        X = grid.X()
        Z = grid.Z()
        xmax = np.max(X)
        
        def Phi(k):
            return np.real(self.Phi @ E)[:,k].reshape((grid.npx, grid.npz))

        fig, axs = plt.subplots(1, figsize=(10,5), facecolor='w', edgecolor='k')
        cont = axs.contourf(X, Z, Phi(0), nlevels, cmap='coolwarm', vmin=wymin, vmax=wymax)
        cbar = fig.colorbar(cont, ax=axs, orientation='vertical')
        cbar.set_ticks(np.linspace(wymin, wymax, num=6, endpoint=True))
        axs.set_title('$Phi_0$')
        axs.set_xlim([1,xmax])

        # Flat plate patch
        delx = 5.0/768.0
        delz = 2.0/384.0
        xc = 249*delx
        zc = 192*delz
        alpha = 20.0*np.pi/180.0
        DL = 80*delx
        DT = 6*delz
        flat_plate = patches.Rectangle((xc - DL*np.cos(alpha)/2. - DT*np.sin(alpha)/2.,
                                        zc + DL*np.sin(alpha)/2. - DT*np.cos(alpha)/2.),
                                        DL,
                                        DT,
                                        angle=-(alpha*180.0/np.pi),
                                        linewidth=1, edgecolor='black', facecolor='black')
        axs.add_patch(flat_plate)

        def animate(k):
            axs.clear()
            cont = axs.contourf(X, Z, Phi(k), nlevels, cmap='coolwarm', vmin=wymin, vmax=wymax)
            axs.set_xlabel('$x$')
            axs.set_ylabel('$z$')
            axs.set_title('$Phi_{%d}$'%k)
            axs.set_aspect('equal', 'box')
            axs.add_patch(flat_plate)
            axs.set_xlim([1,xmax])
            
            return cont

        anim = animation.FuncAnimation(fig, animate, frames = range(0,nx,1), interval=500)

        return anim



    def plot_model_response(self, sys, grid):

        x0 = np.linalg.pinv(sys.C) @ self.Y0[:,0]
        xdmd, ydmd = sys.lsim(x0, self.U0)

        nlevels = 41
        wymin = -10
        wymax = 10

        X = grid.X()
        Z = grid.Z()
        xmax = np.max(X)
        
        def WY(k):
            return self.Y0[:,k].reshape((grid.npx, grid.npz))

        def WY_dmd(k):
            return ydmd[:,k].reshape((grid.npx, grid.npz))

        fig, axs = plt.subplots(2, figsize=(10,8), facecolor='w', edgecolor='k')
        cont = axs[0].contourf(X, Z, WY(0), nlevels, cmap='coolwarm', vmin=wymin, vmax=wymax)
        cont = axs[1].contourf(X, Z, WY_dmd(0), nlevels, cmap='coolwarm', vmin=wymin, vmax=wymax)
       #plt.clim(wymin, wymax)
        cbar = fig.colorbar(cont, ax=axs, orientation='vertical')
       #cbar.ax.set_autoscale_on(True)
        cbar.set_ticks(np.linspace(wymin, wymax, num=6, endpoint=True))

        # Flat plate patch
        delx = 5.0/768.0
        delz = 2.0/384.0
        xc = 249*delx
        zc = 192*delz
        alpha = 20.0*np.pi/180.0
        DL = 80*delx
        DT = 6*delz
        flat_plate = patches.Rectangle((xc - DL*np.cos(alpha)/2. - DT*np.sin(alpha)/2.,
                                        zc + DL*np.sin(alpha)/2. - DT*np.cos(alpha)/2.),
                                        DL,
                                        DT,
                                        angle=-(alpha*180.0/np.pi),
                                        linewidth=1, edgecolor='black', facecolor='black')
        flat_plate_2 = patches.Rectangle((xc - DL*np.cos(alpha)/2. - DT*np.sin(alpha)/2.,
                                          zc + DL*np.sin(alpha)/2. - DT*np.cos(alpha)/2.),
                                          DL,
                                          DT,
                                          angle=-(alpha*180.0/np.pi),
                                          linewidth=1, edgecolor='black', facecolor='black')
        axs[0].add_patch(flat_plate)
        axs[1].add_patch(flat_plate_2)

        axs[0].set_xlim([1,xmax])
        axs[1].set_xlim([1,xmax])
        axs[0].set_title('Time step $k = %d$' % (0))
       #cbar.ax.locator_params(nbins=6)

        def animate(k):

            # Clear axes
            axs[0].clear()
            axs[1].clear()

            # Training Data
            cont = axs[0].contourf(X, Z, WY(k), nlevels, cmap='coolwarm', vmin=wymin, vmax=wymax)
            axs[0].set_xlabel('$x$')
            axs[0].set_ylabel('$z$')
            axs[0].set_aspect('equal', 'box')
            axs[0].add_patch(flat_plate)
            axs[0].set_xlim([1,xmax])
            axs[0].set_title('Time step $k = %d$' % (k))
            
            # Simulation Results
            cont = axs[1].contourf(X, Z, WY_dmd(k), nlevels, cmap='coolwarm', vmin=wymin, vmax=wymax)
            axs[1].set_xlabel('$x$')
            axs[1].set_ylabel('$z$')
            axs[1].set_aspect('equal', 'box')
            axs[1].add_patch(flat_plate_2)
            axs[1].set_xlim([1,xmax])

            return cont

        anim = animation.FuncAnimation(fig, animate, frames = range(0,self.p,4), interval=200)

        return anim


