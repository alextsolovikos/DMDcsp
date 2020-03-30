import numpy as np
from scipy import signal
from scipy.signal import StateSpace
import time
import snapshots


class dmdc(object):
    """Dynamic Mode Decomposition with Control class"""

    def __init__(self, Y0, Y1, U0, nx=10, y_mean=None, u_nominal=1, dt=1):

        if not np.array_equal(Y0.shape, Y1.shape):
            raise ValueError("Snapshot matrices Y0 and Y1 should have the same shape.")
        if Y0.shape[0] != U0.shape[0]:
            raise ValueError("The number of snapshot pairs should be equal to the number of training inputs.")

        if y_mean is None:
            raise ValueError("The mean snapshot y_mean is not defined")

        # Parameters
        self.ny = Y0.shape[1]   # Number of outputs
        self.nu = U0.shape[1]    # Number of inputs
        self.nx = nx            # Number of reduced-order states
        self.p = Y0.shape[0]    # Number of training snapshots
        self.dt = dt            # Time step
        self.npoints = self.ny//3

        if y_mean is None:
            self.y_mean = np.zeros((1,self.ny))
        else:
            self.y_mean = y_mean.reshape(1,self.ny)

        # CHECK IF ZERO IS A GOOD CHOICE
        self.x_mean = np.zeros((1,nx))

        # Training Snapshots
        Y0 = (Y0.copy() - self.y_mean).T
        Y1 = (Y1.copy() - self.y_mean).T
        U0 = U0.T.copy()/u_nominal
        self.u_nominal = u_nominal

        # POD Modes
        self.Uhat = self.compute_POD_basis(Y0, nx)
        self.C = self.Uhat

        # Projected snapshots
        X0 = self.Uhat.conj().T @ Y0
        X1 = self.Uhat.conj().T @ Y1

        # Order reduction error
        pod_error = np.linalg.norm((Y0 - self.Uhat @ X0), ord='fro')/ \
                    np.linalg.norm(Y0, ord='fro')*100
        print('    POD error = ', pod_error, '%')

        # Run DMDc
        self.A, self.B = self.DMDc(X0, X1, U0)

        # State Space Model
        self.sys = StateSpace(self.A, self.B, self.Uhat, np.zeros((self.ny,self.nu)), dt=dt)


    def DMDc(self, X0, X1, U0):
        """
        Inputs: 
            X0 : first snapshot matrix
            X1 : second snapshot matrix (after one time step)
            U0 : corresponding input

        Outputs:
            rsys : state space representation of the reduced-order linear system

        """

        nx = X0.shape[0]
        nu = U0.shape[0]
        p = U0.shape[1]

        U, Sig, VT = np.linalg.svd(np.vstack((X0, U0)), full_matrices=False)
        thres = 1.0e-10
        rtil = np.sum(np.diag(Sig) > thres)
        print('    rtil = ', rtil)
        Util = U[:,:rtil]
        Sigtil = np.diag(Sig)[:rtil,:rtil]
        Vtil = VT.T[:,:rtil]

        U_A = Util[:nx,:]
        U_B = Util[nx:nx+nu,:]

        A = X1 @ Vtil @ np.linalg.inv(Sigtil) @ U_A.conj().T
        B = X1 @ Vtil @ np.linalg.inv(Sigtil) @ U_B.conj().T

        dmdc_error = np.linalg.norm((X1 - (A @ X0 + B @ U0)), ord='fro')/ \
                  np.linalg.norm(X1, ord='fro')*100
        print('    DMDc error = ', dmdc_error, '%')

        print('    Maximum eigenvalue: ', np.max(np.abs(np.linalg.eig(A)[0])))

        return A, B

    def compute_POD_basis(self, Y, nx):

        return np.linalg.svd(Y, full_matrices=False)[0][:,:nx]

    def POD_modes(self):

        return self.Uhat

    def output(self, u, t=None, x0=None, y0=None):

        if x0 is None and y0 is not None:
            x0 = self.Uhat.conj().T @ (y0.flatten() - self.y_mean.flatten())

        t, y, x = self.sys.output(u, t, x0=x0)

        return t, y + self.y_mean, x

    def propagate(self, u, x0):

        x1 = self.A @ x0 + self.B @ u
        y1 = self.C @ x1

        return y1, x1

    def relative_errors(self, u, y_exact, y0=None):

        t, y, x = self.output(u, y0=y0)
        
        nsteps = len(t)

        e_total = np.linalg.norm(y_exact - y, ord='fro')/    \
                  np.linalg.norm(y_exact - self.y_mean, ord='fro')

        print('\nDNS vs DMDc overall error for N = %d time steps: %f %%' % (nsteps, 100*e_total))

        return e_total

    def least_squares_error(self, Y0, Y1, U0):
        X0 = self.Uhat.conj().T @ Y0.T
        X1 = self.Uhat.conj().T @ Y1.T
        e_total = np.linalg.norm((X1 - (self.A @ X0 + self.B @ U0.T)), ord='fro')/ \
                  np.linalg.norm(X1, ord='fro')*100
        print('\nDNS vs DMDc least squares error: %f %%' % (e_total))
        return e_total



#class dmdc(object):
#
#    def __init__(self, snap, r, rtil, y_mean = None, Uhat=None, u_nominal=1.0):
#
#        # Number of training cases
#        self.n_cases = len(snap)
#
#        self.Y = [self.Y]
#        if y_mean is not None:
#            self.y_mean = y_mean
#
#        self.Y0 = snap[0].Y0
#        self.Y1 = snap[0].Y1
#        self.U0 = snap[0].U0
#
#        for i in range(1,self.n_cases):
#            self.Y0 = np.append(self.Y0, snap[i].Y0, axis=1)
#            self.Y1 = np.append(self.Y1, snap[i].Y1, axis=1)
#            self.U0 = np.append(self.U0, snap[i].U0, axis=1)
#
#        self.n = self.Y0.shape[0]
#        self.p = self.Y0.shape[1]
#        self.r = r
#        self.grid = snap[0].grid
#        self.timestep = snap[0].timestep
#        self.y_init = self.Y0[:,0]
#
#        self.u_nominal = u_nominal
#
##       self.Y0 = self.norm(self.Y0)
##       self.Y1 = self.norm(self.Y1)
#        self.U0 /= self.u_nominal
#
#        print('Running DMD...')
#        print('    Size of full model:', self.n)
#        print('    Size of reduced model:', self.r)
#        print('    Number of snapshot pairs:', self.p)
#
#        self.start_time = time.time()
#
#        self.A, self.B, self.Uhat, self.Lambda, self.Beta, self.Phi = DMDc2(self.Y0, self.Y1, self.U0, r, rtil, Uhat)
#       #self.C = self.Uhat[self.snap.n_points + self.snap.cline,:] # Choose velocity v2 of centerline points as output of the model
#
#        self.x_init = np.linalg.pinv(self.Uhat) @ self.y_init
#
#
#
#        #### CHECK THIS LINE::
#        self.C = self.Uhat[self.grid.n_points + self.grid.idx_coarse,:] # Choose velocity v2 of centerline points as output of the model
#
#
#        print('    Maximum eigenvalue: ', np.max(np.absolute(self.Lambda)))
#        print('Done! Time:', time.time() - self.start_time, 's\n')
#
#    # Normalization of snapshots
#    def norm(self, Y):
#        return np.vstack((
#            (Y[:self.n_points] - self.v1_mean)/self.v1_std,
#            (Y[self.n_points:2*self.n_points] - self.v2_mean)/self.v2_std,
#            (Y[2*self.n_points:] - self.v3_mean)/self.v3_std
#        ))
#
#    # Map normalized snapshots back
#    def map(self, Y):
#        return np.vstack((
#            (Y[:self.n_points]*self.v1_std + self.v1_mean),
#            (Y[self.n_points:2*self.n_points]*self.v2_std + self.v2_mean),
#            (Y[2*self.n_points:]*self.v3_std + self.v3_mean)
#        ))
#
##   def y_init(self):
##       return self.y_init
#
##   def x_init(self):
##       return np.linalg.pinv(self.Uhat) @ self.y_init
#
##   def response(self, y_init, u):
##       nsteps = u.shape[1]
##       x = np.zeros((self.r, nsteps))
##       y = np.zeros((self.n, nsteps))
##       y_init_norm = self.norm(np.expand_dims(y_init, axis=1)).flatten()
##       x[:,0] = self.Uhat.conj().t @ y_init_norm
##       y[:,0] = y_init_norm
##       u /= self.u_nominal
##       for i in range(1,nsteps):
##           x[:,i] = self.a @ x[:,i-1] + self.b @ u[:,i-1]
##           y[:,i] = self.Uhat @ x[:,i]
##       return x, self.map(y)
#
#
#    def response(self, x_init, u):
#        nsteps = u.shape[0]
#        x = np.zeros((nsteps+1, self.r))
#        y = np.zeros((nsteps+1, self.n))
#        x[0] = x_init
#        y[0] = self.Uhat @ x_init
#       #u /= self.u_nominal
#        for i in range(1,nsteps):
#            x[i] = self.A @ x[i-1] + self.B @ u[i-1]
#            y[i] = self.Uhat @ x[i]
#        return y, x
#

def DMDc(Y0, Y1, U0, r, rtil, Uhat=None):
    """
    Inputs: 
        Y0 : first snapshot matrix
        Y1 : second snapshot matrix (after one time step)
        U0 : corresponding input
        r  : order of the reduced model

    Outputs:
        rsys : state space representation of the reduced-order linear system

    """

    n = Y1.shape[0]
    m = U0.shape[0]
    p = U0.shape[1]

    U, Sig, VT = np.linalg.svd(np.vstack((Y0, U0)), full_matrices=False)
    thres = 1.0e-5
    rtil = np.sum(np.diag(Sig) > thres)
    #rtil = np.minimum(3*r,p)
    #rtil = np.maximum(r, rtil)
    Util = U[:,:rtil]
    print('    rtil = ', rtil)

    Sigtil = np.diag(Sig)[:rtil,:rtil]
    Vtil = VT.T[:,:rtil]

    U_A = Util[:n,:]
    U_B = Util[n:n+m,:]

    if Uhat is None:
        ## CHECK IF IT IS SVD OF Y1 OR Y0
        Uhat = np.linalg.svd(Y0, full_matrices=False)[0][:,:r]

    print('    Number of POD modes used: ', Uhat.shape[1])

   #U1, Sig1, VT1 = np.linalg.svd(Y1, full_matrices=False)
   #Y1 = U1[:,:rtil] @ np.diag(Sig1)[:rtil,:rtil] @ VT[:rtil,:]

    Ar = (Uhat.conj().T @ Y1) @ Vtil @ np.linalg.inv(Sigtil) @ U_A.conj().T @ Uhat
    Br = (Uhat.conj().T @ Y1) @ Vtil @ np.linalg.inv(Sigtil) @ U_B.conj().T

    eigval, W = np.linalg.eig(Ar)
    Lambda = np.diag(eigval)
    Beta = np.linalg.inv(W) @ Br
    Phi = Uhat @ W

    p_error = np.linalg.norm((Uhat.conj().T.dot(Y1) - Ar.dot(Uhat.conj().T.dot(Y0)) - Br.dot(U0)), ord='fro')/ \
              np.linalg.norm((Uhat.conj().T.dot(Y1)), ord='fro')*100
    print('    DMDc error = ', p_error, '%')

#   return signal.dlti(Lambda, Beta, Phi, np.zeros([n,m]), dt = 100.0)
    return Ar, Br, Uhat, Lambda, Beta, Phi


def DMDc2(Y0, Y1, U0, nr, rtil, Uhat=None):
    """
    Inputs: 
        Y0 : first snapshot matrix
        Y1 : second snapshot matrix (after one time step)
        U0 : corresponding input
        r  : order of the reduced model

    Outputs:
        rsys : state space representation of the reduced-order linear system

    """

    ny = Y1.shape[0]
    nm = U0.shape[0]
    p = U0.shape[1]

    if Uhat is None:
        Uhat = np.linalg.svd(Y0, full_matrices=False)[0][:,:nr]

    print('    Number of POD modes used: ', Uhat.shape[1])

    X0 = Uhat.conj().T @ Y0
    X1 = Uhat.conj().T @ Y1


    U, Sig, VT = np.linalg.svd(np.vstack((X0, U0)), full_matrices=False)
    thres = 1.0e-2
    rtil = np.sum(np.diag(Sig) > thres)
    #rtil = np.minimum(3*r,p)
    #rtil = np.maximum(r, rtil)
    Util = U[:,:rtil]
    print('    rtil = ', rtil)
    Sigtil = np.diag(Sig)[:rtil,:rtil]
    Vtil = VT.T[:,:rtil]

    U_A = Util[:nr,:]
    U_B = Util[nr:nr+nm,:]

    Ar = X1 @ Vtil @ np.linalg.inv(Sigtil) @ U_A.conj().T
    Br = X1 @ Vtil @ np.linalg.inv(Sigtil) @ U_B.conj().T

    eigval, W = np.linalg.eig(Ar)
    Lambda = np.diag(eigval)
    Beta = np.linalg.inv(W) @ Br
    Phi = Uhat @ W

    pod_error = np.linalg.norm((Y1 - Uhat @ X1), ord='fro')/ \
                np.linalg.norm(Y1, ord='fro')*100
    dmd_error = np.linalg.norm((Y1 - Uhat @ (Ar @ X0 + Br @ U0)), ord='fro')/ \
              np.linalg.norm(Y1, ord='fro')*100
    print('    POD error = ', pod_error, '%')
    print('    DMDc error = ', dmd_error, '%')

    return Ar, Br, Uhat, Lambda, Beta, Phi


def dns_vs_dmd_error(ydns, ydmd):
    return np.linalg.norm(ydns - ydmd, ord=2)/np.linalg.norm(ydns, ord=2)
   #return np.linalg.norm(ydns - ydmd, ord='fro')/np.linalg.norm(ydns, ord='fro')




