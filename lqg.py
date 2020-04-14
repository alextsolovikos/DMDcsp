import numpy as np
import control
import scipy as sp


"""
    Useful Functions
"""
def is_pd(X):
    return np.all(np.linalg.eigvals(X) > 0)

def is_psd(X):
    return np.all(np.linalg.eigvals(X) >= 0)


"""
    Linear Quadratic Gaussian Regulator & Estimator Class
"""
class lqg(object):

    def __init__(self, A, B, C, Qr, Rr, Qe, Re):
        self.A = A
        self.B = B
        self.C = C
        self.Qr = Qr
        self.Rr = Rr
        self.Qe = Qe
        self.Re = Re

        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.ny = C.shape[0]

        ###########################################
        # Infinite-horizon discrete-time LQR design
        ###########################################
        ctrb_rank = np.linalg.matrix_rank(control.ctrb(A, B))
#       if ctrb_rank != self.nx:
#           raise ValueError('The pair A, B must be controllable (rank of controllability matrix is %d instead of %d).' % (ctrb_rank, self.nx))

        if not is_psd(Qr):
            raise ValueError('Qr must be positive semi-definite.')

        if not is_pd(Rr):
            raise ValueError('Rr must be positive definite.')

        # Solve Discrete Algebraic Riccati Equation
        Pr = sp.linalg.solve_discrete_are(A, B, Qr, Rr)

        # Feedback gain
        self.Kr = - np.linalg.inv(Rr + B.conj().T @ Pr @ B) @ (B.conj().T @ Pr @ A)



    def lqr(self, x):
        return np.real(self.Kr @ x)



    ###########################################
    # Kalman Filter State Estimation
    ###########################################
    def lqe(self, xk, uk, Pk, z):

        # Propagate x and P to k+1
        x_prior = self.A @ xk + self.B @ uk
        P_prior = self.A @ Pk @ self.A.conj().T + self.Qe

        # Measurement update
        Ke = P_prior @ self.C.conj().T @ np.linalg.inv(self.C @ P_prior @ self.C.conj().T + self.Re)
        x = x_prior + Ke @ (z - self.C @ x_prior)
        P = P_prior - Ke @ self.C @ P_prior

        return x, P







