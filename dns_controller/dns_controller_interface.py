import lqg
import pickle

controller = pickle.load(open('controller.p', 'rb'))

# Read measurements
zkp = np.loadtxt('snapshots/wy.dat', usecols=(236, 967))[:,-1]

# Read previous data
xk = np.load('controller/x_hat_prev.npy')
Pk = np.load('controller/P_prev.npy')
uk = np.loadtxt('ustar.dat')

# Estimate state
xkp, Pkp = controller.lqe(xk, uk, Pk, zkp)
# Feedback control
ukp = controller.lqr(xkp)

# Save data
np.save('controller/x_hat_prev.npy', xkp)
np.save('controller/P_prev.npy', Pkp)
np.savetxt('controller/ustar.dat', ukp)







