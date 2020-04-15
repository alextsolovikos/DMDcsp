import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.rc('text', usetex=True)
plt.rc('font', size=24)
import control
import pickle

# Custom libraries
import data_loader
import dmdcsp


# Import training data
#training_case = "training_1"
#training_case = "training_2"
#training_case = "training_3"
training_case = "training_4"
#training_case = "zero_input"
grid_full = data_loader.grid(training_case, skip_points=1)
train_data = data_loader.flow_data(grid_full, training_case, timestep_skip=1, start=0, end=1500)
#train_data_2 = data_loader.flow_data(grid_full, training_case, timestep_skip=1, start=1500, end=2000)

#train_data.plot_grid()
#plt.show()

# Sensors
sens = [236, 967]

#anim = train_data.plot(sens=sens)
#plt.show()
#exit()

#Y = np.hstack((train_data.wy, train_data_2.wy))
#U = np.hstack((train_data.u, train_data_2.u))
Y = train_data.wy
U = train_data.u

nx = 30 # Number of POD modes to keep

# Full dmdc model
#model = dmdcsp.dmdcsp(Y, U, nx=nx, u_nominal=1, dt=1)
model = dmdcsp.dmdcsp(Y, U, nx=nx, u_nominal=1, dt=1)

#anim = model.plot_dmd_modes(grid_full)
#anim = model.plot_model_response(model.sys_dmd, grid_full)
#plt.show()
#exit()


"""
    Compute Sparse Models
"""
num = 100
niter = 5
gamma = np.logspace(2.0, 6.7, num=num)

stats = model.sparse_batch(gamma, niter)

order = stats['nr']
Ploss = stats['P_loss']
z0 = stats['z_0']

order_uq, indices = np.unique(order, return_index=True)
Ploss_uq = Ploss[indices]

# Choose model
sys_i = int(input('Choose the sparse model id to use: '))

nr = stats['nr'][sys_i]
# Print covariances
C, Qe, Re = model.compute_noise_cov(sys_i, sens)

print('\n Sensors: ', sens)

print('\nQe = ')
print(Qe)
print('eig(Qe)')
print(np.linalg.eig(Qe)[0])

print('\nRe = ')
print(Re)
print('eig(Re)')
print(np.linalg.eig(Re)[0])

print('\nRank of observability matrix: %d of %d' % (np.linalg.matrix_rank(control.obsv(model.rsys[sys_i].A, C)), nr))

Ts = 5
print('Mode frequencies:')
print(np.abs(np.angle(np.diag(model.rsys[sys_i].A)))/(2.0*np.pi*Ts))

"""
    Save full model & final model
"""
pickle.dump(model, open('model_full.p', 'wb'))

final_model = [model.rsys[sys_i], C, Qe, Re, sens]
pickle.dump(final_model, open('model_sparse.p', 'wb'))


"""
    Plot results
"""
 

##########################
# Plot percentage loss
##########################
fig, axs = plt.subplots(1, figsize=(7,5), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.18, right=0.95, top=0.95, bottom=0.18)

axs.plot(order_uq, Ploss_uq, c='k', fillstyle='none', marker='o')
#axs.plot(order, Ploss, c='k', linestyle='solid', marker='o', facecolors='none', edgecolors='k')
#axs.scatter(order, Ploss, facecolors='none', edgecolors='k', marker='o')
axs.plot(order[sys_i], Ploss[sys_i], c='r', fillstyle='none', marker='o')


axs.set_axisbelow(True)
axs.set_xlabel('Model Order')
axs.set_ylabel('Percentage Loss, \%')
plt.grid(True)
#plt.savefig('/Users/atsol/research/papers/AIAA-MPC-of-LSMS/figures/case_3_inputs.eps')

plt.show()


##########################
# Plot eigenvalues
##########################
lamb = np.diag(model.Lambda)
lamb_sp = np.diag(model.rsys[sys_i].A)
fig, axs = plt.subplots(1, figsize=(7,5), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.22, right=0.95, top=0.95, bottom=0.18)

# Unit circle
circle = Circle((0,0), 1.0, edgecolor='k', facecolor='none', linestyle=(0, (5, 10)))
axs.add_artist(circle)

# Eigenvalues of full system
axs.scatter(np.real(lamb), np.imag(lamb), marker='o', facecolor='none', edgecolor='k')

# Eigenvalues of sparse system
axs.scatter(np.real(lamb_sp), np.imag(lamb_sp), marker='x', color='r')

#axs.axis('equal')
axs.set_axisbelow(True)
axs.set_xlabel('$Re(\lambda_i)$')
axs.set_ylabel('$Im(\lambda_i)$')
#axs.set_ylim([0,1.4])
plt.grid(True)
#plt.savefig('/Users/atsol/research/papers/AIAA-MPC-of-LSMS/figures/case_3_inputs.eps')

plt.show()


##########################
# Plot frequencies
##########################
#freq = np.imag(np.log(np.diag(model.Lambda)))
# Full system
Ts = 5.0
freq = np.abs(np.angle(np.diag(model.rsys[0].A)))/(2.0*np.pi)/Ts
ampl = np.abs(z0[0])

# Sparse system
freq_sp = np.abs(np.angle(np.diag(model.rsys[sys_i].A)))/(2.0*np.pi)/Ts
ampl_sp = np.abs(z0[sys_i][:nx])

fig, axs = plt.subplots(1, figsize=(7,5), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.22, right=0.95, top=0.95, bottom=0.18)

axs.scatter(freq, ampl, marker='o', facecolor='none', edgecolor='k')
axs.scatter(freq_sp, ampl_sp, marker='x', color='r')
#axs.axis('equal')
axs.set_axisbelow(True)
axs.set_xlabel('Frequency')
axs.set_ylabel('$\|z(0)\|$')
#axs.set_xlim([0,np.max(freq)])
plt.grid(True)
#plt.savefig('/Users/atsol/research/papers/AIAA-MPC-of-LSMS/figures/case_3_inputs.eps')

plt.show()

