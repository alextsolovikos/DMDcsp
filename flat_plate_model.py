import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.rc('text', usetex=True)
plt.rc('font', size=24)

# Custom libraries
import snapshots
import dmdcsp


# Import training data
training_case = "training_1"
#training_case = "zero_input"
grid_full = snapshots.grid(training_case, skip_points=1)
train_data = snapshots.flow_data(grid_full, training_case, timestep_skip=1, start=0, end=1000)

#anim = train_data.plot()
#plt.show()
#exit()


Y = train_data.wy
U = train_data.u

nx = 31 # Number of POD modes to keep

# Full dmdc model
model = dmdcsp.dmdcsp(Y, U, nx=nx, u_nominal=1, dt=1)

#anim = model.plot_dmd_modes(grid_full)
#anim = model.plot_model_response(model.sys_dmd, grid_full)
#plt.show()
#exit()

# Test sparsity
num = 10
niter = 5
gamma = np.logspace(2.0, 5.5, num=num)

stats = model.sparse_batch(gamma, niter)

#for i in range(num):
#    rsys[i], stats[i] = model.sparse(gamma[i])
#    Ploss[i] = stats[i]['P_loss']
#    order[i] = stats[i]['nr']
#    x_ref[i] = stats[i]['x_ref']


order = stats['nr']
Ploss = stats['P_loss']
x_ref = stats['x_ref']

order, indices = np.unique(order, return_index=True)
Ploss = Ploss[indices]


# Plot percentage loss
fig, axs = plt.subplots(1, figsize=(7,5), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.18, right=0.95, top=0.95, bottom=0.18)

axs.plot(order, Ploss, c='k', fillstyle='none', marker='o')
#axs.plot(order, Ploss, c='k', linestyle='solid', marker='o', facecolors='none', edgecolors='k')
#axs.scatter(order, Ploss, facecolors='none', edgecolors='k', marker='o')

axs.set_axisbelow(True)
axs.set_xlabel('Model Order')
axs.set_ylabel('Percentage Loss, \%')
plt.grid(True)
#plt.savefig('/Users/atsol/research/papers/AIAA-MPC-of-LSMS/figures/case_3_inputs.eps')

plt.show()


# Plot eigenvalues
lamb = np.diag(model.Lambda)
fig, axs = plt.subplots(1, figsize=(7,5), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.22, right=0.95, top=0.95, bottom=0.18)

# Unit circle
circle = Circle((0,0), 1.0, edgecolor='k', facecolor='none', linestyle=(0, (5, 10)))
axs.add_artist(circle)
axs.scatter(np.real(lamb), np.imag(lamb), marker='o', facecolor='none', edgecolor='k')

#axs.axis('equal')
axs.set_axisbelow(True)
axs.set_xlabel('$Re(\lambda_i)$')
axs.set_ylabel('$Im(\lambda_i)$')
#axs.set_ylim([0,1.4])
plt.grid(True)
#plt.savefig('/Users/atsol/research/papers/AIAA-MPC-of-LSMS/figures/case_3_inputs.eps')

plt.show()

# Plot frequencies
#freq = np.imag(np.log(np.diag(model.Lambda)))
freq = np.abs(np.angle(np.diag(model.Lambda)))/(2.0*np.pi)
ampl = np.abs(x_ref[0][:nx])
fig, axs = plt.subplots(1, figsize=(7,5), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.22, right=0.95, top=0.95, bottom=0.18)

axs.scatter(1./freq, ampl, marker='o', facecolor='none', edgecolor='k')
#axs.axis('equal')
axs.set_axisbelow(True)
axs.set_xlabel('Frequency')
axs.set_ylabel('$\|z(0)\|$')
#axs.set_xlim([0,np.max(freq)])
plt.grid(True)
#plt.savefig('/Users/atsol/research/papers/AIAA-MPC-of-LSMS/figures/case_3_inputs.eps')

plt.show()

