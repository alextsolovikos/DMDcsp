import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import patches
plt.rc('text', usetex=True)
plt.rc('font', size=16)
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pickle

# Custom libraries
import data_loader
import dmdcsp

# Load full model
model = pickle.load(open('model_full.p', 'rb'))

# Load sparse
[sys, C, Qe, Re, sens] = pickle.load(open('model_sparse.p', 'rb'))

# Load controller
controller = pickle.load(open('dns_controller_data/controller.p', 'rb'))

order = model.sp_stats['nr']
Ploss = model.sp_stats['P_loss']
z0 = model.sp_stats['z_0']
nx = sys.nx

order_uq, indices = np.unique(order, return_index=True)
Ploss_uq = Ploss[indices]

# Choose data to plot
sys_i = 49
case_name = "gamma_lqg_8_modes_R4"

# Load DNS data
nsteps = 4500
nu = model.nu

grid_full = data_loader.grid(case_name, skip_points=1)
dns_data = data_loader.flow_data(grid_full, case_name, timestep_skip=1, start=0, end=nsteps)

Y = dns_data.wy
U = dns_data.u
Z = Y[sens,:]


x_best = np.linalg.pinv(sys.C) @ Y
x_init = np.zeros(nx, dtype=complex)
P_init = np.eye(nx, dtype=complex)
x_hat = np.zeros((nx, nsteps), dtype=complex)
P = np.zeros((nx,nx,nsteps), dtype=complex)
x_hat[:,0] = x_init
P[:,:,0] = P_init
u_star = np.zeros((nu,nsteps))

for k in range(nsteps-1):
    u_star[:,k] = controller.lqr(x_hat[:,k])
    x_hat[:,k+1], P[:,:,k+1] = controller.lqe(x_hat[:,k], u_star[:,k],  P[:,:,k], Z[:,k+1])

y_hat = sys.C @ x_hat

x_error = np.linalg.norm(x_hat - x_best, ord=2, axis=0) / np.linalg.norm(x_best, ord=2, axis=0)
y_error = np.linalg.norm(y_hat - Y, ord=2, axis=0) / np.linalg.norm(Y, ord=2, axis=0)


# Load Tecplot data
wy_data_0 = np.loadtxt('data/timestep_1.txt', skiprows=1)
wy_data_1 = np.loadtxt('data/timestep_2.txt', skiprows=1)
wy_data_2 = np.loadtxt('data/timestep_3.txt', skiprows=1)


m_size = 50
spm_size = 50

##########################
# Plot percentage loss
##########################
fig, axs = plt.subplots(1, figsize=(6,4), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.18, right=0.95, top=0.95, bottom=0.18)

axs.plot(order_uq, Ploss_uq, c='k', zorder=9, clip_on=False)
axs.scatter(order_uq, Ploss_uq, s=m_size, marker='o', facecolor='none', edgecolor='k', zorder=10, clip_on=False)
axs.scatter(order[sys_i], Ploss[sys_i], s=spm_size, marker='x', color='darkred', zorder=11, clip_on=False)

axs.set_axisbelow(True)
axs.set_xlabel('$n_x$')
axs.set_ylabel('$P_{\\mathrm{error}},\ \%$')
axs.set_xticks(np.arange(0, 31, 5))
axs.set_yticks(np.arange(0, 101, 20))
plt.grid(True)
axs.set_xlim([0,30])
axs.set_ylim([0,100])
plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/Ploss.eps')


##########################
# Plot eigenvalues
##########################
lamb = np.diag(model.Lambda)
lamb_sp = np.diag(sys.A)
print('eigenvalue magnitudes:')
print(np.abs(lamb_sp))
fig, axs = plt.subplots(1, figsize=(6,4), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.22, right=0.95, top=0.95, bottom=0.18)

# Unit circle
circle = Circle((0,0), 1.0, edgecolor='k', facecolor='none', linewidth=1)
axs.add_artist(circle)

# Eigenvalues of full system
axs.scatter(np.real(lamb), np.imag(lamb), s=m_size, marker='o', facecolor='none', edgecolor='k', zorder=15, clip_on=False)

# Eigenvalues of sparse system
axs.scatter(np.real(lamb_sp), np.imag(lamb_sp), s=spm_size, marker='x', color='darkred', zorder=16, clip_on=False)

axs.set_axisbelow(True)
axs.set_xlabel('$\mathrm{Re}(\lambda_i)$')
axs.set_ylabel('$\mathrm{Im}(\lambda_i)$')
axs.set_xticks(np.arange(0.97, 1.02, 0.01))
axs.set_yticks(np.arange(-0.2, 0.25, 0.1))
axs.set_xlim([0.97, 1.005])
axs.set_ylim([-0.25, 0.25])
plt.grid(True)
plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/eigenvalues.eps')


##########################
# Plot frequencies
##########################

Ts = 5.0 * 0.00125

# Full system
freq = np.abs(np.imag(np.log(np.diag(model.rsys[0].A)))/(2.0*np.pi)/Ts)
ampl = np.abs(z0[0])

# Sparse system
freq_sp = np.abs(np.imag(np.log(np.diag(model.rsys[sys_i].A)))/(2.0*np.pi)/Ts)
ampl_sp = np.abs(z0[sys_i][:nx])
print(freq_sp)
print(ampl_sp)

fig, axs = plt.subplots(1, figsize=(6,4), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.22, right=0.95, top=0.95, bottom=0.2)

axs.scatter(freq, ampl, s=m_size, marker='o', facecolor='none', edgecolor='k', zorder=10, clip_on=False)
axs.scatter(freq_sp, ampl_sp, s=spm_size, marker='x', color='darkred', zorder=11, clip_on=False)
axs.set_axisbelow(True)
axs.set_xlabel('$\mathrm{Im}(\log(\lambda_i))$')
axs.set_ylabel('$\|x(0)\|$')
axs.set_xlim([0, 6])
axs.set_ylim([0, 100])
axs.set_xticks(np.arange(0, 6.5, 1.0))
axs.set_yticks(np.arange(0, 101, 20))
plt.grid(True)
plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/amplitudes.eps')


##########################
# Plot optimal input
##########################
fig, axs = plt.subplots(1, figsize=(6,3), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.2, right=0.95, top=0.95, bottom=0.3)

t = np.arange(0,nsteps,1)*Ts
axs.plot(t, U[0,:], c = 'k')

axs.set_axisbelow(True)
axs.set_xlabel('$kT_s$')
axs.set_ylabel('$u(k)$')
axs.set_xlim([0,nsteps*Ts])
axs.set_ylim([-1, 1])
axs.set_xticks(np.arange(0, nsteps*Ts + 0.01, 5))
axs.set_yticks(np.arange(-1, 1.01, 0.5))
plt.grid(True)
plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/optimal_input.eps')



##########################
# Plot sensor meas.
##########################
fig, axs = plt.subplots(1, figsize=(6,3), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.2, right=0.95, top=0.95, bottom=0.3)

t = np.arange(0,nsteps,1)*Ts
axs.plot(t, Z[1,:], c = 'k')

axs.set_axisbelow(True)
axs.set_xlabel('$kT_s$')
axs.set_ylabel('$z_1(k)$')
axs.set_xlim([0,nsteps*Ts])
axs.set_xticks(np.arange(0, nsteps*Ts + 0.01, 5))
axs.set_yticks(np.arange(-8, 8.01, 4))
plt.grid(True)
plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/measurement_2.eps')

"""
##########################
# Plot optimal input AND sensor meas.
##########################
fig, axs = plt.subplots(1, figsize=(7,4), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.2, right=0.95, top=0.95, bottom=0.18)

t = np.arange(0,nsteps,1)*Ts
axs.plot(t, U[0,:], c = 'k')
#axs.plot(u_star[0,:], c = 'k')
#axs.plot(U[0,:nsteps], c = 'r', linestyle='dashed')

axs.set_axisbelow(True)
axs.set_xlabel('$kT_s$')
axs.set_ylabel('$u(k)$')
axs.set_xlim([0,nsteps*Ts])
axs.set_ylim([-1, 1])
axs.set_xticks(np.arange(0, nsteps*Ts + 0.01, 5))
axs.set_yticks(np.arange(-1, 1.01, 0.5))

axs2 = axs.twinx()
axs2.plot(t, Z[1,:], c = 'k', linestyle='dashed')
axs2.set_ylabel('$z_1(k)$')
axs2.set_yticks(np.arange(-8, 8.01, 4))


plt.grid(True)
plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/optimal_input_and_meas.eps')
"""

##########################
# Plot estimation errors
##########################
fig, axs = plt.subplots(1, figsize=(6,3), facecolor='w', edgecolor='k')
plt.subplots_adjust(hspace=0.6, left=0.2, right=0.95, top=0.95, bottom=0.3)

t = np.arange(0,nsteps,1)*Ts
axs.plot(t, 100*y_error, c = 'k')

axs.set_axisbelow(True)
axs.set_xlabel('$kT_s$')
axs.set_ylabel('$e_y(k),\, \%$')
axs.set_xlim([0,nsteps*Ts])
axs.set_ylim([0,100])
axs.set_xticks(np.arange(0, nsteps*Ts + 0.01, 5))
axs.set_yticks(np.arange(0, 101, 25))
plt.grid(True)
plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/output_error.eps')








##########################
# Plot vorticity contours with DMD grid
##########################
nlevels = 51
wymin = -25.
wymax = 25.

X_dns = wy_data_0[:,0].reshape(385,769)
Z_dns = wy_data_0[:,2].reshape(385,769)
Wy_0 = wy_data_0[:,3].reshape(385,769)
xmax = np.max(X_dns)

fig, axs = plt.subplots(1, figsize=(6,4), facecolor='w', edgecolor='k')
plt.subplots_adjust(right=0.96, top=0.98, bottom=0.0, left=0.08)
cont = axs.contourf(X_dns, Z_dns, Wy_0, nlevels, cmap='coolwarm', vmin=wymin, vmax=wymax)
m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(Wy_0)
m.set_clim(-10, 10)
plt.colorbar(m, boundaries=np.linspace(-12, 12, 7), shrink=0.6, orientation='horizontal', pad=0.25)

# Plot grid
axs.scatter(grid_full.x, grid_full.z, color='k', s=0.05)

# Plot sensors
axs.scatter(grid_full.x[sens], grid_full.z[sens], color='k', s=30, marker='^')

# Plot actuator
axs.scatter(1.95, 0.96, color='k', s=40, marker='o')
L = 0.15
Hw = 0.01
Hl = 0.01
x_act = 1.95
y_act = 0.96
dx_act = L*np.cos(70.*np.pi/180.)
dy_act = L*np.sin(70.*np.pi/180.)
axs.annotate("", xytext=(x_act, y_act), xy=(x_act + dx_act, y_act + dy_act), arrowprops=dict(arrowstyle="->"))
axs.annotate("", xytext=(x_act, y_act), xy=(x_act - dx_act, y_act - dy_act), arrowprops=dict(arrowstyle="->"))

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

# Grid rectangle
grid_rectangle = patches.Rectangle((grid_full.xmin, grid_full.zmin),
                                    grid_full.xmax - grid_full.xmin,
                                    grid_full.zmax - grid_full.zmin,
                                    linewidth = 0.75, edgecolor='k', facecolor='none')
axs.add_patch(grid_rectangle)

axs.set_xlim([1.1,3.4])
axs.set_ylim([0.4,1.6])
axs.set_xlabel("Streamwise")
axs.set_ylabel("Normal")
axs.set_xticks(np.arange(1.5, 3.5, 0.5))
axs.set_aspect('equal', 'box')

plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/dns_with_grid.eps')


##########################
# Plot vorticity contours - 2
##########################
Wy_1 = wy_data_1[:,3].reshape(385,769)

fig, axs = plt.subplots(1, figsize=(6,4), facecolor='w', edgecolor='k')
plt.subplots_adjust(right=0.96, top=0.98, bottom=0.0, left=0.08)
cont = axs.contourf(X_dns, Z_dns, Wy_1, nlevels, cmap='coolwarm', vmin=wymin, vmax=wymax)
m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(Wy_0)
m.set_clim(-10, 10)

plt.colorbar(m, boundaries=np.linspace(-12, 12, 7), shrink=0.6, orientation='horizontal', pad=0.25)

# Plot sensors
axs.scatter(grid_full.x[sens], grid_full.z[sens], color='k', s=30, marker='^')

# Plot actuator
axs.scatter(1.95, 0.96, color='k', s=40, marker='o')

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

axs.set_xlim([1.1,3.4])
axs.set_ylim([0.4,1.6])
axs.set_xlabel("Streamwise")
axs.set_ylabel("Normal")
axs.set_xticks(np.arange(1.5, 3.5, 0.5))
axs.set_aspect('equal', 'box')

plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/dns_2.eps')


##########################
# Plot vorticity contours - 3
##########################
Wy_2 = wy_data_2[:,3].reshape(385,769)

fig, axs = plt.subplots(1, figsize=(6,4), facecolor='w', edgecolor='k')
plt.subplots_adjust(right=0.96, top=0.98, bottom=0.0, left=0.08)
cont = axs.contourf(X_dns, Z_dns, Wy_2, nlevels, cmap='coolwarm', vmin=wymin, vmax=wymax)
m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(Wy_0)
m.set_clim(-10, 10)
plt.colorbar(m, boundaries=np.linspace(-12, 12, 7), shrink=0.6, orientation='horizontal', pad=0.25)
#plt.colorbar(m, boundaries=np.linspace(-10, 10, 6))
#plt.clim(wymin, wymax)
#cbar = fig.colorbar(cont, ax=axs, orientation='vertical')
#cbar.ax.set_autoscale_on(True)
#cbar.set_ticks(np.linspace(wymin, wymax, num=6, endpoint=True))

# Plot sensors
axs.scatter(grid_full.x[sens], grid_full.z[sens], color='k', s=30, marker='^')

# Plot actuator
axs.scatter(1.95, 0.96, color='k', s=40, marker='o')

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

axs.set_xlim([1.1,3.4])
axs.set_ylim([0.4,1.6])
axs.set_xlabel("Streamwise")
axs.set_ylabel("Normal")
axs.set_xticks(np.arange(1.5, 3.5, 0.5))
axs.set_aspect('equal', 'box')

plt.savefig('/Users/atsol/research/papers/dmdcsp-paper/figures/dns_3.eps')


plt.show()




