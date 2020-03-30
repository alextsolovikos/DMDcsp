import numpy as np
import os


# Training case
case_name = input("Enter case name: ")

# Directories
# home_dir = os.path.expanduser('~/')
home_dir = '/'
project_dir = 'Volumes/RESEARCH/dns/dmdcsp_paper/flat_plate_wake/'
case_dir = case_name + '/'
snapshot_data_dir = home_dir + project_dir + case_dir + 'snapshots/'
#input_data_dir = home_dir + project_dir + case_dir + 'setup/input-signal/'

grid = np.loadtxt(snapshot_data_dir + 'grid.dat')
wy = np.loadtxt(snapshot_data_dir + 'wy.dat').T
u = np.expand_dims(np.loadtxt(snapshot_data_dir + 'input.dat'), axis=0)

np.save('data/' + case_name + '-grid.npy', grid)
np.save('data/' + case_name + '-wy.npy', wy)
np.save('data/' + case_name + '-input.npy', u)





