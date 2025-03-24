import vtk # Needed for pyvista
import ufl

import numpy   as np
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py         import MPI
from utilities.mesh import create_ventricle_volumes_meshtags

# Set matplotlib properties
plt.rcParams.update({
    "font.family" : "Arial",
    "axes.spines.top" : False,
    "axes.spines.right" : False
})

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
model_version = 'C'
mesh_version= 'original'
molecule = 'D3'
cilia_scenarios = ['all_cilia', 'rm_dorsal', 'rm_ventral', 'rm_anterior']
f = 2.22
dt = 1/f/20
original_mesh_input_filename = f'../output/flow/checkpoints/velocity_mesh={mesh_version}_model={model_version}_ciliaScenario={cilia_scenarios[0]}_dt={dt:.4g}'
mesh = a4d.read_mesh(comm=comm, filename=original_mesh_input_filename, engine="BP4", ghost_mode=gm)
ct, ROI_tags = create_ventricle_volumes_meshtags(mesh)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=ct)
volumes = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(tag))), op=MPI.SUM) for tag in ROI_tags]
volumes[3] += volumes[2]+volumes[1]+volumes[0] # Add ROI volumes 1, 2, and 3 to get the total of ROI 4

# Load data c_hat, the total concenctration in each ROI
with open(f'../output/transport/mesh={mesh_version}_model={model_version}_molecule={molecule}_ciliaScenario={cilia_scenarios[0]}_dt={dt:.4g}/data/c_hats.npy', 'rb') as file: c_bar_original    = np.load(file)
with open(f'../output/transport/mesh={mesh_version}_model={model_version}_molecule={molecule}_ciliaScenario={cilia_scenarios[1]}_dt={dt:.4g}/data/c_hats.npy', 'rb') as file: c_bar_dorsal  = np.load(file)
with open(f'../output/transport/mesh={mesh_version}_model={model_version}_molecule={molecule}_ciliaScenario={cilia_scenarios[2]}_dt={dt:.4g}/data/c_hats.npy', 'rb') as file: c_bar_ventral = np.load(file)
with open(f'../output/transport/mesh={mesh_version}_model={model_version}_molecule={molecule}_ciliaScenario={cilia_scenarios[3]}_dt={dt:.4g}/data/c_hats.npy', 'rb') as file: c_bar_anterior = np.load(file)

# Divide the total concentrations in the ROIs by the volume of the respective ROI
for i in ROI_tags: 
    c_bar_original[:, i-1] /= volumes[i-1]
    c_bar_dorsal[:, i-1]   /= volumes[i-1]
    c_bar_ventral[:, i-1]  /= volumes[i-1]
    c_bar_anterior[:, i-1]  /= volumes[i-1]

# Get the number of timesteps
num_timesteps = c_bar_original.shape[0]
c_threshold1 = 0.25
c_threshold2 = 0.1
times = dt*np.arange(num_timesteps)

#-----------------------------------------------#
# Calculate times to threshold
t_hats_original = np.array([0]*len(ROI_tags), dtype=np.float64)
t_hats_dorsal = np.array([0]*len(ROI_tags), dtype=np.float64)
t_hats_ventral = np.array([0]*len(ROI_tags), dtype=np.float64)
t_hats_anterior = np.array([0]*len(ROI_tags), dtype=np.float64)

# define the first time-instant where c_threshold is exceeded
# as the "time to reach threshold" 
for i in range(len(t_hats_original)):
    # Set threshold value to be used to calculate "time to threshold"
    if i<4:
        c_threshold = c_threshold1
    else:
        c_threshold = c_threshold2
    t_hat_original = np.where(c_bar_original[:, i] > c_threshold)[0][0]
    t_hats_original[i] = t_hat_original
    t_hat_dorsal = np.where(c_bar_dorsal[:, i] > c_threshold)[0][0]
    t_hats_dorsal[i] = t_hat_dorsal
    t_hat_ventral = np.where(c_bar_ventral[:, i] > c_threshold)[0][0]
    t_hats_ventral[i] = t_hat_ventral
    t_hat_anterior = np.where(c_bar_anterior[:, i] > c_threshold)[0][0]
    t_hats_anterior[i] = t_hat_anterior

# Multiply by timestep size to get t_hats in seconds
t_hats_original *= dt; t_hats_dorsal *= dt; t_hats_ventral *= dt; t_hats_anterior *= dt

#-------- Plot the figures --------#
fig_c, ax_c = plt.subplots(num=1, nrows=2, ncols=3, figsize=[20, 12]) # Concentration curves
lw = 3 # linewidth
msize = 10

import colormaps as cm
green = cm.dark2_3.colors[0]
orange = cm.dark2_3.colors[1]
purple = cm.puor_4.colors[3]
colors = ['k', green, purple, orange]

# Plot concentrations
idx = 0
for row_idx in range(ax_c.shape[0]):
    for col_idx in range(ax_c.shape[1]):
        tag = ROI_tags[idx]
        ca_c = ax_c[row_idx, col_idx]
        
        ca_c.plot(times, c_bar_original[:, idx], color=colors[0], label='Original', linewidth=lw)
        ca_c.plot(times, c_bar_dorsal[:, idx]  , color=colors[1], linestyle='--', marker='o', markevery=2000, label=r'Dorsal paralyzed', linewidth=lw, markersize=msize)
        ca_c.plot(times, c_bar_ventral[:, idx] , color=colors[2], linestyle='--', marker='^', markevery=2000, label=r'Ventral paralyzed', linewidth=lw, markersize=msize)
        ca_c.plot(times, c_bar_anterior[:, idx] , color=colors[3], linestyle='--', marker='s', markevery=2000, label=r'Anterior paralyzed', linewidth=lw, markersize=msize)

        ca_c.set_title(f'ROI {tag}', fontsize=30, pad=2.0)
        ca_c.set_xticks([0, 250, 500, 750])
        ca_c.tick_params(labelsize=30)

        if col_idx==0: ca_c.set_ylabel(r"Mean conc. $\overline{c}$ [-]", fontsize=32)  
        if row_idx==(ax_c.shape[0]-1): ca_c.set_xlabel("Time [s]", fontsize=35, labelpad=25)
        if idx==0: ca_c.legend(loc='lower right', fontsize=22, frameon=True, fancybox=False, edgecolor='k')
        
        idx += 1

# Set tight layout and show (and optionally save)
plt.tight_layout()
plt.subplots_adjust(hspace=0.30)
save_figs = 1
if save_figs: fig_c.savefig(f"../output/illustrations/compare_cilia/concentrations_model{model_version}.png")

plt.show()