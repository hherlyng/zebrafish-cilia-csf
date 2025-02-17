import vtk # Needed for pyvista
import ufl

import numpy   as np
import pandas  as pd
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
k = 1 # element degree
model_version = 'C'
original_mesh_input_filename = f"../output/flow/checkpoints/variable_tau/pressure+original/model_{model_version}/velocity_data_dt=0.02252"
mesh = a4d.read_mesh(comm=comm, filename=original_mesh_input_filename, engine="BP4", ghost_mode=gm)
ct, ROI_tags = create_ventricle_volumes_meshtags(mesh)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=ct)
volumes = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(tag))), op=MPI.SUM) for tag in ROI_tags]
volumes[3] += volumes[2]+volumes[1]+volumes[0]

# Load data c_hat, the total concenctration in each ROI
with open(f"../output/transport/results/variable_tau/original/log_model_{model_version}_D3_DG1_pressureBC/data/c_hats.npy", 'rb') as file: c_bar_original    = np.load(file)
with open(f"../output/transport/results/variable_tau+rm_dorsal/original/log_model_{model_version}_D3_DG1_pressureBC/data/c_hats.npy", 'rb') as file: c_bar_dorsal  = np.load(file)
with open(f"../output/transport/results/variable_tau+rm_ventral/original/log_model_{model_version}_D3_DG1_pressureBC/data/c_hats.npy", 'rb') as file: c_bar_ventral = np.load(file)
with open(f"../output/transport/results/variable_tau+rm_telencephalic/original/log_model_{model_version}_D3_DG1_pressureBC/data/c_hats.npy", 'rb') as file: c_bar_telenc = np.load(file)

# Divide the total concentrations in the ROIs by the volume of the respective ROI
for i in ROI_tags: 
    c_bar_original[:, i-1] /= volumes[i-1]
    c_bar_dorsal[:, i-1]   /= volumes[i-1]
    c_bar_ventral[:, i-1]  /= volumes[i-1]
    c_bar_telenc[:, i-1]  /= volumes[i-1]

# Get the number of timesteps
num_timesteps = c_bar_original.shape[0]
c_threshold1 = 0.25
c_threshold2 = 0.1
f = 2.22
dt = 1/f/20
times = dt*np.arange(num_timesteps)

#-----------------------------------------------#
# Calculate times to threshold
t_hats_original = np.array([0]*len(ROI_tags), dtype=np.float64)
t_hats_dorsal = np.array([0]*len(ROI_tags), dtype=np.float64)
t_hats_ventral = np.array([0]*len(ROI_tags), dtype=np.float64)
t_hats_telenc = np.array([0]*len(ROI_tags), dtype=np.float64)

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
    t_hat_telenc = np.where(c_bar_telenc[:, i] > c_threshold)[0][0]
    t_hats_telenc[i] = t_hat_telenc

# Multiply by timestep size to get t_hats in seconds
t_hats_original *= dt; t_hats_dorsal *= dt; t_hats_ventral *= dt; t_hats_telenc *= dt

#-------- Plot the figures --------#
fig_c, ax_c = plt.subplots(num=1, nrows=2, ncols=3, figsize=[20, 12]) # Concentration curves
lw = 3 # linewidth
msize = 10

import colormaps as cm
green = cm.dark2_3.colors[0]
orange = cm.dark2_3.colors[1]
purple = cm.puor_4.colors[3]
colors = ['k', green, orange, purple]

# Plot concentrations
idx = 0
for row_idx in range(ax_c.shape[0]):
    for col_idx in range(ax_c.shape[1]):
        tag = ROI_tags[idx]
        ca_c = ax_c[row_idx, col_idx]
        
        ca_c.plot(times, c_bar_original[:, idx], color=colors[0], label='Original', linewidth=lw)
        ca_c.plot(times, c_bar_dorsal[:, idx]  , color=colors[1], linestyle='--', marker='o', markevery=2000, label=r'Dorsal paralyzed', linewidth=lw, markersize=msize)
        ca_c.plot(times, c_bar_ventral[:, idx] , color=colors[2], linestyle='--', marker='^', markevery=2000, label=r'Ventral paralyzed', linewidth=lw, markersize=msize)
        ca_c.plot(times, c_bar_telenc[:, idx] , color=colors[3], linestyle='--', marker='s', markevery=2000, label=r'Telencephalic paralyzed', linewidth=lw, markersize=msize)

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


cols = range(4)
final_c = pd.DataFrame(index=ROI_tags, columns=cols)
for i, c_bar in enumerate([c_bar_original, c_bar_dorsal, c_bar_ventral, c_bar_telenc]):
    final_c[i] = c_bar[-1, :]

fig_bars, ax_bars = plt.subplots(num=4, figsize=[18, 8])
final_c.reset_index(inplace=True)
bars = final_c.plot.bar(x='index', y=cols,
                         color=colors,
                         ax=ax_bars, rot=True, width=0.75)
hatches = ['', '\\', 'x', '/', '^']
for bar_container, hatch in zip(bars.containers, hatches):
    for bar in bar_container:
        bar.set_hatch(hatch)

ax_bars.set_xlabel("ROI number", fontsize=35, labelpad=25)
ax_bars.set_ylabel(r"Final mean concentration $\bar{c}(T)$ [-]", fontsize=35, labelpad=50)
ax_bars.tick_params(labelsize=35)
ax_bars.get_legend().remove()
# ax_bars.legend(labels=['Original', 'Fore', 'Middle', 'Hind', 'Fore+middle+hind'], loc='right', fontsize=32, frameon=True, fancybox=False, edgecolor='k')
fig_bars.tight_layout()

if save_figs: fig_bars.savefig(f"../output/illustrations/compare_cilia/final_concentrations_model{model_version}.png")

plt.show()