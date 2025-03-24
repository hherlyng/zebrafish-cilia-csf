import ufl

import numpy   as np
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py       import MPI
from utilities.mesh import create_ventricle_volumes_meshtags

# Set matplotlib properties
plt.rcParams.update({
    "font.family" : "Arial",
    "axes.spines.top" : False,
    "axes.spines.right" : False
})
plt.rcParams["text.latex.preamble"] += "\\usepackage{sfmath}" # Enable sans-serif math font

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
f = 2.22
dt = 1/f/20
molecule = 'D3'
mesh_version = 'original'
mesh_input_filename = f'../output/flow/checkpoints/velocity_mesh={mesh_version}_model=C_ciliaScenario=all_cilia_dt={dt:.4g}'
mesh = a4d.read_mesh(comm=comm, filename=mesh_input_filename, engine="BP4", ghost_mode=gm)

# Create meshtags and calculate ROI volumes
mt, ROI_tags = create_ventricle_volumes_meshtags(mesh)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=mt)
volumes = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(tag))), op=MPI.SUM) for tag in ROI_tags]
volumes[3] += volumes[2]+volumes[1]+volumes[0] # Add ROI 1, 2, 3 volumes to ROI 4 volume

# Load data c_bar, the total concenctration in each ROI
with open(f'../output/transport/mesh={mesh_version}_model=B_molecule={molecule}_ciliaScenario=all_cilia_dt={dt:.4g}/data/c_hats.npy', 'rb') as file:
    c_bar_B = np.load(file)
with open(f'../output/transport/mesh={mesh_version}_model=C_molecule={molecule}_ciliaScenario=all_cilia_dt={dt:.4g}/data/c_hats.npy', 'rb') as file:
    c_bar_C = np.load(file)

# Scale the total concentrations in the ROIs by the volume of the respective ROI
for i in ROI_tags: 
    c_bar_B[:, i-1] /= volumes[i-1]
    c_bar_C[:, i-1] /= volumes[i-1]

# Get the number of timesteps
num_timesteps = c_bar_B.shape[0]
c_threshold1 = 0.25 # threshold value to be used to calculate "time to threshold" for ROIs 1-4
c_threshold2 = 0.10 # threshold value to be used to calculate "time to threshold" for ROIs 5 and 6
times = dt*np.arange(num_timesteps)

#-----------------------------------------------#
# Calculate times to threshold
t_hatsB = np.array([0]*len(ROI_tags), dtype=np.float64)
t_hatsC = np.zeros_like(t_hatsB)

# define the first time-instant where c_threshold is exceeded
# as the "time to reach threshold" 
for i in range(len(t_hatsB)):
    c_threshold = c_threshold1 if i<4 else c_threshold2 

    t_hatB = np.where(c_bar_B[:, i] > c_threshold)[0][0]
    t_hatsB[i] = t_hatB

    t_hatC = np.where(c_bar_C[:, i] > c_threshold)[0][0]
    t_hatsC[i] = t_hatC

# Multiply by timestep size to get t_hats in seconds
t_hatsB *= dt; t_hatsC *= dt

#-------- Plot the figures --------#
fig_c, ax_c = plt.subplots(num=1, nrows=3, ncols=2, figsize=[14, 14]) # Concentration curves
fig_t, ax_t = plt.subplots(num=2, nrows=3, ncols=2, figsize=[14, 14]) # Time to threshold scatter plots
lw = 5 # linewidth
msize = 4 # markersize
msize_c_plot = 15
c_str = r'$\overline{c}$'
t_str = r'$\hat{t}$'

colors = ['k',        # Black, control
          '#bf0014ff' # Dark red, mutant
]

for idx, tag in enumerate(ROI_tags):
    if tag % 2 != 0:
        # Odd number -> plot in 1st column
        row_idx = int((tag-1)/2)
        col_idx = 0
    else:
        # Even number -> plot in 2nd column
        row_idx = int((tag-2)/2)
        col_idx = 1
    
    c_threshold = c_threshold1 if tag<5 else c_threshold2

    ca_c = ax_c[row_idx, col_idx]
    ca_t = ax_t[row_idx, col_idx]
        
    # Plot concentration
    ca_c.plot(times, c_bar_C[:, idx], color=colors[0],  linewidth=lw)
    ca_c.plot(times, c_bar_B[:, idx], color=colors[1],  linewidth=lw)
    ca_c.axhline(y=c_threshold, color='k', linewidth=3, alpha=0.5) # hlines: xmin=0.0, xmax=max(t_hatsB[idx], t_hatsC[idx]),
    
    # Add marker at time to threshold point
    ca_c.plot(t_hatsC[idx], c_threshold, marker='o', label='Cilia+cardiac',markersize=msize_c_plot, color=colors[0])
    ca_c.plot(t_hatsB[idx], c_threshold, marker='s', label='Cardiac', markersize=msize_c_plot, color=colors[1])

    # Add textbox with times to threshold
    textstr = r'${\hat{t}}_0$'+f'= {t_hatsC[idx]:.1f} s\n' + \
              r'${\hat{t}}_{\rm{II}}$'+f'= {t_hatsB[idx]:.1f} s'  
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    ca_c.text(0.05, 0.95, textstr, transform=ca_c.transAxes, fontsize=32, verticalalignment='top', bbox=props)
    
    ca_c.set_title(f'ROI {tag}', fontsize=35)
    ca_c.set_xticks([0, 250, 500, 750])
    ca_c.tick_params(labelsize=36)
    if idx==0: ca_c.set_yticks([0.0, 0.5, 1.0])
    if col_idx==0: ca_c.set_ylabel(r"$\overline{c}$ [-]", fontsize=40, labelpad=25)
    if row_idx==2: ca_c.set_xlabel("Time [s]", fontsize=40, labelpad=25)    

    # Plot time to threshold
    ca_t.scatter(0.75, t_hatsB[idx], color=colors[1], label='model II', linewidths=msize, marker='^')
    ca_t.scatter(0.25, t_hatsC[idx], color=colors[0], label='model 0', linewidths=msize, marker='o')
    
    ca_t.set_title(rf'{t_str} in ROI {tag}', fontsize=24)
    ca_t.tick_params(labelsize=36)
    ca_t.set_ylabel(r"Time [s]", fontsize=35, labelpad=25)
    ca_t.set_xticks([0, 0.25, 0.75, 1])
    ca_t.set_xticklabels(['', 'baseline', 'cardiac', ''])
    ca_t.set_ylim(0, ca_t.get_ylim()[1]*1.25)

    if row_idx==(ax_c.shape[0]-1): ca_c.set_xlabel("Time [s]", fontsize=40)
    if idx==0:
        ca_c.legend(loc='best', fontsize=30, frameon=True, fancybox=False, edgecolor='k')
        ca_t.legend(loc='best', fontsize=30, frameon=True, fancybox=False, edgecolor='k')

# Set tight layout and show (and optionally save)
fig_c.tight_layout(); fig_t.tight_layout()
save_figs = 1
if save_figs:
    fig_c.savefig(f"../output/illustrations/compare_models_concentrations_molecule={molecule}_mesh={mesh_version}.png")
    fig_t.savefig(f"../output/illustrations/compare_models_time_to_threshold_molecule={molecule}_mesh={mesh_version}.png")
plt.show()