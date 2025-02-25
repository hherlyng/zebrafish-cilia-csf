import ufl

import numpy   as np
import pandas  as pd
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py       import MPI
from utilities.mesh import create_ventricle_volumes_meshtags

"""
Plot and compare results for the original ventricles mesh and the (more) shrunk ventricles mesh. The original ventricles mesh
is based on the initial surface geometry retrieved from the Jurisch-Yaksi lab, the shrunk versions are the original mesh
modified using the "shrink" mesh modifier in Blender. 

Three different regions of the mesh were modified, and four modified meshes are considered:
    - fore-shrunk: a mesh with shrinkage of the connection between the telencephalic ventricle and the diencephalic ventricle 
    - middle-shrunk: a mesh with shrinkage of the diencephalic ventricle in the lateral direction (shrinkage along y-axis of mesh)
    - fore-shrunk: a mesh with shrinkage of the connection between the diencephalic ventricle and the rhombencephalic ventricle
    - fore+middle+hind shrunk: a mesh with all the three modifications outlined above performed

Everything in this script with a subscript 1 is an attribute/variable of the original mesh.
Everything in this script with a subscript 2 is an attribute/variable of the fore-shrunk mesh.
Everything in this script with a subscript 3 is an attribute/variable of the middle-shrunk mesh.
Everything in this script with a subscript 4 is an attribute/variable of the hind-shrunk mesh.
Everything in this script with a subscript 5 is an attribute/variable of the fore+middle+hind-shrunk mesh.

Author: Halvor Herlyng, 2024.
"""

# Set matplotlib properties
plt.rcParams.update({
    "font.family" : "Arial",
    "axes.spines.top" : False,
    "axes.spines.right" : False
})
plt.rcParams["text.latex.preamble"] += "\\usepackage{sfmath}" # Enable sans-serif math font

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
k = 1 # element degree
model_version = 'C'
molecule = 'D3'
tau_version = 'variable_tau'
transport_dir = f"../output/transport/results/{tau_version}/"
mesh1_input_filename = f"../output/flow/checkpoints/{tau_version}/pressure+original/model_{model_version}/velocity_data_dt=0.02252"
mesh2_input_filename = f"../output/flow/checkpoints/{tau_version}/pressure+shrunk/model_{model_version}/velocity_data_dt=0.02252"
mesh3_input_filename = f"../output/flow/checkpoints/{tau_version}/pressure+middle_shrunk/model_{model_version}/velocity_data_dt=0.02252"
mesh4_input_filename = f"../output/flow/checkpoints/{tau_version}/pressure+hind_shrunk/model_{model_version}/velocity_data_dt=0.02252"
mesh5_input_filename = f"../output/flow/checkpoints/{tau_version}/pressure+fore_middle_hind_shrunk/model_{model_version}/velocity_data_dt=0.02252"
mesh1 = a4d.read_mesh(comm=comm, filename=mesh1_input_filename, engine="BP4", ghost_mode=gm)
mesh2 = a4d.read_mesh(comm=comm, filename=mesh2_input_filename, engine="BP4", ghost_mode=gm)
mesh3 = a4d.read_mesh(comm=comm, filename=mesh3_input_filename, engine="BP4", ghost_mode=gm)
mesh4 = a4d.read_mesh(comm=comm, filename=mesh4_input_filename, engine="BP4", ghost_mode=gm)
mesh5 = a4d.read_mesh(comm=comm, filename=mesh5_input_filename, engine="BP4", ghost_mode=gm)

#-----------------------------------------------------#
#----- Create meshtags and calculate ROI volumes -----#
#-----------------------------------------------------#
# For the original mesh
ct1, ROI_tags = create_ventricle_volumes_meshtags(mesh1)
dx1 = ufl.Measure('dx', domain=mesh1, subdomain_data=ct1)
volumes1 = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx1(tag))), op=MPI.SUM) for tag in ROI_tags]

# For the fore shrunk mesh
ct2, _ = create_ventricle_volumes_meshtags(mesh2)
dx2 = ufl.Measure('dx', domain=mesh2, subdomain_data=ct2)
volumes2 = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx2(tag))), op=MPI.SUM) for tag in ROI_tags]

# For the middle shrunk mesh
ct3, _ = create_ventricle_volumes_meshtags(mesh3)
dx3 = ufl.Measure('dx', domain=mesh3, subdomain_data=ct3)
volumes3 = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx3(tag))), op=MPI.SUM) for tag in ROI_tags]

# For the hind shrunk mesh
ct4, _ = create_ventricle_volumes_meshtags(mesh4)
dx4 = ufl.Measure('dx', domain=mesh4, subdomain_data=ct4)
volumes4 = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx4(tag))), op=MPI.SUM) for tag in ROI_tags]

# For the fore+middle+hind shrunk mesh
ct5, _ = create_ventricle_volumes_meshtags(mesh5)
dx5 = ufl.Measure('dx', domain=mesh5, subdomain_data=ct5)
volumes5 = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx5(tag))), op=MPI.SUM) for tag in ROI_tags]

# Add ROI 1, 2, 3 volumes to ROI 4 volume
for volume in [volumes1, volumes2, volumes3, volumes4, volumes5]:
    volume[3] += volume[2]+volume[1]+volume[0] 

# Load data c_bar, the total concentration in each ROI
with open(transport_dir+f"original/log_model_{model_version}_{molecule}_DG1_pressureBC/data/c_hats.npy", 'rb') as file: c_bar1 = np.load(file)
with open(transport_dir+f"shrunk/log_model_{model_version}_{molecule}_DG1_pressureBC/data/c_hats.npy", 'rb') as file: c_bar2 = np.load(file)
with open(transport_dir+f"middle_shrunk/log_model_{model_version}_{molecule}_DG1_pressureBC/data/c_hats.npy", 'rb') as file: c_bar3 = np.load(file)
with open(transport_dir+f"hind_shrunk/log_model_{model_version}_{molecule}_DG1_pressureBC/data/c_hats.npy", 'rb') as file: c_bar4 = np.load(file)
with open(transport_dir+f"fore_middle_hind_shrunk/log_model_{model_version}_{molecule}_DG1_pressureBC/data/c_hats.npy", 'rb') as file: c_bar5 = np.load(file)

# Divide the total concentrations in the ROIs by the volume of the respective ROI
for i in ROI_tags:
    c_bar1[:, i-1] /= volumes1[i-1]
    c_bar2[:, i-1] /= volumes2[i-1]
    c_bar3[:, i-1] /= volumes3[i-1]
    c_bar4[:, i-1] /= volumes4[i-1]
    c_bar5[:, i-1] /= volumes5[i-1]
    print(f"Tag {i}")
    print(f"Final concentrations: ", [c_bar[-1, i-1] for c_bar in [c_bar1, c_bar2, c_bar3, c_bar4, c_bar5]])

# Get the number of timesteps
num_timesteps = c_bar1.shape[0]
f = 2.22
dt = 1/f/20
times = dt*np.arange(num_timesteps)

#-------- Plot the figures --------#
fig_c, ax_c = plt.subplots(num=1, nrows=2, ncols=3, figsize=[20, 12]) # Concentration curves
lw = 2.5 # linewidth
msize = 10 # markersize
c_str = r'$\overline{c}$'
t_str = r'$\hat{t}$'
save_figs = 1
import colormaps as cm
green = cm.dark2_3.colors[0]
orange = cm.dark2_3.colors[1]
purple = cm.puor_4.colors[3]
yellow = cm.puor_4.colors[1]
colors = ['k', green, purple, orange, yellow]
print([colors[i]*255 for i in range(1, len(colors))])

# Plot concentrations
idx = 0
for row_idx in range(ax_c.shape[0]):
    for col_idx in range(ax_c.shape[1]):
        tag = ROI_tags[idx]
        ca_c = ax_c[row_idx, col_idx]
    
        ca_c.plot(times, c_bar1[:, idx], color=colors[0], label='Original', linewidth=lw)
        ca_c.plot(times, c_bar2[:, idx], color=colors[1], linestyle='--', marker='o', markevery=2000, label='Shrunk fore-mid', linewidth=lw, markersize=msize)
        ca_c.plot(times, c_bar3[:, idx], color=colors[2], linestyle='--', marker='^', markevery=2000, label='Shrunk middle', linewidth=lw, markersize=msize)
        ca_c.plot(times, c_bar4[:, idx], color=colors[3], linestyle='--', marker='s', markevery=2000, label='Shrunk mid-hind', linewidth=lw, markersize=msize)
        ca_c.plot(times, c_bar5[:, idx], color=colors[4], linestyle='--', marker='p', markevery=2000, label='Fore+middle+hind', linewidth=lw, markersize=msize)

        ca_c.set_title(f'ROI {tag}', fontsize=30, pad=2.0)
        ca_c.set_xticks([0, 250, 500, 750])
        ca_c.tick_params(labelsize=30)
        
        if col_idx==0: ca_c.set_ylabel(r"Mean conc. $\overline{c}$ [-]", fontsize=32)    
        if row_idx==(ax_c.shape[0]-1): ca_c.set_xlabel("Time [s]", fontsize=35, labelpad=25)
        if tag==1: ca_c.legend(loc='lower right', fontsize=22, frameon=True, fancybox=False, edgecolor='k')

        idx += 1

plt.tight_layout()
plt.subplots_adjust(hspace=0.30)
if save_figs: fig_c.savefig(f"../output/illustrations/compare_geometry/all_ROIs_model{model_version}.png")
plt.show()