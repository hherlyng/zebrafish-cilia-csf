import ufl

import numpy   as np
import pandas  as pd
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py       import MPI
from scipy.io     import loadmat
from imports.mesh import create_ventricle_volumes_meshtags

# Set matplotlib properties
plt.rcParams.update({
    "text.usetex" : True,
    "font.family" : "sans-serif",
    "axes.spines.top" : False,
    "axes.spines.right" : False
})
plt.rcParams["text.latex.preamble"] += "\\usepackage{sfmath}" # Enable sans-serif math font

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
colors = loadmat("./data_photoconversion/aggregated_data/colors")['color'] # The colors used for plotting
fig_num = 1 # Figure number index

# Directories
flow_dir = './output/flow/checkpoints/'
transport_dir = './output/transport/results/'

# Problem version
model_version = 'C'
molecule = 'D3'
tau_version  = 'variable_tau'
mesh_version = 'original'
output_dir = f'output/illustrations/{mesh_version}/{tau_version}/'

# Read mesh
mesh_input_filename = flow_dir+f"{tau_version}/pressure+{mesh_version}/model_{model_version}/velocity_data_dt=0.02252"
mesh = a4d.read_mesh(comm=comm, filename=mesh_input_filename, engine="BP4", ghost_mode=gm)

# Create meshtags and calculate ROI volumes
mt, ROI_tags = create_ventricle_volumes_meshtags(mesh)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=mt)
volumes = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(tag))), op=MPI.SUM) for tag in ROI_tags]
volumes[3] += volumes[2]+volumes[1]+volumes[0] # Add ROI 1, 2, 3 volumes to ROI 4 volume

# Analysis parameters
f = 2.22
dt = 1/f/20

# Plot parameters
save_fig = 1 # Save figures if set to 1, don't save if set to 0
add_legend = 1 # Adds legend if set to 1, else no legend
lw = 6 # linewidth
msize = 4 # marker size

# Loop over molecules
transport_data_filename = transport_dir + \
    f"{tau_version}/{mesh_version}/log_model_{model_version}_{molecule}_DG1_pressureBC/data/c_hats.npy"

# Load transport data: c_bars = the total concentration in each ROI
with open(transport_data_filename, "rb") as file: c_bars = np.load(file) 

# Scale the total concentrations in the ROIs by the volume of the respective ROI
for i in ROI_tags: c_bars[:, i-1] /= volumes[i-1]

# Get the number of timesteps
num_timesteps = c_bars.shape[0]
times = dt*np.arange(num_timesteps)

# Plot the figures
fig, ax = plt.subplots(num=fig_num, figsize=([18, 14]))

[ax.plot(times, c_bars[:, i-1], label=f'ROI {i}', color=colors[i-1], linewidth=lw) for i in ROI_tags]

ax.set_xlabel("Time [s]", fontsize=60, labelpad=25)
# Add legend and y-axis label
if add_legend: ax.legend(loc='best', fontsize=50, frameon=True, fancybox=False, edgecolor='k')
ax.set_ylabel(r"Mean concentration $\overline{c}$ [-]", fontsize=60, labelpad=25)
ax.tick_params(labelsize=60)
fig.tight_layout()

if save_fig: fig.savefig(f"{output_dir}/simulations_concentration_model{model_version}_{molecule}.png")
plt.show()