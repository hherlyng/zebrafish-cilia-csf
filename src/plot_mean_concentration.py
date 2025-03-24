import ufl

import numpy   as np
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py       import MPI
from scipy.io     import loadmat
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
colors = loadmat("../data/data_photoconversion/aggregated_data/colors")['color'] # The colors used for plotting

# Directories
flow_dir = '../output/flow/checkpoints/'
transport_dir = '../output/transport/results/'

# Problem version
model_version = 'C'
molecule = 'D2'
cilia_string = 'all_cilia'
mesh_version = 'original'
output_dir = f'../output/illustrations/'
f = 2.22
dt = 1/f/20

# Read mesh
# mesh_input_filename = flow_dir+f"{tau_version}/pressure+neighbor_refined/model_{model_version}/velocity_data_dt=0.02252"
mesh_input_filename = f'../output/flow/checkpoints/velocity_mesh={mesh_version}_model={model_version}_ciliaScenario={cilia_string}_dt={dt:.4g}'
mesh = a4d.read_mesh(comm=comm, filename=mesh_input_filename, engine="BP4", ghost_mode=gm)

# Create meshtags and calculate ROI volumes
mt, ROI_tags = create_ventricle_volumes_meshtags(mesh)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=mt)
volumes = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(tag))), op=MPI.SUM) for tag in ROI_tags]
volumes[3] += volumes[2]+volumes[1]+volumes[0] # Add ROI 1, 2, 3 volumes to ROI 4 volume

# ROI concentration thresholds
c_threshold1 = 0.25 # threshold value used to calculate "time to threshold" in ROIs 1-4
c_threshold2 = 0.10 # threshold value used to calculate "time to threshold" in ROIs 5-6

# Plot parameters
save_fig = 0 # Save figures if set to 1, don't save if set to 0
lw = 6 # linewidth

# Loop over molecules
transport_data_filename = '../output/transport/mesh=original_model=C_molecule=D3_ciliaScenario=all_cilia_dt=0.02252/data/c_hats.npy'
# transport_data_filename = transport_dir + \
#                 f"{tau_version}/{mesh_version}/log_model_{model_version}_{molecule}_DG1_pressureBC/data/c_hats.npy"

# Load transport data: c_bars = the total concentration in each ROI
with open(transport_data_filename, "rb") as file: c_bars = np.load(file) 

# Scale the total concentrations in the ROIs by the volume of the respective ROI
for i in ROI_tags: c_bars[:, i-1] /= volumes[i-1]
print("c_bar at final time: ", c_bars[-1, :])

# Get the number of timesteps
num_timesteps = c_bars.shape[0]
times = dt*np.arange(num_timesteps)

# Plot the figures
fig, ax = plt.subplots(figsize=([18, 14]))

[ax.plot(times, c_bars[:, i-1], label=f'ROI {i}', color=colors[i-1], linewidth=lw) for i in ROI_tags]

# Add labels
ax.set_xlabel("Time [s]", fontsize=60, labelpad=25)
ax.set_ylabel(r"Mean concentration $\overline{c}$ [-]", fontsize=60, labelpad=25)
ax.tick_params(labelsize=60)
fig.tight_layout()

if save_fig: fig.savefig(f"{output_dir}/simulations_time_evolution_model{model_version}_{molecule}_no_legend.png")
plt.show()