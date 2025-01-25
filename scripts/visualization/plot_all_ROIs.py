import vtk # Needed for pyvista
import ufl

import numpy   as np
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py       import MPI
from imports.mesh import create_ventricle_volumes_meshtags

# Set latex text properties
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif"
})
comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
k = 1 # element degree
model_version = 'C'
molecule = 'D3'
mesh_version = 'original'
mesh_input_filename = f"output/flow/checkpoints/reversed_tau/pressure+{mesh_version}/model_{model_version}/velocity_data_dt=0.02252"
mesh = a4d.read_mesh(comm=comm, filename=mesh_input_filename, engine="BP4", ghost_mode=gm)

# Create meshtags and calculate ROI volumes
mt, ROI_tags = create_ventricle_volumes_meshtags(mesh)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=mt)
volumes = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(tag))), op=MPI.SUM) for tag in ROI_tags]

# Load data c_bar, the total concenctration in each ROI
with open(f"./output/transport/results/reversed_tau/{mesh_version}/log_model_{model_version}_{molecule}_DG1_pressureBC/injection_site_middle_dorsal_posterior/data/c_hats.npy", "rb") as file:
# with open("output/transport/original+yrange_ROI1+flux_mod/log_model_C_D3_DG1_ALE_flux_BC_off/injection_site_middle_dorsal_posterior/data/c_hats.npy", "rb") as file:    
    c_bars = np.load(file)

# Scale the total concentrations in the ROIs by the volume of the respective ROI
for i in ROI_tags: 
    c_bars[:, i-1] /= volumes[i-1]

# Get the number of timesteps
num_timesteps = c_bars.shape[0]
c_threshold = 0.25 # threshold value to be used to calculate "time to threshold"
f = 2.22
dt = 1/f/20
times = dt*np.arange(num_timesteps)

# Plot the figures
fig, ax = plt.subplots(num=1, figsize=([11, 9]))

lw = 4 # linewidth
colors = np.array([
    [0, 114, 189], # Blue
    [126, 47, 142], # Purple
    [162, 20, 47], # Red
    [237, 177, 32], # Yellow
    [77, 190, 238], # Cyan
    [217, 83, 25], # Orange
    [119, 172, 48] # Green
]) / 255 # Normalize to [0, 1] range
[ax.plot(times, c_bars[:, i-1], label=f'ROI {i}', color=colors[i-1], linewidth=lw) for i in ROI_tags]

if model_version=='A':
    title_str = 'Model I (cilia, no heartbeat)'
elif model_version=='B':
    title_str = 'Model II (heartbeat, no cilia)'
elif model_version=='C':
    title_str = 'Model 0 (cilia+heartbeat)'
else:
    raise ValueError('Model version must be A, B or C.')
ax.set_title(title_str, fontsize=30)
ax.set_xlabel("Time [s]", fontsize=25)
ax.set_ylabel("Mean concentration [-]", fontsize=25)
ax.legend(loc='upper left', fontsize=20, frameon=True, fancybox=False, edgecolor='k')
ax.tick_params(labelsize=15)
fig.tight_layout()
save_fig = 0
if save_fig: fig.savefig(f"output/illustrations/concentration_all_ROIs_model{model_version}_{molecule}.png")
plt.show()

# Calculate times to threshold
t_hats = np.array([0]*len(ROI_tags), dtype=np.float64)
# define the first time-instant where c_threshold is exceeded
# as the "time to reach threshold" 
for j in range(len(ROI_tags)):
    t_hat = np.where(c_bars[:, j] > c_threshold)[0][0]
    t_hats[j] = t_hat
t_hats *= dt

msize = 4
fig2, ax2 = plt.subplots(num=4, figsize=[10, 7])
ax2.scatter(ROI_tags, t_hats, linewidths=msize)
ax2.set_title(f"Time to reach threshold, model {model_version}", fontsize=30)
ax2.set_xlabel("ROI number", fontsize=25)
ax2.set_ylabel("Time [s]", fontsize=25)
ax2.tick_params(labelsize=15)
if save_fig: fig2.savefig(f"output/illustrations/time_to_threshold_all_ROIs_model{model_version}_{molecule}.png")
plt.show()