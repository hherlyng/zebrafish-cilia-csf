import vtk # Needed for pyvista
import ufl

import numpy   as np
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py       import MPI
from basix.ufl    import element
from utilities.mesh import create_ventricle_volumes_meshtags

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
mesh_input_filename = "../output/flow/checkpoints/pressure+original/model_C/velocity_data_dt=0.02252"
mesh = a4d.read_mesh(comm=comm, filename=mesh_input_filename, engine="BP4", ghost_mode=gm)

# Get mesh properties
x_geo = mesh.geometry.x

x_min = x_geo[:, 0].min()
x_max = x_geo[:, 0].max()
x_mid = (x_min + x_max) / 2

y_max = x_geo[:, 1].max()

z_min = x_geo[:, 2].min()
z_max = x_geo[:, 2].max()
z_mid = (z_min + z_max) / 2

# Set up unstructured grids with data
W = dfx.fem.functionspace(mesh=mesh, element=element("DG", mesh.basix_cell(), k)) # DG1 function space
c_in = dfx.fem.Function(W) # Concentration finite element function
cells, types, x = dfx.plot.vtk_mesh(W)

# Create meshtags and calculate ROI volumes
mt, ROI_tags = create_ventricle_volumes_meshtags(mesh, mod=True)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=mt)
volumes_mod = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(tag))), op=MPI.SUM) for tag in ROI_tags]
mt, ROI_tags = create_ventricle_volumes_meshtags(mesh, mod=False)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=mt)
volumes = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(tag))), op=MPI.SUM) for tag in ROI_tags]

# Load data c_hat, the total concenctration in each ROI
# with open(f'output/transport/model_B_{molecule}_DG1_pressureBC/injection_site_middle_dorsal_posterior/npy_data/c_hats.npy', 'rb') as file:
#     c_hats_B = np.load(file)
with open("../output/transport/original+mod_ROI/log_model_C_D3_DG1_pressureBC/injection_site_middle_dorsal_posterior/npy_data/c_hats.npy", "rb") as file:
    c_hats_B = np.load(file)
with open(f"../output/transport/original/log_model_C_D3_DG1_pressureBC/injection_site_middle_dorsal_posterior/npy_data/c_hats.npy", "rb") as file:
    c_hats_C = np.load(file)

# Scale the total concentrations in the ROIs by the volume of the respective ROI
for i in ROI_tags: 
    c_hats_B[:, i-1] /= volumes_mod[i-1]
    c_hats_C[:, i-1] /= volumes[i-1]

# Get the number of timesteps
num_timesteps = c_hats_B.shape[0]
c_threshold = 0.25 # threshold value to be used to calculate "time to threshold"
f = 2.22
dt = 1/f/20
times = dt*np.arange(num_timesteps)

#-----------------------------------------------#
# Calculate times to threshold
t_hats_B = np.array([0]*len(ROI_tags), dtype=np.float64)
t_hats_C = np.array([0]*len(ROI_tags), dtype=np.float64)

# define the first time-instant where c_threshold is exceeded
# as the "time to reach threshold" 
for i in range(len(t_hats_B)):
    t_hat_B = np.where(c_hats_B[:, i] > c_threshold)[0][0]
    t_hats_B[i] = t_hat_B

    t_hat_C = np.where(c_hats_C[:, i] > c_threshold)[0][0]
    t_hats_C[i] = t_hat_C

t_hats_B *= dt
t_hats_C *= dt


# Plot the figures
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[9, 10])
msize = 4 # markersize
t_str = r'$\hat{t}$'

for idx, tag in enumerate(ROI_tags):
    if tag % 2 != 0:
        # Odd number -> plot in 1st column
        row_idx = int((tag-1)/2)
        col_idx = 0
    else:
        # Even number -> plot in 2nd column
        row_idx = int((tag-2)/2)
        col_idx = 1
    ca = ax[row_idx, col_idx]
    # Plot
    ca.scatter(0.75, t_hats_B[idx], color='r', label='model B', linewidths=msize)
    ca.scatter(0.25, t_hats_C[idx], color='k', label='model C', linewidths=msize)
    ca.set_xticks([0, 0.25, 0.75, 1])
    ca.set_xticklabels(['', 'Baseline model', 'Cardiac-only', ''])
    ca.set_title(rf'{t_str} in ROI {tag}', fontsize=25)
    ca.tick_params(labelsize=15)
    ca.set_ylabel(r"Time [s]", fontsize=20)
    ca.set_ylim(0, ca.get_ylim()[1]*1.25)

fig.tight_layout()
save_fig = 0
if save_fig: fig.savefig(f"ROI_time_to_threshold_{molecule}.png")
plt.show()