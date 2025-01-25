import vtk # Needed for pyvista
import ufl
import time

import numpy   as np
import pyvista as pv
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py    import MPI
from basix.ufl import element

# Set latex text properties
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif"
})

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
k = 1 # element degree
mesh_input_filename = "output/checkpoints/pressure/model_C/velocity_data_dt=0.02252"#f"./output/checkpoints/transport/DG{k}/pressure/model_C/D3/concentration_inj=middle_dorsal_posterior"
mesh = a4d.read_mesh(comm=comm, file=mesh_input_filename, engine="BP4", ghost_mode=gm)

DEFAULT   = 2
MIDDLE    = 3
ANTERIOR  = 4
POSTERIOR = 5

def create_ventricle_volumes_meshtags(mesh: dfx.mesh.Mesh):
    z_min = mesh.geometry.x[:, 2].min()
    z_max = mesh.geometry.x[:, 2].max()
    z_mid = (z_min + z_max) / 2
    a1 = [0.165, 0.0, z_mid]
    b1 = [0.125, 0.0, z_max*0.78]

    a2 = [0.33, 0.0, z_mid*0.8]
    b2 = [0.345, 0.0, z_max*0.75]

    line1 = lambda t: a1[2] + (b1[2] - a1[2])/(b1[0] - a1[0])*(t - a1[0])
    line2 = lambda t: a2[2] + (b2[2] - a2[2])/(b2[0] - a2[0])*(t - a2[0])

    def middle_ventricle(x):
        return np.logical_and(x[2] >= line1(x[0]), x[2] >= line2(x[0]))

    def anterior_ventricle(x):
        return x[2] < line1(x[0])

    def posterior_ventricle(x):
        return x[2] < line2(x[0])

    tdim = mesh.topology.dim
    ## TODO: update to using midpoint coordinate of cells located
    middle_cells = dfx.mesh.locate_entities(mesh, tdim, middle_ventricle)
    anterior_cells = dfx.mesh.locate_entities(mesh, tdim, anterior_ventricle)
    posterior_cells = dfx.mesh.locate_entities(mesh, tdim, posterior_ventricle)

    num_volumes   = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts # Total number of volumes
    volume_marker = np.full(num_volumes, DEFAULT, dtype=np.int32) # Default volume marker value
    volume_marker[middle_cells] = MIDDLE
    volume_marker[anterior_cells] = ANTERIOR
    volume_marker[posterior_cells] = POSTERIOR

    return dfx.mesh.meshtags(mesh, tdim, np.arange(num_volumes, dtype=np.int32), volume_marker)
# with dfx.io.XDMFFile(MPI.COMM_WORLD, 'lines_check.xdmf', 'w') as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_meshtags(mt, mesh.geometry)
#     exit()
mt = create_ventricle_volumes_meshtags(mesh=mesh)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=mt)
vol_middle = MPI.COMM_WORLD.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(MIDDLE))), op=MPI.SUM)
vol_anterior = MPI.COMM_WORLD.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(ANTERIOR))), op=MPI.SUM)
vol_posterior = MPI.COMM_WORLD.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(POSTERIOR))), op=MPI.SUM)

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
num_timesteps = 20
arrays = [np.zeros((6, num_timesteps)) for _ in range(3)]

tic = time.perf_counter()
# for j in range(num_timesteps):
#     for i in [2]:#range(3):    
#         input_filename = f"./output/checkpoints/transport/DG{k}/pressure/model_C/D{i+1}/concentration_inj=middle_dorsal_posterior"
#         a4d.read_function(u=c_in, filename=input_filename, engine="BP4", time=j)

#         c_mean_mid  = 1/vol_middle*dfx.fem.assemble_scalar(dfx.fem.form(c_in*dx(MIDDLE)))
#         c_mean_ant  = 1/vol_anterior*dfx.fem.assemble_scalar(dfx.fem.form(c_in*dx(ANTERIOR)))
#         c_mean_post = 1/vol_posterior*dfx.fem.assemble_scalar(dfx.fem.form(c_in*dx(POSTERIOR)))
        
#         arrays[i][0][j] = c_mean_mid
#         arrays[i][1][j] = c_mean_ant
#         arrays[i][2][j] = c_mean_post
# print(f"Read for-loop time: {time.perf_counter()-tic:.2f} sec")

# # Save arrays to file
# with open('new_mean_c_arrays_varying_D.npy', 'wb') as file:
#     np.save(file, arrays)

# # load data
# with open('new_mean_c_arrays_varying_D.npy', 'rb') as file:
#     arrays = np.load(file)
# with open('output/results-model_C/DG1/D3/injection_site_middle_dorsal_posterior/c_means.npy', 'rb') as file:
#     arrays_B = np.load(file)
# with open('output/results-model_C/DG1/D3/injection_site_middle_dorsal_posterior/c_means.npy', 'rb') as file:
#     arrays_C = np.load(file)
with open('output/results-model_C/DG1/D1/injection_site_anterior_dorsal/c_hats.npy', 'rb') as file:
    arrays_B = np.load(file)
with open('output/results-model_C/DG1/D1/injection_site_anterior_dorsal/c_hats.npy', 'rb') as file:
    arrays_C = np.load(file)
# with open('pure_diffusion_c_means.npy', 'rb') as file:
#     arrays = np.load(file)
# Prepare data by scaling the total concentrations c_hat by the maximum value of c_hat
c_hat_max_B = np.max(arrays_B)
arrays_B /= c_hat_max_B
c_hat_max_C = np.max(arrays_C)
arrays_C /= c_hat_max_C
c_threshold = 0.3 # threshold value to be used to calculate "time to threshold"
f = 2.22
dt = 1/f/20
times = dt*np.arange(num_timesteps)
from IPython import embed;embed()

# Plot the figures
separate_plots = True
if separate_plots:
    fig1, ax1 = plt.subplots(num=1, figsize=([12, 8]))
    fig2, ax2 = plt.subplots(num=2, figsize=([12, 8]))

    lw = 2 # linewidth
    ax1.plot(times, arrays_B[:, 0], color=(0.6, 0.752941, 0.145098), label='middle', linewidth=lw)
    ax1.plot(times, arrays_B[:, 1], color=(0.196078, 0.501961, 0), label='anterior', linewidth=lw)
    ax1.plot(times, arrays_B[:, 2], color='k', label='posterior', linewidth=lw)
    ax1.set_title(r'Model B (no cilia, only heartbeat)', fontsize=25)
    ax2.plot(times, arrays_C[:, 0], color=(0.6, 0.752941, 0.145098), label='middle', linewidth=lw)
    ax2.plot(times, arrays_C[:, 1], color=(0.196078, 0.501961, 0), label='anterior', linewidth=lw)
    ax2.plot(times, arrays_C[:, 2], color='k', label='posterior', linewidth=lw)
    ax2.set_title(r'Model C (cilia+heartbeat)', fontsize=25)
    [ax.plot(times, c_threshold*np.ones(len(times)), color='0.3', linestyle='dashed', linewidth=lw) for ax in [ax1, ax2]]
    [ax.set_xlabel(r"$t$ [s]", fontsize=25) for ax in [ax1, ax2]]
    [ax.set_ylabel(r"$\overline{c}$ [-]", fontsize=25) for ax in [ax1, ax2]]
    [ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, frameon=False) for ax in [ax1, ax2]]
    [ax.tick_params(labelsize=20) for ax in [ax1, ax2]]
    plt.tight_layout()
    # fig1.savefig("./output/illustrations/mean_concentration_profiles_D1.png")
    # fig2.savefig("./output/illustrations/mean_concentration_profiles_D2.png")
    # fig3.savefig("./output/illustrations/mean_concentration_profiles_D3.png")
    plt.show()
else:
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=([8, 10]), sharex=True)
    lw = 2 # linewidth
    ax[0].plot(times, arrays[0][0], color=(0.6, 0.752941, 0.145098), label='middle', linewidth=lw)
    ax[0].plot(times, arrays[0][1], color=(0.196078, 0.501961, 0), label='anterior', linewidth=lw)
    ax[0].plot(times, arrays[0][2], color='k', label='posterior', linewidth=lw)
    ax[0].set_title(r'$D_1$', fontsize=25)
    ax[1].plot(times, arrays[1][0], color=(0.6, 0.752941, 0.145098), label='middle', linewidth=lw)
    ax[1].plot(times, arrays[1][1], color=(0.196078, 0.501961, 0), label='anterior', linewidth=lw)
    ax[1].plot(times, arrays[1][2], color='k', label='posterior', linewidth=lw)
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=20, frameon=False)
    ax[1].set_title(r'$D_2$', fontsize=25)
    ax[2].plot(times, arrays[:, 0], color=(0.6, 0.752941, 0.145098), label='middle', linewidth=lw)
    ax[2].plot(times, arrays[:, 1], color=(0.196078, 0.501961, 0), label='anterior', linewidth=lw)
    ax[2].plot(times, arrays[:, 2], color='k', label='posterior', linewidth=lw)
    ax[2].set_title(r'$D_3$', fontsize=25)
    ax[2].set_xlabel(r"$t$ [s]", fontsize=25)
    plt.tight_layout()
    plt.show()


c_hats1 = np.where(arrays[0] > c_threshold)[0]
c_hats2 = np.where(arrays[1] > c_threshold)[0]
c_hats3 = np.where(arrays[2] > c_threshold)[0]

t_hats = np.array([[0, 0, 0],
                   [0, 0, 0]],
                   dtype=np.float64)

# define the first time-instant where c_threshold is exceeded
# as the "time to reach threshold" 
for idx, c_hat in enumerate([c_hats1, c_hats2, c_hats3]):
    t_hats[idx] = c_hat[0] 

# print(f"Time to reach threshold D1: {t_hats[0]*dt:.2e}")
# print(f"Time to reach threshold D2: {t_hats[1]*dt:.2e}")
# print(f"Time to reach threshold D3: {t_hats[2]*dt:.2e}")

# from IPython import embed;embed()