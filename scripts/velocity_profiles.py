import vtk # Needed for pyvista

import numpy   as np
import pyvista as pv
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from mpi4py    import MPI
from basix.ufl import element

# Set latex text properties
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet

# Read velocity data
filename = "../output/checkpoints/pressure/model_A/velocity_data_dt=0.02252"
mesh = a4d.read_mesh(comm=comm, file=filename, engine="BP4", ghost_mode=gm)
W = dfx.fem.functionspace(mesh=mesh, element=element("DG", mesh.basix_cell(), 1, shape=(3,))) # DG1 vector function space
u = dfx.fem.Function(W)
a4d.read_function(u=u, filename=filename, engine="BP4")

# Interpolate velocity into continuous P2 space
V = dfx.fem.functionspace(mesh=mesh, element=element("Lagrange", mesh.basix_cell(), 2, shape=(3,)))
v = dfx.fem.Function(V)
v.interpolate(u)

u_x = v.sub(0).collapse().x.array
u_x *= 1000 # scale to micrometers/s
cells, types, x = dfx.plot.vtk_mesh(V)
grid = pv.UnstructuredGrid(cells, types, x)
grid.point_data["u_x"] = u_x
grid.set_active_scalars("u_x")
a = [0.166102, 0.140979, 0.138141]
b = [0.16292, 0.149926, 0.23489]
data = grid.sample_over_line(pointa=a, pointb=b)
z_start = data.bounds[4]
z_end = data.bounds[5]
z_coords = np.linspace(z_start, z_end, len(data.active_scalars))
z_coords *= 1000 # scale to micrometers
interp = interp1d(z_coords, data.active_scalars, kind='cubic')

# Plot
fig, ax = plt.subplots(figsize=([12, 8]))
ax.plot(interp(z_coords), z_coords, color='b', linewidth=3, label='Line 1')
ax.plot(np.zeros_like(z_coords), z_coords, 'k--')
ax.set_title(rf"Velocity $u_x$ as a function of $z$", fontsize=40)
ax.set_xlabel(r"Velocity $u_x(z)$ [$\mu$m/s]", fontsize=30)
ax.set_ylabel(r"$z$ coordinate [$\mu$m]", fontsize=30)
ax.set_ylim([np.min(z_coords), np.max(z_coords)])

a = [0.201881, 0.1396, 0.135443]
b = [0.19654, 0.152511, 0.243321]
data = grid.sample_over_line(pointa=a, pointb=b)
z_start = data.bounds[4]
z_end = data.bounds[5]
z_coords = np.linspace(z_start, z_end, len(data.active_scalars))
z_coords *= 1000 # scale to micrometers
interp = interp1d(z_coords, data.active_scalars, kind='cubic')
ax.plot(interp(z_coords), z_coords, color='r', linewidth=3, label='Line 2')

a = [0.243276, 0.13789, 0.118939]
b = [0.238356, 0.155721, 0.244081]
data = grid.sample_over_line(pointa=a, pointb=b)
z_start = data.bounds[4]
z_end = data.bounds[5]
z_coords = np.linspace(z_start, z_end, len(data.active_scalars))
z_coords *= 1000 # scale to micrometers
interp = interp1d(z_coords, data.active_scalars, kind='cubic')
ax.plot(interp(z_coords), z_coords, color='g', linewidth=3, label='Line 3')

ax.legend(fontsize=25)
ax.tick_params(labelsize=25)


plt.tight_layout()
fig.savefig("../output/illustrations/velocity_x_in_z_direction.png")
plt.show()