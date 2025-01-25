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

dt = 0.02252
n_1 = 4
n_2 = 8
n_3 = 12
n_4 = 16
t_1 = n_1*dt
t_2 = n_2*dt
t_3 = n_3*dt
t_4 = n_4*dt

# Read velocity data
filename = f"../output/checkpoints/pressure/model_C/velocity_data_dt={dt}"
mesh = a4d.read_mesh(comm=comm, file=filename, engine="BP4", ghost_mode=gm)
W = dfx.fem.functionspace(mesh=mesh, element=element("DG", mesh.basix_cell(), 1, shape=(3,))) # DG1 vector function space
u = dfx.fem.Function(W)
a4d.read_function(u=u, filename=filename, engine="BP4", time=n_1)

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
a = [0.201881, 0.1396, 0.135443]
b = [0.19654, 0.152511, 0.243321]
data = grid.sample_over_line(pointa=a, pointb=b)
z_start = data.bounds[4]
z_end = data.bounds[5]
z_coords = np.linspace(z_start, z_end, len(data.active_scalars))
z_coords *= 1000 # scale to micrometers
interp = interp1d(z_coords, data.active_scalars, kind='cubic')

# Plot
fig, ax = plt.subplots(figsize=([12, 8]))
ax.plot(interp(z_coords), z_coords, color='b', linewidth=3, label=rf'Time $t={t_1:.2g}$')
ax.plot(np.zeros_like(z_coords), z_coords, 'k--')
ax.set_title(rf"Velocity $u_x$ as a function of $z$", fontsize=40)
ax.set_xlabel(r"Velocity $u_x(z)$ [$\mu$m/s]", fontsize=30)
ax.set_ylabel(r"$z$ coordinate [$\mu$m]", fontsize=30)
ax.set_ylim([np.min(z_coords), np.max(z_coords)])


# New timestep
a4d.read_function(u=u, filename=filename, engine="BP4", time=n_2)

# Interpolate velocity into continuous P2 space
V = dfx.fem.functionspace(mesh=mesh, element=element("Lagrange", mesh.basix_cell(), 2, shape=(3,)))
v = dfx.fem.Function(V)
v.interpolate(u)

u_x = v.sub(0).collapse().x.array
u_x *= 1000 # scale to micrometers/s
grid.point_data["u_x"] = u_x
grid.set_active_scalars("u_x")

data = grid.sample_over_line(pointa=a, pointb=b)
z_start = data.bounds[4]
z_end = data.bounds[5]
z_coords = np.linspace(z_start, z_end, len(data.active_scalars))
z_coords *= 1000 # scale to micrometers
interp = interp1d(z_coords, data.active_scalars, kind='cubic')
ax.plot(interp(z_coords), z_coords, color='r', linewidth=3, label=rf'Time $t={t_2:.2g}$')

# New timestep
a4d.read_function(u=u, filename=filename, engine="BP4", time=n_3)

# Interpolate velocity into continuous P2 space
V = dfx.fem.functionspace(mesh=mesh, element=element("Lagrange", mesh.basix_cell(), 2, shape=(3,)))
v = dfx.fem.Function(V)
v.interpolate(u)

u_x = v.sub(0).collapse().x.array
u_x *= 1000 # scale to micrometers/s
grid.point_data["u_x"] = u_x
grid.set_active_scalars("u_x")

data = grid.sample_over_line(pointa=a, pointb=b)
z_start = data.bounds[4]
z_end = data.bounds[5]
z_coords = np.linspace(z_start, z_end, len(data.active_scalars))
z_coords *= 1000 # scale to micrometers
interp = interp1d(z_coords, data.active_scalars, kind='cubic')
ax.plot(interp(z_coords), z_coords, color='g', linewidth=3, label=rf'Time $t={t_3:.2g}$')

# New timestep
a4d.read_function(u=u, filename=filename, engine="BP4", time=n_4)

# Interpolate velocity into continuous P2 space
V = dfx.fem.functionspace(mesh=mesh, element=element("Lagrange", mesh.basix_cell(), 2, shape=(3,)))
v = dfx.fem.Function(V)
v.interpolate(u)

u_x = v.sub(0).collapse().x.array
u_x *= 1000 # scale to micrometers/s
grid.point_data["u_x"] = u_x
grid.set_active_scalars("u_x")

data = grid.sample_over_line(pointa=a, pointb=b)
z_start = data.bounds[4]
z_end = data.bounds[5]
z_coords = np.linspace(z_start, z_end, len(data.active_scalars))
z_coords *= 1000 # scale to micrometers
interp = interp1d(z_coords, data.active_scalars, kind='cubic')
ax.plot(interp(z_coords), z_coords, color='k', linewidth=3, label=rf'Time $t={t_4:.2g}$')

ax.legend(fontsize=25)
ax.tick_params(labelsize=25)


plt.tight_layout()
fig.savefig("../output/illustrations/velocity_x_different_times.png")
plt.show()