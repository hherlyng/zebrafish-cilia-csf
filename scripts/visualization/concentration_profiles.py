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
filename = "../output/checkpoints/transport/DG1/pressure/model_C/D1/concentration_inj=middle_dorsal_posterior"
mesh = a4d.read_mesh(comm=comm, file=filename, engine="BP4", ghost_mode=gm)
W = dfx.fem.functionspace(mesh=mesh, element=element("DG", mesh.basix_cell(), 1)) # DG1 function space
u = dfx.fem.Function(W)
a4d.read_function(u=u, filename=filename, engine="BP4", time=10000)

# Interpolate velocity into continuous P2 space
V = dfx.fem.functionspace(mesh=mesh, element=element("Lagrange", mesh.basix_cell(), 2))
c = dfx.fem.Function(V)
c.interpolate(u)

a = [0.135448, 0.140393, 0.176273]
b = [0.329539, 0.140393, 0.176273]
cells, types, x = dfx.plot.vtk_mesh(V)
grid = pv.UnstructuredGrid(cells, types, x)
grid.point_data["c"] = c.x.array
grid.set_active_scalars("c")
data = grid.sample_over_line(pointa=a, pointb=b)
x_start = data.bounds[0]
x_end = data.bounds[1]
x_coords = np.linspace(x_start, x_end, len(data.active_scalars))
x_coords *= 1000 # scale to micrometers
interp = interp1d(x_coords, data.active_scalars, kind='cubic')
fig, ax = plt.subplots(figsize=([12, 8]))
ax.plot(x_coords, interp(x_coords), color='r', linewidth=3, label=r'$D_1$')


# New timestep
filename = "../output/checkpoints/transport/DG1/pressure/model_C/D2/concentration_inj=middle_dorsal_posterior"
a4d.read_function(u=u, filename=filename, engine="BP4", time=10000)

# Interpolate velocity into continuous P2 space
V = dfx.fem.functionspace(mesh=mesh, element=element("Lagrange", mesh.basix_cell(), 2))
c = dfx.fem.Function(V)
c.interpolate(u)
cells, types, x = dfx.plot.vtk_mesh(V)
grid = pv.UnstructuredGrid(cells, types, x)
grid.point_data["c"] = c.x.array
grid.set_active_scalars("c")
data = grid.sample_over_line(pointa=a, pointb=b)
x_start = data.bounds[0]
x_end = data.bounds[1]
x_coords = np.linspace(x_start, x_end, len(data.active_scalars))
x_coords *= 1000 # scale to micrometers
interp = interp1d(x_coords, data.active_scalars, kind='cubic')
ax.plot(x_coords, interp(x_coords), color='g', linewidth=3, label=r'$D_2$')

# New timestep
filename = "../output/checkpoints/transport/DG1/pressure/model_C/D3/concentration_inj=middle_dorsal_posterior"
a4d.read_function(u=u, filename=filename, engine="BP4", time=10000)

# Interpolate velocity into continuous P2 space
V = dfx.fem.functionspace(mesh=mesh, element=element("Lagrange", mesh.basix_cell(), 2))
c = dfx.fem.Function(V)
c.interpolate(u)
cells, types, x = dfx.plot.vtk_mesh(V)
grid = pv.UnstructuredGrid(cells, types, x)
grid.point_data["c"] = c.x.array
grid.set_active_scalars("c")
data = grid.sample_over_line(pointa=a, pointb=b)
x_start = data.bounds[0]
x_end = data.bounds[1]
x_coords = np.linspace(x_start, x_end, len(data.active_scalars))
x_coords *= 1000 # scale to micrometers
interp = interp1d(x_coords, data.active_scalars, kind='cubic')
ax.plot(x_coords, interp(x_coords), color='b', linewidth=3, label=r'$D_3$')


# cells, types, x = dfx.plot.vtk_mesh(V)
# grid = pv.UnstructuredGrid(cells, types, x)
# grid.point_data["c"] = c.x.array
# grid.set_active_scalars("c")
# data = grid.sample_over_line(pointa=a, pointb=b)
# x_start = data.bounds[0]
# x_end = data.bounds[1]
# x_coords = np.linspace(x_start, x_end, len(data.active_scalars))
# x_coords *= 1000 # scale to micrometers
# interp = interp1d(x_coords, data.active_scalars, kind='cubic')
# fig, ax = plt.subplots(figsize=([12, 8]))
# ax.plot(x_coords, interp(x_coords), color='r', linewidth=3, label=r'Time $t_1$')


# # New timestep
# a4d.read_function(u=u, filename=filename, engine="BP4", time=5000)

# # Interpolate velocity into continuous P2 space
# V = dfx.fem.functionspace(mesh=mesh, element=element("Lagrange", mesh.basix_cell(), 2))
# c = dfx.fem.Function(V)
# c.interpolate(u)
# cells, types, x = dfx.plot.vtk_mesh(V)
# grid = pv.UnstructuredGrid(cells, types, x)
# grid.point_data["c"] = c.x.array
# grid.set_active_scalars("c")
# data = grid.sample_over_line(pointa=a, pointb=b)
# x_start = data.bounds[0]
# x_end = data.bounds[1]
# x_coords = np.linspace(x_start, x_end, len(data.active_scalars))
# x_coords *= 1000 # scale to micrometers
# interp = interp1d(x_coords, data.active_scalars, kind='cubic')
# ax.plot(x_coords, interp(x_coords), color='g', linewidth=3, label=r'Time $t_2$')

# # New timestep
# a4d.read_function(u=u, filename=filename, engine="BP4", time=15000)

# # Interpolate velocity into continuous P2 space
# V = dfx.fem.functionspace(mesh=mesh, element=element("Lagrange", mesh.basix_cell(), 2))
# c = dfx.fem.Function(V)
# c.interpolate(u)
# cells, types, x = dfx.plot.vtk_mesh(V)
# grid = pv.UnstructuredGrid(cells, types, x)
# grid.point_data["c"] = c.x.array
# grid.set_active_scalars("c")
# data = grid.sample_over_line(pointa=a, pointb=b)
# x_start = data.bounds[0]
# x_end = data.bounds[1]
# x_coords = np.linspace(x_start, x_end, len(data.active_scalars))
# x_coords *= 1000 # scale to micrometers
# interp = interp1d(x_coords, data.active_scalars, kind='cubic')
# ax.plot(x_coords, interp(x_coords), color='b', linewidth=3, label=r'Time $t_3$')

# Annotate figure and display plot
ax.set_xlabel(r"Coordinate $x$ [$\mu$m]", fontsize=30)
ax.set_ylabel(r"Concentration $c$ [-]", fontsize=30)
ax.legend(fontsize=25)
ax.tick_params(labelsize=25)
plt.tight_layout()
fig.savefig("../output/illustrations/concentration_profiles_varying_D.png")
plt.show()