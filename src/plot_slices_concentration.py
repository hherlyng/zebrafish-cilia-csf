import vtk # Needed for pyvista

import numpy   as np
import pyvista as pv
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py    import MPI
from basix.ufl import element
from utilities.create_colormap import register_colormaps

""" Visualize concentration slices at various time instants. """

# Set latex text properties
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
k = 1 # element degree
model_version = 'C'
tau_version = 'variable_tau'
input_filename1 = f"../output/transport/results/{tau_version}/original/log_model_{model_version}_D1_DG1_pressureBC/checkpoints/concentration_snapshots/"
input_filename2 = f"../output/transport/results/{tau_version}/original/log_model_{model_version}_D2_DG1_pressureBC/checkpoints/concentration_snapshots/"
input_filename3 = f"../output/transport/results/{tau_version}/original/log_model_{model_version}_D3_DG1_pressureBC/checkpoints/concentration_snapshots/"
mesh1 = a4d.read_mesh(comm=comm, filename=input_filename1, engine="BP4", ghost_mode=gm)
mesh2 = a4d.read_mesh(comm=comm, filename=input_filename2, engine="BP4", ghost_mode=gm)
mesh3 = a4d.read_mesh(comm=comm, filename=input_filename3, engine="BP4", ghost_mode=gm)

# Get mesh properties
x_geo = mesh1.geometry.x

x_min = x_geo[:, 0].min()
x_max = x_geo[:, 0].max()
x_mid = (x_min + x_max) / 2

y_max = x_geo[:, 1].max()

z_min = x_geo[:, 2].min()
z_max = x_geo[:, 2].max()
z_mid = (z_min + z_max) / 2

# Set up unstructured grids with data
W1 = dfx.fem.functionspace(mesh=mesh1, element=element("DG", mesh1.basix_cell(), k)) # DG1 function space
W2 = dfx.fem.functionspace(mesh=mesh2, element=element("DG", mesh1.basix_cell(), k)) # DG1 function space
W3 = dfx.fem.functionspace(mesh=mesh3, element=element("DG", mesh1.basix_cell(), k)) # DG1 function space
record_periods = np.array([10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750])
f = 2.22 # Cardiac frequency [Hz]
period = 1 / f # Cardiac period [s]
dt = period / 20
times = record_periods*period
vis_periods = [record_periods[2], record_periods[3], record_periods[4],
               record_periods[5], record_periods[6], record_periods[7],
               record_periods[8], record_periods[9]]

grids = []
m = 0 # grid index
for j in range(len(vis_periods)):
    for i in [1, 2, 3]:    
        if   i==1: W = W1
        elif i==2: W = W2
        elif i==3: W = W3

        cells, types, x = dfx.plot.vtk_mesh(W)
        grids.append(pv.UnstructuredGrid(cells, types, x))
        c_in = dfx.fem.Function(W)
        input_filename = f"../output/transport/results/{tau_version}/original/log_model_{model_version}_D{i}_DG1_pressureBC/checkpoints/concentration_snapshots/"
        a4d.read_function(u=c_in, filename=input_filename, engine="BP4", time=vis_periods[j])
        grids[m].point_data["c"] = c_in.x.array.real
        grids[m].set_active_scalars("c")
        m += 1 # increment

# Create plot object
plot_shape = (len(vis_periods), 3)
plotter = pv.Plotter(shape=plot_shape, window_size=[1200, 1200], border=False)

# Set colormap
register_colormaps()
cmap = "wgk"

# Set colorbar limits
last_slice = grids[-1].slice(normal=[0, -1, 0])
upper_limit = last_slice.point_data["c"].max()
clim = [0, upper_limit] # Colorbar max is max concentration

# Add grids to plotter
m = 0 # grid index
for i in range(plot_shape[0]):
    for j in range(plot_shape[1]):
        plotter.subplot(i, j)

        plotter.camera.position    = (x_mid, 0, z_mid*1.1)
        plotter.camera.focal_point = (x_mid, y_max, z_mid)
        plotter.camera.zoom(0.375)
        slice = grids[m].slice(normal=[0, -1, 0])
        plotter.add_mesh(slice, show_scalar_bar=False, cmap=cmap, clim=clim)
        m += 1

# Display the plot and save to file
save_fig = 1
if save_fig:
    output_filename = f'../output/illustrations/original/{tau_version}/concentration_slices_varying_D'
    plotter.show(screenshot=output_filename+'.png')
else:
    plotter.show()