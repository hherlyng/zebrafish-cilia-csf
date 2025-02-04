import vtk # Needed for pyvista

import numpy   as np
import pyvista as pv
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py    import MPI
from basix.ufl import element

""" Visualize concentration slices in a 3x3 figure. """


# Set latex text properties
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
k = 1 # element degree
model_version = 'C'
molecule = 'D3'
tau_version = 'variable_tau'
input_filename = f"../output/flow/checkpoints/{tau_version}/pressure+original/model_C/velocity_data_dt=0.02252/"
mesh = a4d.read_mesh(comm=comm, filename=input_filename, engine="BP4", ghost_mode=gm)

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
W = dfx.fem.functionspace(mesh=mesh, element=element("DG", mesh.basix_cell(), k)) # DGk function space
cells, types, x = dfx.plot.vtk_mesh(W)
grid = pv.UnstructuredGrid(cells, types, x)

# Load time to threshold function values
c_in = dfx.fem.Function(W)
input_filename = f"../output/transport/results/{tau_version}/original/log_model_{model_version}_{molecule}_DG1_pressureBC/t_hat/"
a4d.read_function(u=c_in, filename=input_filename, engine="BP4", time=0)
grid.point_data["c"] = c_in.x.array.real
grid.set_active_scalars("c")
grid = grid.threshold(0.0) # Remove all points with values below zero (points that never reached the threshold)
grid = grid.threshold(840, invert=True) # Clip off posterior edge providing spurious coloring of cells

# Create plot object
plotter = pv.Plotter(window_size=[1600, 900], border=False)

# Set colormap
import colormaps as cm
cmap = cm.dense_r
my_colors = {'colors' : cmap.colors}
from scipy.io import savemat
# savemat('/Users/hherlyng/data_photoconversion/codes/colors.mat', my_colors) # Save to matlab file

# Add grid to plotter
plotter.add_mesh(grid.slice(normal=[0, -1, 0]), cmap=cmap, clim=[0, 800], show_scalar_bar=False)
plotter.add_lines(lines=np.array([[0.025, plotter.center[1], z_max*0.9], [0.075, plotter.center[1], z_max*0.9]]), color='k', width=10) # Add micrometer scalebar that is 50 microns long
plotter.camera.position    = (x_mid, 0, z_mid*1.1)
plotter.camera.focal_point = (x_mid, y_max, z_mid)
plotter.camera.zoom(0.275)

# Display the plot and save to file
save_fig = 1
if save_fig:
    output_filename = f'../output/illustrations/original/{tau_version}/slice_time_to_threshold_model{model_version}_{molecule}'
    plotter.show(screenshot=output_filename+'.png')
else:
    plotter.show()