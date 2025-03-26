import vtk # Needed for pyvista

import numpy   as np
import pyvista as pv
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py    import MPI
from basix.ufl import element

""" Visualize time-to-threshold for a slice of the geometry. """

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
k = 2 # element degree
model_version = 'C'
molecule = 'D3'
cilia_scenario = 'all_cilia'
mesh_version = 'original'
f = 2.22
dt = 1/f/20
input_filename = f'../output/flow/checkpoints/velocity_mesh={mesh_version}_model={model_version}_ciliaScenario={cilia_scenario}_dt={dt:.4g}'
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
W = dfx.fem.functionspace(mesh=mesh, element=element('DG', mesh.basix_cell(), k)) # DG_k function space
cells, types, x = dfx.plot.vtk_mesh(W)
grid = pv.UnstructuredGrid(cells, types, x)

# Load time to threshold function values
t_hat = dfx.fem.Function(W)
input_filename = f'../output/transport/mesh={mesh_version}_model={model_version}_molecule={molecule}_ciliaScenario={cilia_scenario}_dt={dt:.4g}/t_hat/'
a4d.read_function(u=t_hat, filename=input_filename, engine="BP4", time=0, name='t_hat')
grid.point_data['t_hat'] = t_hat.x.array.real
grid.set_active_scalars('t_hat')
grid = grid.threshold(0.0) # Remove all points with values below zero (points that never reached the threshold)
grid = grid.threshold(840, invert=True) # Clip off posterior edge providing spurious coloring of cells

# Create plot object
plotter = pv.Plotter(window_size=[1600, 900], border=False)

# Set colormap
import colormaps as cm
cmap = cm.dense_r
my_colors = {'colors' : cmap.colors}

# Add grid to plotter
plot_point_cloud = False
if plot_point_cloud:
    # Convert grid to polydata
    points = grid.slice(normal=[0, -1, 0]).points  # Extract coordinates
    scalars = grid.slice(normal=[0, -1, 0]).point_data['t_hat']  # Extract values

    point_cloud = pv.PolyData(points)
    point_cloud['t_hat'] = scalars 
    plotter.add_mesh(point_cloud, point_size=5, render_points_as_spheres=True, cmap=cmap, clim=[0, 800], show_scalar_bar=False)
else:
    plotter.add_mesh(grid.slice(normal=[0, -1, 0]), cmap=cmap, clim=[0, 800], show_scalar_bar=False)

plotter.add_lines(lines=np.array([[0.025, plotter.center[1], z_max*0.9], [0.075, plotter.center[1], z_max*0.9]]), color='k', width=10) # Add micrometer scalebar that is 50 microns long
plotter.camera.position    = (x_mid, 0, z_mid*1.1)
plotter.camera.focal_point = (x_mid, y_max, z_mid)
plotter.camera.zoom(0.275)

# Display the plot and save to file
save_fig = 0
if save_fig:
    output_filename = f'../output/illustrations/slice_time_to_threshold_model={model_version}_molecule={molecule}'
    plotter.show(screenshot=output_filename+'.png')
else:
    plotter.show()