import vtk # Needed for pyvista

import numpy   as np
import pyvista as pv
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py    import MPI
from basix.ufl import element

# Set latex text properties
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
k = 1 # element degree
input_filename = f"../output/checkpoints/transport/DG{k}/pressure/model_C/D1/concentration_inj=middle_dorsal_posterior"
mesh = a4d.read_mesh(comm=comm, file=input_filename, engine="BP4", ghost_mode=gm)

# Get mesh properties
x_geo = mesh.geometry.x

x_min = x_geo[:, 0].min()
x_max = x_geo[:, 0].max()
x_mid = (x_min + x_max) / 2

y_max = x_geo[:, 1].max()

z_min = x_geo[:, 2].min()
z_max = x_geo[:, 2].max()
z_mid = (z_min + z_max) / 2

a1 = [0.165, 0.0, z_mid]
b1 = [0.125, 0.0, z_max*0.78]

a2 = [0.33, 0.0, z_mid*0.8]
b2 = [0.345, 0.0, z_max*0.75]

# Set up unstructured grids with data
W = dfx.fem.functionspace(mesh=mesh, element=element("DG", mesh.basix_cell(), k)) # DG1 function space
cells, types, x = dfx.plot.vtk_mesh(W)
grid = pv.UnstructuredGrid(cells, types, x)

c_in = dfx.fem.Function(W)
a4d.read_function(u=c_in, filename=input_filename, engine="BP4", time=0)
grid.point_data["c"] = c_in.x.array.real
grid.set_active_scalars("c")


# Create plot object
plotter = pv.Plotter(window_size=[3200, 1400], border=False)

# Set colormap
from create_colormap import register_colormaps
register_colormaps()
cmap = "wgk"

# Colorbar arguments
sargs = dict(
    title_font_size=100,
    label_font_size=150,
    shadow=True,
    n_labels=2,
    italic=True,
    fmt="%.1g",
    font_family="times",
    vertical=True,
    title='',
    width=0.05,
    height=0.8,
    position_y=0.1
)

plotter.add_mesh(grid.slice(normal=[0, -1, 0]), color='white', show_edges=True, specular=0.75, specular_power=15)
plotter.add_lines(np.array([a1, b1, a2, b2]), color='black', width=10)
# plotter.camera.position    = (x_mid, 0, z_mid*1.1)
# plotter.camera.focal_point = (x_mid, y_max, z_mid)
plotter.camera_position='xz'
plotter.camera.zoom(2.0)
        
# Display the plot and save to file
output_filename = f'../output/illustrations/geometry_slice_with_middle_lines'
# plotter.show(screenshot=output_filename+'.png')
plotter.show()