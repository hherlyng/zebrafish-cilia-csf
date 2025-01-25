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
input_filename = "../output/checkpoints/transport/DG1/pressure/model_C/D1/concentration_inj=anterior_dorsal"
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

# Set up unstructured grids with data
k = 1 # element degree
W = dfx.fem.functionspace(mesh=mesh, element=element("DG", mesh.basix_cell(), k)) # DG1 function space
cells, types, x = dfx.plot.vtk_mesh(W)
timesteps = [100, 500, 1000] # the timesteps to visualize
grids = [pv.UnstructuredGrid(cells, types, x) for _ in range(3*len(timesteps))]

m = 0 # grid index
for j in range(len(timesteps)):
    for i in [1, 2, 3]:    
        c_in = dfx.fem.Function(W)
        a4d.read_function(u=c_in, filename=input_filename, engine="BP4", time=timesteps[j])
        grids[m].point_data["c"] = c_in.x.array.real
        grids[m].set_active_scalars("c")
        m += 1 # increment

# Create plot object
plot_shape = (len(timesteps), 3)
plotter = pv.Plotter(shape=plot_shape, window_size=[3200, 1400], border=False)
# cmap = "CET_D1A" # Cold-warm colormap
cmap = "speed"

# Colorbar arguments
sargs = dict(
    title_font_size=50,
    label_font_size=40,
    shadow=True,
    n_labels=3,
    italic=True,
    fmt="%.1g",
    font_family="times",
    position_x=0.2,
)

# Add grids to plotter
m = 0 # grid index
f = 2.22 # Cardiac frequency [Hz]
period = 1 / f
dt = period / 20
props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
for i in range(plot_shape[0]):
    timelabel_text = rf'$t={timesteps[i]*dt:.1f}$'
    for j in range(plot_shape[1]):            
        plotter.subplot(i, j)
        #if i==0: plotter.add_text(rf"$D_{j+1}$", position='upper_edge', font_size=20)
        #if j==0: plotter.add_text(timelabel_text, position='left_edge', font_size=20)
        plotter.camera.position    = (x_mid, 0, z_mid*1.1)
        plotter.camera.focal_point = (x_mid, y_max, z_mid)
        plotter.camera.zoom(0.325)
        slice = grids[m].slice(normal=[0, -1, 0])
        plotter.add_mesh(slice, show_scalar_bar=False, cmap=cmap)
        m += 1
        
# Display the plot
plotter.show(screenshot='../output/illustrations/example_concentration_slices_grid.png')