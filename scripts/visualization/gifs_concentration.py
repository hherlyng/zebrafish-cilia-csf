import vtk # Needed for pyvista

import pyvista as pv
import dolfinx as dfx
import adios4dolfinx     as a4d

from mpi4py    import MPI
from basix.ufl import element

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
k = 1 # finite element degree
D_num = 2
inj_site = 'posterior_dorsal'
read_file = f"../output/checkpoints/transport/DG{k}/pressure/model_C/D{D_num}/concentration_inj={inj_site}"
mesh = a4d.read_mesh(comm=comm, file=read_file, engine="BP4", ghost_mode=gm)
plotter = pv.Plotter(window_size=([1280, 720]))
# Get mesh properties
x_geo = mesh.geometry.x

x_min = x_geo[:, 0].min()
x_max = x_geo[:, 0].max()
x_mid = (x_min + x_max) / 2

y_max = x_geo[:, 1].max()

z_min = x_geo[:, 2].min()
z_max = x_geo[:, 2].max()
z_mid = (z_min + z_max) / 2
plotter.camera.position    = (x_mid, 0, z_mid*1.1)
plotter.camera.focal_point = (x_mid, y_max, z_mid)
plotter.camera.zoom(0.267)


skip = 50
dt = 0.02252
fps = 40
plotter.open_gif(f"../output/illustrations/DG{k}_Diff{D_num}_concentration_inj={inj_site}gi_fps={fps}.gif", fps=fps)#int(1/dt)//skip)
f = 2.22 # Cardiac frequency [Hz]
period = 1 / f
T  = 1000*period
dt = period / 20
num_steps = int(T / dt)

W_scalar = dfx.fem.functionspace(mesh=mesh, element=element("DG", mesh.basix_cell(), k)) # DG function space
c = dfx.fem.Function(W_scalar)
cells, types, x = dfx.plot.vtk_mesh(mesh) if k==0 else dfx.plot.vtk_mesh(W_scalar) 
function_grid = pv.UnstructuredGrid(cells, types, x)
# Set colormap
from create_colormap import register_colormaps
register_colormaps()
cmap = "wgk"
for i in range(0, num_steps, skip):
    a4d.read_function(u=c, filename=read_file, engine="BP4", time=i)
    if k==0:
        function_grid.cell_data["c"] = c.x.array
    else:
        function_grid.point_data["c"] = c.x.array
    function_grid.set_active_scalars("c")
    actor1 = plotter.add_mesh(function_grid.slice(normal=[0, -1, 0]), clim=[0, 1.0], show_edges=False, cmap=cmap)
    time = i*dt
    actor2 = plotter.add_text(rf"$t = {time:.2f}$ sec")
    plotter.write_frame()
    plotter.remove_actor(actor1)
    plotter.remove_actor(actor2)

plotter.close()