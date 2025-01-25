import vtk # Needed for pyvista

import pyvista as pv
import dolfinx as dfx
import adios4dolfinx     as a4d

from mpi4py    import MPI
from basix.ufl import element

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
k = 1 # finite element degree
read_file1 = f"../output/checkpoints/transport/DG{k}/pressure/model_C/D1/concentration_inj=middle_dorsal_posterior"
read_file2 = f"../output/checkpoints/transport/DG{k}/pressure/model_C/D2/concentration_inj=middle_dorsal_posterior"
read_file3 = f"../output/checkpoints/transport/DG{k}/pressure/model_C/D3/concentration_inj=middle_dorsal_posterior"
mesh = a4d.read_mesh(comm=comm, file=read_file1, engine="BP4", ghost_mode=gm)
plotter = pv.Plotter(shape=(1, 3), window_size=[3200, 1400])
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
plotter.camera.zoom(0.2)


skip = 50
dt = 0.02252
fps = 40
plotter.open_gif(f"../output/illustrations/DG{k}_compare_diff_coeffs_fps={fps}.gif", fps=fps)#int(1/dt)//skip)
f = 2.22 # Cardiac frequency [Hz]
period = 1 / f
T  = 1000*period
dt = period / 20
num_steps = int(T / dt)

W_scalar = dfx.fem.functionspace(mesh=mesh, element=element("DG", mesh.basix_cell(), k)) # DG function space
c1 = dfx.fem.Function(W_scalar)
c2 = dfx.fem.Function(W_scalar)
c3 = dfx.fem.Function(W_scalar)
cells, types, x = dfx.plot.vtk_mesh(mesh) if k==0 else dfx.plot.vtk_mesh(W_scalar) 
grids = [pv.UnstructuredGrid(cells, types, x) for _ in range(3)]
cmap = "speed"
for i in range(0, 5*skip, skip):
    a4d.read_function(u=c1, filename=read_file1, engine="BP4", time=i)
    a4d.read_function(u=c2, filename=read_file2, engine="BP4", time=i)
    a4d.read_function(u=c3, filename=read_file3, engine="BP4", time=i)
    if k==0:
        grids[0].cell_data["c"] = c1.x.array
        grids[1].cell_data["c"] = c2.x.array
        grids[2].cell_data["c"] = c3.x.array
    else:
        grids[0].point_data["c"] = c1.x.array
        grids[1].point_data["c"] = c2.x.array
        grids[2].point_data["c"] = c3.x.array
    if i==0: [grids[i].set_active_scalars("c") for i in [0, 1, 2]]
    plotter.subplot(0)
    slice1 = grids[0].slice(normal=[0, -1, 0])
    actor1 = plotter.add_mesh(slice1, clim=[0, 1.0], cmap=cmap)
    time = i*dt
    actor2 = plotter.add_text(rf"$t = {time:.2f}$ sec")

    plotter.subplot(1)
    if i==0:
        plotter.camera.position    = (x_mid, 0, z_mid*1.1)
        plotter.camera.focal_point = (x_mid, y_max, z_mid)
        plotter.camera.zoom(0.5)
    
    slice2 = grids[1].slice(normal=[0, -1, 0])
    actor3 = plotter.add_mesh(slice2, clim=[0, 1.0], cmap=cmap)

    
    plotter.subplot(2)
    slice3 = grids[2].slice(normal=[0, -1, 0])
    actor4 = plotter.add_mesh(slice3, clim=[0, 1.0], cmap=cmap)
    actor5 = plotter.set_position((x_mid, 0, z_mid*1.1))
    actor6 = plotter.set_focus((x_mid, y_max, z_mid))
    actor7 = plotter.camera.zoom(0.5)

    plotter.write_frame()

    plotter.remove_actor(actor1)
    plotter.remove_actor(actor2)
    plotter.remove_actor(actor3)
    plotter.remove_actor(actor4)
    plotter.remove_actor(actor5)
    plotter.remove_actor(actor6)
    plotter.remove_actor(actor7)

plotter.close()