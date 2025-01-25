import vtk # Needed for pyvista

import numpy   as np
import pyvista as pv
import dolfinx as dfx
import adios4dolfinx     as a4d

from basix.ufl import element

from mpi4py            import MPI

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
filename = "../output/checkpoints/transport/model_C/D1/concentration_inj=posterior_dorsal"
mesh = a4d.read_mesh(comm=comm, file=filename, engine="BP4", ghost_mode=gm)

def visualize_concentration_slices(mesh: dfx.mesh.Mesh, input_filename: str, output_filename: str):
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
    W = dfx.fem.functionspace(mesh=mesh, element=element("DG", mesh.basix_cell(), 1)) # DG1 function space
    cells, types, x = dfx.plot.vtk_mesh(W)
    timesteps = [0, 1000, 2000, 3000, 4000, 5000] # the timesteps to visualize
    grids = [pv.UnstructuredGrid(cells, types, x) for _ in range(len(timesteps))]
    for i in range(len(timesteps)):
        c_in = dfx.fem.Function(W)
        a4d.read_function(u=c_in, filename=input_filename, engine="BP4", time=timesteps[i])
        c_in.x.array[np.where(c_in.x.array[:] <= 0)] = 0.0
        grids[i].point_data["c"] = c_in.x.array.real
        grids[i].set_active_scalars("c")

    # Create plot object
    plot_shape = (2, 3)
    plotter = pv.Plotter(shape=plot_shape, window_size=[3000, 1400], border=False)
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
    k = 0 # grid index
    f = 2.22 # Cardiac frequency [Hz]
    period = 1 / f
    dt = period / 20
    for i in range(plot_shape[0]):
        for j in range(plot_shape[1]):
            plotter.subplot(i, j)
            plotter.camera.position    = (x_mid, 0, z_mid*1.1)
            plotter.camera.focal_point = (x_mid, y_max, z_mid)
            plotter.camera.zoom(0.267)
            slice = grids[k].slice(normal=[0, -1, 0])
            time_label = rf'$t={timesteps[k]*dt:.1f}$'
            if k==1:
                plotter.add_mesh(slice, scalar_bar_args=sargs, cmap=cmap, label=time_label)
            else:
                plotter.add_mesh(slice, show_scalar_bar=False, cmap=cmap, label=time_label)
            _ = plotter.add_legend(bcolor='w', face=None)
            k += 1
            
    # Display the plot and save to file
    plotter.show(screenshot='../output/illustrations/' + output_filename + ".png")

visualize_concentration_slices(mesh=mesh, input_filename=filename, output_filename="loop_viz")