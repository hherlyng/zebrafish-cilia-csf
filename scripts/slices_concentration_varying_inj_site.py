import vtk # Needed for pyvista

import numpy   as np
import pyvista as pv
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py    import MPI
from basix.ufl import element

import matplotlib as mpl
import matplotlib.colors
colors = [[14, 28, 31], [14, 29, 32], [15, 30, 33], [15, 32, 34], [16, 33, 36], [16, 34, 37], [16, 35, 38], [17, 37, 39], [17, 38, 40], [18, 39, 42], [18, 40, 43], [18, 42, 44], [19, 43, 45], [19, 44, 46], [19, 45, 47], [20, 47, 48], [20, 48, 49], [20, 49, 49], [20, 50, 50], [21, 51, 51], [21, 52, 52], [21, 54, 52], [21, 55, 53], [21, 56, 54], [22, 57, 55], [22, 58, 56], [22, 59, 56], [22, 60, 57], [22, 61, 58], [23, 62, 58], [23, 63, 59], [23, 64, 59], [23, 65, 60], [23, 66, 61], [23, 67, 61], [24, 68, 62], [24, 69, 63], [24, 70, 63], [24, 71, 64], [24, 72, 64], [25, 73, 65], [25, 75, 66], [25, 76, 66], [26, 78, 67], [26, 79, 67], [26, 80, 68], [27, 82, 68], [27, 83, 69], [27, 85, 69], [27, 86, 70], [28, 88, 70], [28, 89, 71], [28, 90, 71], [28, 91, 71], [28, 92, 71], [28, 93, 71], [29, 94, 71], [29, 95, 71], [29, 96, 71], [29, 97, 71], [29, 98, 71], [29, 99, 71], [29, 100, 71], [29, 101, 71], [29, 102, 71], [29, 103, 71], [29, 104, 71], [29, 105, 71], [30, 106, 70], [30, 107, 70], [30, 108, 70], [30, 109, 70], [30, 110, 70], [30, 111, 70], [30, 112, 69], [30, 113, 69], [30, 114, 69], [30, 115, 69], [30, 116, 68], [30, 117, 68], [30, 118, 67], [30, 119, 67], [30, 119, 66], [30, 120, 66], [30, 121, 65], [30, 122, 64], [30, 122, 64], [30, 123, 63], [30, 124, 63], [30, 125, 62], [30, 125, 61], [30, 126, 61], [30, 127, 60], [30, 127, 59], [30, 128, 58], [30, 129, 58], [30, 129, 57], [30, 130, 56], [30, 130, 55], [29, 131, 54], [29, 132, 54], [29, 132, 53], [29, 133, 52], [29, 134, 51], [29, 134, 50], [29, 135, 49], [29, 135, 48], [29, 136, 47], [29, 136, 46], [29, 137, 45], [29, 137, 44], [29, 138, 43], [28, 138, 42], [28, 139, 41], [28, 140, 40], [28, 140, 39], [28, 141, 38], [27, 142, 36], [27, 143, 35], [26, 143, 34], [26, 144, 33], [25, 145, 31], [25, 146, 30], [24, 146, 29], [23, 147, 27], [23, 148, 25], [22, 149, 24], [21, 150, 22], [21, 150, 21], [22, 151, 20], [22, 152, 20], [23, 153, 19], [24, 154, 19], [24, 155, 18], [25, 156, 17], [25, 156, 17], [26, 157, 16], [26, 158, 15], [27, 159, 15], [27, 160, 14], [28, 161, 13], [30, 162, 14], [32, 163, 14], [35, 164, 15], [37, 165, 15], [39, 165, 16], [41, 166, 17], [43, 167, 17], [45, 168, 18], [47, 169, 19], [49, 170, 19], [51, 171, 20], [52, 172, 20], [54, 173, 21], [56, 174, 22], [59, 175, 22], [61, 176, 23], [63, 176, 24], [65, 177, 24], [68, 178, 25], [70, 179, 25], [72, 180, 26], [74, 181, 27], [76, 182, 27], [78, 182, 28], [80, 183, 29], [82, 184, 29], [84, 185, 30], [86, 186, 31], [89, 187, 32], [91, 187, 33], [93, 188, 34], [96, 189, 35], [98, 190, 36], [100, 190, 37], [102, 191, 38], [105, 192, 39], [107, 193, 40], [109, 194, 41], [111, 194, 42], [114, 195, 44], [117, 196, 45], [119, 197, 47], [122, 198, 48], [124, 198, 50], [127, 199, 52], [129, 200, 53], [132, 201, 55], [134, 201, 56], [137, 202, 58], [139, 203, 59], [141, 204, 61], [144, 205, 62], [146, 205, 63], [148, 206, 64], [150, 207, 65], [153, 208, 66], [155, 209, 67], [157, 209, 68], [159, 210, 70], [161, 211, 71], [164, 212, 72], [166, 212, 73], [168, 213, 74], [170, 214, 75], [172, 215, 77], [175, 216, 79], [177, 216, 81], [180, 217, 83], [182, 218, 85], [184, 219, 87], [187, 220, 90], [189, 220, 92], [191, 221, 94], [194, 222, 96], [196, 223, 98], [198, 223, 100], [201, 224, 102], [203, 225, 105], [205, 226, 107], [207, 227, 110], [209, 228, 113], [211, 229, 116], [213, 229, 118], [215, 230, 121], [217, 231, 124], [219, 232, 127], [221, 233, 129], [223, 234, 132], [225, 235, 135], [227, 235, 137], [228, 236, 140], [230, 236, 143], [231, 237, 146], [233, 237, 148], [234, 238, 151], [236, 239, 154], [237, 239, 157], [239, 240, 159], [240, 240, 162], [242, 241, 165], [243, 241, 168], [245, 242, 170], [246, 243, 174], [247, 243, 179], [248, 244, 184], [248, 245, 188], [249, 245, 193], [250, 246, 198], [251, 247, 202], [252, 247, 207], [252, 248, 212], [253, 249, 216], [254, 250, 221], [254, 250, 225], [255, 251, 230]]
cmap = matplotlib.colors.ListedColormap(colors, name='g-w')
mpl.colormaps.register(cmap)

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

# Set up unstructured grids with data
W = dfx.fem.functionspace(mesh=mesh, element=element("DG", mesh.basix_cell(), k)) # DG1 function space
cells, types, x = dfx.plot.vtk_mesh(W)
timesteps = [1000, 8000, 18000] # the timesteps to visualize
grids = [pv.UnstructuredGrid(cells, types, x) for _ in range(3*len(timesteps))]

m = 0 # grid index
for j in range(len(timesteps)):
    for i in [1, 2, 3]:    
        c_in = dfx.fem.Function(W)
        if i==1:
            inj_site_str = 'anterior_dorsal'
        elif i==2:
            inj_site_str = 'middle_dorsal_posterior'
        elif i==3:
            inj_site_str = 'posterior_dorsal'
        input_filename = f"../output/checkpoints/transport/DG{k}/pressure/model_C/D2/concentration_inj={inj_site_str}"
        a4d.read_function(u=c_in, filename=input_filename, engine="BP4", time=timesteps[j])
        grids[m].point_data["c"] = c_in.x.array.real
        grids[m].set_active_scalars("c")
        m += 1 # increment

# Create plot object
plot_shape = (len(timesteps), 3)
# plotter = pv.Plotter(shape=plot_shape, window_size=[3200, 1400], border=False)
plotter = pv.Plotter(shape=plot_shape, window_size=[3200, 1400], border=False)

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

# Add grids to plotter
m = 0 # grid index
f = 2.22 # Cardiac frequency [Hz]
period = 1 / f
dt = period / 20
# plotter.add_mesh(grids[0].slice(normal=[0, -1, 0]), scalar_bar_args=sargs, cmap=rcmap, clim=[0, 1.0])
for i in range(plot_shape[0]):
    timelabel_text = rf'$t={timesteps[i]*dt:.1f}$'
    for j in range(plot_shape[1]):
        plotter.subplot(i, j)

        # Add text labels
        #if i==0: plotter.add_text(rf"$D_{j+1}$", position='upper_edge', font_size=20)
        #if j==0: plotter.add_text(timelabel_text, position='left_edge', font_size=20)

        plotter.camera.position    = (x_mid, 0, z_mid*1.1)
        plotter.camera.focal_point = (x_mid, y_max, z_mid)
        plotter.camera.zoom(0.35)
        slice = grids[m].slice(normal=[0, -1, 0])
        # if m==0:
        #     plotter.add_mesh(slice, scalar_bar_args=sargs, cmap=cmap)
        # else:
        plotter.add_mesh(slice, show_scalar_bar=False, cmap=cmap)
        m += 1
        
# Display the plot and save to file
output_filename = f'../output/illustrations/concentration_slices_DG{k}_varying_inj_site'
plotter.show(screenshot=output_filename+'.png')
# plotter.show()