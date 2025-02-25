import ufl

import numpy   as np
import pandas  as pd
import dolfinx as dfx
import adios4dolfinx     as a4d
import matplotlib.pyplot as plt

from mpi4py       import MPI
from scipy.io     import loadmat
from utilities.mesh import create_ventricle_volumes_meshtags

# Set matplotlib properties
plt.rcParams.update({
    "text.usetex" : True,
    "font.family" : "sans-serif",
    "axes.spines.top" : False,
    "axes.spines.right" : False
})
plt.rcParams["text.latex.preamble"] += "\\usepackage{sfmath}" # Enable sans-serif math font

comm = MPI.COMM_WORLD # MPI Communicator
gm   = dfx.mesh.GhostMode.shared_facet
colors = loadmat("../data/data_photoconversion/aggregated_data/colors")['color'] # The colors used for plotting
fig_num = 1 # Figure number index

# Directories
flow_dir = '../output/flow/checkpoints/'
transport_dir = '../output/transport/results/'

# Problem version
model_version = 'C'
molecules = ['D1', 'D2', 'D3']
tau_version  = 'variable_tau'
mesh_version = 'original'
output_dir = f'../output/illustrations/{mesh_version}/{tau_version}/'

# Read mesh
mesh_input_filename = flow_dir+f"{tau_version}/pressure+{mesh_version}/model_{model_version}/velocity_data_dt=0.02252"
mesh = a4d.read_mesh(comm=comm, filename=mesh_input_filename, engine="BP4", ghost_mode=gm)

# Create meshtags and calculate ROI volumes
mt, ROI_tags = create_ventricle_volumes_meshtags(mesh)
dx = ufl.Measure('dx', domain=mesh, subdomain_data=mt)
volumes = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(tag))), op=MPI.SUM) for tag in ROI_tags]
volumes[3] += volumes[2]+volumes[1]+volumes[0] # Add ROI 1, 2, 3 volumes to ROI 4 volume

# Analysis parameters
c_threshold1 = 0.25 # threshold value used to calculate "time to threshold" in ROIs 1-4
c_threshold2 = 0.10 # threshold value used to calculate "time to threshold" in ROIs 5-6
f = 2.22
dt = 1/f/20

# Plot parameters
save_figs = 0 # Save figures if set to 1, don't save if set to 0
lw = 8 # linewidth
msize = 4 # marker size

# Create an empty dataframe for storing the t_hats (time to threshold)
t_hat_df = pd.DataFrame(index=ROI_tags, columns=molecules)

# Loop over molecules
for idx, molecule in enumerate(molecules):
    transport_data_filename = transport_dir + \
        f"{tau_version}/{mesh_version}/log_model_{model_version}_{molecule}_DG1_pressureBC/data/c_hats.npy"
    
    # Load transport data: c_bars = the total concentration in each ROI
    with open(transport_data_filename, "rb") as file: c_bars = np.load(file) 
    

    # Scale the total concentrations in the ROIs by the volume of the respective ROI
    for i in ROI_tags: c_bars[:, i-1] /= volumes[i-1]
    print("c_bar at final time: ", c_bars[-1, :])

    # Write final concentration values to file
    with open(file=output_dir+f"final_c_bar_values.txt", mode="w+" if idx==0 else "a+") as file:
        file.write(str(c_bars[-1, :])+'\n')

    # Get the number of timesteps
    num_timesteps = c_bars.shape[0]
    times = dt*np.arange(num_timesteps)

    # Plot the figures
    fig, ax = plt.subplots(num=fig_num, figsize=([15, 12]))
    fig_num += 1

    [ax.plot(times, c_bars[:, i-1], label=f'ROI {i}', color=colors[i-1], linewidth=lw) for i in ROI_tags]
    
    ax.set_xlabel("Time [s]", fontsize=60, labelpad=25)
    if molecule=='D1': 
        # Add legend 
        ax.legend(loc='best', fontsize=50, frameon=True, fancybox=False, edgecolor='k')
    # Add y-axis label
    ax.set_ylabel(r"Mean concentration $\overline{c}$ [-]", fontsize=60, labelpad=25)
    ax.tick_params(labelsize=55)
    fig.tight_layout()

    if save_figs: fig.savefig(f"{output_dir}/simulations_time_evolution_model{model_version}_{molecule}.png")

    # Calculate times to threshold
    t_hat_arr = np.array([0]*len(ROI_tags), dtype=np.float64)
    # define the first time-instant where c_threshold is exceeded
    # as the "time to reach threshold" 
    for j in ROI_tags:
        if j in [1, 2, 3, 4]:
            c_threshold = c_threshold1
        else:
            c_threshold = c_threshold2
        if c_bars[-1, j-1] < c_threshold: c_threshold = 0.0
        t_hat = np.where(c_bars[:, j-1] > c_threshold)[0][0]
        t_hat_arr[j-1] = t_hat
    t_hat_arr *= dt
    t_hat_df[molecule] = t_hat_arr

fig2, ax2 = plt.subplots(num=4, figsize=[13, 8])
t_hat_df.reset_index(inplace=True)
bars = t_hat_df.plot.bar(x='index', y=molecules,
                         color=[[0.6, 0.752941, 0.145098], [0.294118, 0.564706, 0], [0, 0.376471, 0]],
                         ax=ax2, rot=True, width=0.75)
hatches = ['/', '\\', 'x']
for bar_container, hatch in zip(bars.containers, hatches):
    for bar in bar_container:
        bar.set_hatch(hatch)

ax2.set_xlabel("ROI number", fontsize=35, labelpad=25)
ax2.set_ylabel(r"Time to threshold $\hat{t}$ [s]", fontsize=35, labelpad=50)
ax2.tick_params(labelsize=35)
ax2.legend(labels=[r'$D_1$', r'$D_2$', r'$D_3$'], loc='upper left', fontsize=32, frameon=True, fancybox=False, edgecolor='k')
ax2.yaxis.tick_right()
ax2.yaxis.label_position = ['Right']
ax2.spines['right'].set_visible(True)
ax2.spines['left'].set_visible(False)
fig2.tight_layout()

if save_figs: fig2.savefig(f"{output_dir}/simulations_time_to_threshold_barplot_model{model_version}.png")
plt.show()