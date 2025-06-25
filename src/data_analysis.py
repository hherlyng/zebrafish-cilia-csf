from mpi4py import MPI
from utilities.mesh import create_ventricle_volumes_meshtags

import ufl
import numpy   as np
import pandas  as pd
import dolfinx as dfx
import colormaps as cm
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
save_figs = 1

# Plotting style
blue = [0, 0.447058823529412, 0.741176470588235]
green = cm.dark2_3.colors[0]
orange = cm.dark2_3.colors[1]
purple = cm.puor_4.colors[3]
yellow = cm.puor_4.colors[1]
colors = ['k', green, purple, orange, yellow, blue]
markers = ['', '.', '^', 'x', 's']
linewidth = 2.5
markersize = 12

# Prepare calculation of time to threshold
dt = 0.02252
t_hat_arr = np.array([0.0]*6, dtype=np.float64) # Times to threshold for each ROI
t_hat_df = pd.DataFrame(index=range(1, 7), columns=range(5)) # Times to threshold for all meshes and each ROI
c_threshold1 = 0.25 # threshold value used to calculate "time to threshold" in ROIs 1-4
c_threshold2 = 0.10 # threshold value used to calculate "time to threshold" in ROIs 5-6

fig1, ax1 = plt.subplots(figsize=[13, 8])

# Loop over the five ventricles meshes
for i in [0, 1, 2, 3, 4]:
    print(f"\n Ref {i}")
    data = np.load(f"../output/transport/verification/ref{i}/data/c_hats.npy", "r")
    if i==0: times = np.arange(data.shape[0])*dt

    # Create meshtags and calculate ROI volumes
    mesh_input_filename = f'../geometries/ventricles/ventricles_{i}.xdmf'
    with dfx.io.XDMFFile(comm, mesh_input_filename, "r") as xdmf:
        mesh = xdmf.read_mesh()
    
    mt, ROI_tags = create_ventricle_volumes_meshtags(mesh)
    dx = ufl.Measure('dx', domain=mesh, subdomain_data=mt)
    volumes = [comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx(tag))), op=MPI.SUM) for tag in ROI_tags]
    volumes[3] += volumes[2]+volumes[1]+volumes[0] # Add ROI 1, 2, 3 volumes to ROI 4 volume
    
    if i==0: final_c_vals_ref0 = []
    final_c_vals_refX = []

    for roi_idx in range(6):
        c_bars = data[:, roi_idx]/volumes[roi_idx] # Scale the total concentrations in the ROIs by the volume of the respective ROI
        ax1.plot(times,
                c_bars,
                color=colors[roi_idx],
                marker=markers[i],
                linestyle='--' if i>0 else '-',
                label=f"ref{i}" if roi_idx==0 else None,
                markevery=2500,
                linewidth=linewidth,
                markersize=markersize)
        print("ROI ", roi_idx+1)
        print("Final c bar: ", c_bars[-1])
        if i==0:
            final_c_vals_ref0.append(c_bars[-1])
        else:
            final_c_vals_refX.append(c_bars[-1])

        # Find time to threshold
        # define the first time-instant where c_threshold is exceeded
        # as the "time to reach threshold" 
    
        if roi_idx+1 in [1, 2, 3, 4]:
            c_threshold = c_threshold1
        else:
            c_threshold = c_threshold2
        if c_bars[-1] < c_threshold: c_threshold = 0.0
        t_hat = np.where(c_bars[:] > c_threshold)[0][0]
        t_hat_arr[roi_idx] = t_hat
    t_hat_arr *= dt
    t_hat_df[i] = t_hat_arr # per refinement
    
    if i>0:
        print(f"Percentage diff rel to ref0 final ROI c: {(np.array(final_c_vals_refX)/np.array(final_c_vals_ref0)-1)*100}")
        final_c_vals_refX = []


    # Check final total c
    final_c = open(f"../output/transport/verification/ref{i}/data/final_total_c.txt", "r")
    vol = comm.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(1*dx)), op=MPI.SUM)
    if i==0:
        final_c_ref0 = float(final_c.read())
        print(f"Final total c (ref{i}): {final_c_ref0/vol:.2e}")
    else:
        final_c = float(final_c.read())
        print(f"Final mean total c (ref{i}): {final_c/vol:.2e}")
        percentage_diff = (final_c/final_c_ref0-1)*100
        print(f"Percentage diff rel to ref0 final total c: {percentage_diff}")

print("Times to threshold\n(row = ref)\n(col = ROI)\n", t_hat_df.T)

# Format figure 1
ax1.set_xlabel("Time [s]", fontsize=30, labelpad=20)
ax1.set_ylabel(r"Mean concentration $\overline{c}$ [-]", fontsize=30, labelpad=20)
ax1.tick_params(labelsize=30)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
leg = ax1.legend(labels=['ROI 1', '2', '3', '4', '5', '6'], 
                loc='upper left',
                fontsize=25,
                frameon=True,
                fancybox=False,
                edgecolor='k',
                ncols=6,
                handlelength=1.2,
                borderpad=0.4,
                columnspacing=0.6,
                handletextpad=0.5)
for line in leg.get_lines(): line.set_linewidth(6.0) # Increase the line width in the legend
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize=[13, 8])
for i in range(1, 7):
    div_factor = t_hat_df[t_hat_df.index==i][0]
    for j in range(5):
        t_hat_df.loc[i, j] *= 100/div_factor.values
bars = t_hat_df.plot.bar(y=t_hat_df.columns,
                         color=colors[:-1],
                         ax=ax2, rot=True, width=0.75)
hatches = ['/', '\\', 'x', 's', '^']
for bar_container, hatch in zip(bars.containers, hatches):
    for bar in bar_container:
        bar.set_hatch(hatch)

ax2.set_xlabel("ROI number", fontsize=30, labelpad=25)
ax2.set_ylabel(r"Relative time to threshold $\hat{t}_i/\hat{t}_0$ [\%]", fontsize=30, labelpad=50)
ax2.tick_params(labelsize=30)
ax2.legend(labels=['Ref. 0', 'Ref. 1', 'Ref. 2', 'Ref. 3', 'Ref. 4'], loc='upper left', fontsize=25, frameon=True, fancybox=False, edgecolor='k')
ax2.yaxis.tick_right()
ax2.yaxis.label_position = ['Right']
ax2.spines['right'].set_visible(True)
ax2.spines['left'].set_visible(False)
fig2.tight_layout()

if save_figs:
    fig1.savefig("../output/illustrations/mean_ROI_concentrations_refinement.png")
    fig2.savefig("../output/illustrations/times_to_threshold_refinement.png")

plt.show()