import numpy   as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

# Set matplotlib properties
plt.rcParams.update({
    "font.family" : "Liberation Serif",
    "mathtext.fontset" : "dejavuserif",
    "axes.spines.top" : False,
    "axes.spines.right" : False
})

# Set data directory and load .mat files
data_dir = "../data/photoconversion_data/aggregated_data/"
metadata = loadmat(data_dir+"metadata") # Metadata, e.g. FPS and pixel size
colors   = loadmat(data_dir+"colors")['color'] # The colors used for plotting
control = loadmat(data_dir+"control_data")['control'] # The control fish cohort photoconversion data
mutant  = loadmat(data_dir+"mutant_data")['mut'] # The mutant fish cohort photoconversion data
tt_control = loadmat(data_dir+"tt_control")['tt_control'][0][0] # Time-to-threshold data for the control cohort
tt_mutant  = loadmat(data_dir+"tt_mutant")['tt_mutant'][0][0] # Time-to-threshold data for the mutant cohort 
fps = metadata['metadata'][0][0][2][0][0] # Extract fps from metadata
import colormaps as cm
blue = colors[0]
green = cm.dark2_3.colors[0]
orange = cm.dark2_3.colors[1]
purple = cm.puor_4.colors[3]
yellow = cm.puor_4.colors[1]
colors = ['k', green, purple, orange, yellow, blue]
# Get the number of timesteps
num_timesteps = control.shape[2]
times = 1/fps*np.arange(num_timesteps)
ROI_idx = range(6)

# Plot the figures
fig1, ax1 = plt.subplots(num=1, figsize=([18, 14])) # Plot for the control data
fig2, ax2 = plt.subplots(num=2, figsize=([18, 14])) # Plot for the mutant data

lw = 4.5 # linewidth

for i in ROI_idx:
    # Plot control data
    y = np.mean(control[:, i, :], axis=0)
    err = np.std(control[:, i, :], axis=0)/np.sqrt(control.shape[0])
    ax1.plot(times, y, label=f'ROI {i+1}' if i==0 else f'{i+1}', color=colors[i], linewidth=lw)
    ax1.fill_between(times, y-err, y+err, color=colors[i], alpha=0.25)

    # Plot mutant data
    y = np.mean(mutant[:, i, :], axis=0)
    err = np.std(mutant[:, i, :], axis=0)/np.sqrt(mutant.shape[0])
    ax2.plot(times, y, label=f'ROI {i+1}', color=colors[i], linewidth=lw)
    ax2.fill_between(times, y-err, y+err, color=colors[i], alpha=0.25)

ax1.set_xlabel("Time [s]", fontsize=60, labelpad=25)
ax1.set_ylabel(r"Fluoresc. intensity change $\Delta F$ [-]", fontsize=60, labelpad=25)
leg = ax1.legend(loc='upper left', fontsize=45, frameon=True, fancybox=False, edgecolor='k',
           ncols=len(ROI_idx),
           handlelength=1.2, borderpad=0.4, columnspacing=0.6, handletextpad=0.5)
# Increase the line width in the legend
for line in leg.get_lines():
    line.set_linewidth(10.0)
ax1.tick_params(labelsize=60)
ax2.set_xlabel("Time [s]", fontsize=60, labelpad=25)
ax2.set_ylabel(r"Fluoresc. intensity change $\Delta F$ [-]", fontsize=60, labelpad=25)
ax2.legend(loc='upper left', fontsize=45, frameon=True, fancybox=False, edgecolor='k')
ax2.tick_params(labelsize=60)

fig1.tight_layout()
fig2.tight_layout()

save_figs = 1
if save_figs:
    fig1.savefig(f"../output/illustrations/experimental_time_evolution_control.png")
    fig2.savefig(f"../output/illustrations/experimental_time_evolution_mutant.png")
plt.show()