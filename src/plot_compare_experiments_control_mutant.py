import numpy   as np
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.stats import mannwhitneyu

# Set matplotlib properties
plt.rcParams.update({
    "font.family" : "Arial",
    "axes.spines.top" : False,
    "axes.spines.right" : False
})
plt.rcParams["text.latex.preamble"] += "\\usepackage{sfmath}" # Enable sans-serif math font

# Set data directory and load .mat files
data_dir = "../data/photoconversion_data/aggregated_data/"
metadata = loadmat(data_dir+"metadata") # Metadata, e.g. FPS and pixel size
colors   = loadmat(data_dir+"colors")['color'] # The colors used for plotting
control = loadmat(data_dir+"control_data")['control'] # The control fish cohort photoconversion data
mutant  = loadmat(data_dir+"mutant_data")['mut'] # The mutant fish cohort photoconversion data
ttt_control = loadmat(data_dir+"tt_control")['tt_control'][0][0] # Time-to-threshold data for the control cohort
ttt_mutant  = loadmat(data_dir+"tt_mutant")['tt_mutant'][0][0] # Time-to-threshold data for the mutant cohort 
fps = metadata['metadata'][0][0][2][0][0] # Extract fps from metadata
ROI_tags = [1, 2, 3, 4, 5, 6]

# Clean ttt data and store it in dataframes
df_control = pd.DataFrame({tag: np.array(arr, dtype=np.float64).flatten() for tag, arr in zip(ROI_tags, ttt_control)})
df_mutant  = pd.DataFrame({tag: np.array(arr, dtype=np.float64).flatten() for tag, arr in zip(ROI_tags, ttt_mutant )})

# Get the number of timesteps
num_timesteps = control.shape[2]
c_threshold1 = 0.25 # threshold value used to calculate "time to threshold" in ROIs 1-4
c_threshold2 = 0.10 # threshold value used to calculate "time to threshold" in ROIs 5-6
times = 1/fps*np.arange(num_timesteps)

#-------- Plot the figures --------#
fig_dff, ax_dff = plt.subplots(num=1, nrows=3, ncols=2, figsize=[14, 14]) # Concentration curves
fig_ttt, ax_ttt = plt.subplots(num=2, nrows=1, ncols=6, figsize=[28, 5]) # Time to threshold scatter plots
lw = 3.5 # linewidth
msize = 20 # markersize
c_str = r'$\overline{c}$'
t_str = r'$\hat{t}$'
colors = ['k',        # Black, control
          '#bf0014ff' # Dark red, mutant
]

for idx, tag in enumerate(ROI_tags):
    if tag % 2 != 0:
        # Odd number -> plot in 1st column
        row_idx = int((tag-1)/2)
        col_idx = 0
    else:
        # Even number -> plot in 2nd column
        row_idx = int((tag-2)/2)
        col_idx = 1

    ca_dff = ax_dff[row_idx, col_idx]
    ca_ttt = ax_ttt[idx]

    # Plot control data
    y = np.mean(control[:, idx, :], axis=0)
    err = np.std(control[:, idx, :], axis=0)/np.sqrt(control.shape[0])
    ca_dff.plot(times, y, label='Control', color=colors[0], linewidth=lw)
    ca_dff.fill_between(times, y-err, y+err, color=colors[0], alpha=0.25)

    # Plot mutant data
    y = np.mean(mutant[:, idx, :], axis=0)
    err = np.std(mutant[:, idx, :], axis=0)/np.sqrt(mutant.shape[0])
    ca_dff.plot(times, y, label='Mutant', color=colors[1], linewidth=lw)
    ca_dff.fill_between(times, y-err, y+err, color=colors[1], alpha=0.25)

    ca_dff.set_title(f'ROI {tag}', fontsize=35)
    ca_dff.set_xticks([0, 250, 500, 750])
    ca_dff.tick_params(labelsize=36)
    if col_idx==0: ca_dff.set_ylabel(r"$\Delta F$ [-]", fontsize=40, labelpad=25)
    if row_idx==(ax_dff.shape[0]-1): ca_dff.set_xlabel("Time [s]", fontsize=40, labelpad=25)    

    # Plot time to threshold
    df = pd.DataFrame(index=range(df_control.shape[0]), columns=['control', 'mutant'])
    df['control'] = df_control[tag]
    df['mutant'] = df_mutant[tag]

    # Create a swarm plot of the times to threshold
    sns.swarmplot(data=df, x=0.15, y='control', size=msize, color=colors[0], alpha=0.5, ax=ca_ttt)
    sns.swarmplot(data=df, x=0.85, y='mutant', size=msize, color=colors[1], alpha=0.5, ax=ca_ttt)

    # Overlay a violin plot with horizontal lines at each marker
    sns.violinplot(data=df, x=0.15, y='control', inner=None, width=0.5, density_norm='width', cut=0, color=colors[0], alpha=0.25, ax=ca_ttt)
    sns.violinplot(data=df, x=0.85, y='mutant', inner=None, width=0.5, density_norm='width', cut=0, color=colors[1], alpha=0.25, ax=ca_ttt)
    
    # Plot horizontal line at means
    control_mean = df['control'].mean()
    mutant_mean = df['mutant'].mean()
    print("Control mean: ", control_mean)
    print("Mutant mean: ", mutant_mean)
    ca_ttt.axhline(y=control_mean, xmin=0.1, xmax=0.4, color='k', linewidth=3)
    ca_ttt.axhline(y=mutant_mean, xmin=0.6, xmax=0.9, color='k', linewidth=3)

    # Calculate and annotate p-value calculated with a Mann-Whitney U test (also known as Wilcoxon rank-sum test)
    # This test compares the control and mutant data and tests if they are from the same population
    _, p_value = mannwhitneyu(x=df['control'].values, y=df['mutant'].dropna().values, alternative='two-sided')
    ca_ttt.text(0.05, 0.95, f'$p$={p_value:.4f}', transform=ca_ttt.transAxes, fontsize=32, verticalalignment='top')
    
    ca_ttt.tick_params(axis='y', labelsize=36)
    ca_ttt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ca_ttt.set_ylabel("Time [s]", fontsize=0)
    ca_ttt.set_xlabel(f'ROI {tag}', fontsize=36, labelpad=25)
    ca_ttt.set_ylim(0, ca_ttt.get_ylim()[1]*1.25)

    if idx==0:
        ca_dff.legend(loc='lower right', fontsize=32, frameon=True, fancybox=False, edgecolor='k')
        ca_ttt.legend(labels=['Control', 'Mutant'], loc='lower right', fontsize=27, frameon=True, fancybox=False, edgecolor='k')
        ca_ttt.set_ylabel("Time [s]", fontsize=40, labelpad=25)

# Set tight layout and show (and optionally save)
fig_dff.tight_layout(); fig_ttt.tight_layout()
save_figs = 1
if save_figs:
    fig_dff.savefig(f"../output/illustrations/experiments_dff_compare_control_mutant.png")
    fig_ttt.savefig(f"../output/illustrations/experiments_ttt_compare_control_mutant.png")
plt.show()