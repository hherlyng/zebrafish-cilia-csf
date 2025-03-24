import numpy     as np
import pandas    as pd
import colormaps as cm
import matplotlib.pyplot as plt

# Set matplotlib properties
plt.rcParams.update({
    "font.family" : "Arial",
    "axes.spines.top" : False,
    "axes.spines.right" : False
})

save_figs = 1
regions = ['Dorsal', 'Ventral', 'Anterior', 'Total']
versions = ['Original', 'Shrunk fore-mid', 'Shrunk middle', 'Shrunk mid-hind', 'Fore+middle+hind']


##########################
##  FIGURE 5 BAR PLOT   ##
##########################
green = cm.dark2_3.colors[0]
orange = cm.dark2_3.colors[1]
purple = cm.puor_4.colors[3]
colors = [green, purple, orange, 'k']

fig, ax = plt.subplots(figsize=[11.35, 10])
idx = [1, 2, 3, 4]

# Forces applied [dorsal, ventral, anterior, total]
forces = np.array([3.72E-06, 2.41E-06, 9.78E-07, 7.11E-06])*1e6 # Forces in micro Newton

hatches = ['/', '\\', 'x', '']
bars = ax.bar(x=idx, height=forces, width=0.5, color=colors, hatch=hatches)

ax.set_title('Regional cilia contribution', fontsize=52, pad=20)
ax.set_ylabel(r'Tangential force $\mathrm{[\mu N]}$', fontsize=48, labelpad=25)
ax.set_yticks(range(1, 8))
ax.set_xticks(idx)
ax.set_xticklabels(regions, fontsize=48)
ax.tick_params(axis='both', labelsize=48, pad=25)
fig.tight_layout()

if save_figs: fig.savefig('../output/illustrations/compare_cilia/fig5_cilia_forces_bar_plot.png')

##########################
##  FIGURE 5 BAR PLOT   ##
##########################
green = cm.dark2_3.colors[0]
orange = cm.dark2_3.colors[1]
purple = cm.puor_4.colors[3]
yellow = cm.puor_4.colors[1]
colors = ['k', green, purple, orange, yellow]

# Forces applied [dorsal, ventral, anterior, total] in micro Newton
forces = np.array([[3.72E-06, 2.41E-06, 9.78E-07, 7.11E-06], # original
                   [3.69E-06, 2.18E-06, 8.20E-07, 6.69E-06], # fore shrunk
                   [3.65E-06, 2.57E-06, 9.55E-07, 7.18E-06], # middle shrunk
                   [3.54E-06, 2.41E-06, 9.14E-07, 6.86E-06], # hind shrunk
                   [3.77E-06, 2.33E-06, 7.65E-07, 6.86E-06]])*1e6 # fore+middle+hind shrunk
for i in range(forces.shape[1]):
    forces[:, i] /= forces[0, i]
forces = forces*100

df_forces = pd.DataFrame(data=forces.T, columns=versions, index=regions)
fig_bars, ax_bars = plt.subplots(figsize=[14, 4.5])
df_forces.reset_index(inplace=True)

print(df_forces)
bars = df_forces.plot.bar(x='index', y=versions, width=0.65,
                          color=colors, ax=ax_bars, rot=True)

hatches = ['', '\\', 'x', '/', '+']
for bar_container, hatch in zip(bars.containers, hatches):
    for bar in bar_container:
        bar.set_hatch(hatch)


ax_bars.legend(labels=versions, loc='center', bbox_to_anchor=(1.125, 0.5),
                   fontsize=17, frameon=True, fancybox=False, edgecolor='k')
ax_bars.set_ylabel('Relative tangential force [%]', fontsize=22, labelpad=20)
ax_bars.set_ylim([65, 115])
ax_bars.set_yticks([70, 80, 90, 100, 110])
ax_bars.set_xticks([])
ax_bars.set_xticks(df_forces.index.values)
ax_bars.set_xticklabels(regions)
ax_bars.set_xlabel('')
ax_bars.tick_params(axis='both', labelsize=20)
fig_bars.subplots_adjust(right=0.80, left=0.1)
if save_figs: fig_bars.savefig('../output/illustrations/compare_geometry/fig6_cilia_forces_bar_plot.png')
plt.show()