import numpy  as np
import matplotlib.pyplot as plt

plt.style.use('petroff10')

# Set matplotlib properties
plt.rcParams.update({
    "font.family" : "Arial",
    "axes.spines.top" : False,
    "axes.spines.right" : False
})

save_fig = 1

import colormaps as cm
green = cm.dark2_3.colors[0]
orange = cm.dark2_3.colors[1]
purple = cm.puor_4.colors[3]
colors = [green, orange, purple, 'k']

fig, ax = plt.subplots(figsize=[11.35, 10])
idx = [1, 2, 3, 4]

# Forces applied [dorsal, ventral, anterior, total]
forces = np.array([3.64E-06, 2.41E-06, 9.78E-07, 7.03E-06])*1e6 # Forces in micro Newton

hatches = ['/', '\\', 'x', '']
bars = ax.bar(x=idx, height=forces, width=0.5, color=colors, hatch=hatches)

ax.set_title('Cilia population contribution', fontsize=50, pad=30)
ax.set_ylabel(r'Tangential force $\mathrm{[\mu N]}$', fontsize=45, labelpad=25)
ax.set_yticks(range(1, 8))
ax.set_xticks(idx)
ax.set_xticklabels(['Dorsal', 'Ventral', 'Anterior', 'Total'], fontsize=45)
ax.tick_params(axis='both', labelsize=45, pad=25)
fig.tight_layout()

if save_fig: fig.savefig('../output/illustrations/compare_cilia/cilia_forces_bar_plot.png')
plt.show()