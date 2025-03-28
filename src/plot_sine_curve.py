import numpy as np
import matplotlib.pyplot as plt

plt.style.use('petroff10')

# Set matplotlib properties
plt.rcParams.update({
    "font.family" : "Liberation Serif",
    "mathtext.fontset" : "dejavuserif",
    "axes.spines.top" : False,
    "axes.spines.right" : False
})

x = np.linspace(0, 1, 101)
sine_curve = lambda t: np.sin(2*np.pi*t)
y = sine_curve(x)

fig, ax = plt.subplots(figsize=[16, 7])

ax.plot(x, y, linewidth=5)
font = {'family': 'Liberation Serif',
        'size'  : 50}

ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'], fontdict=font)
ax.set_yticks([-1.0, 0.0, 1.0])
ax.set_yticklabels([r'$-A$', '0', r'$A$'], fontdict=font)
ax.tick_params(axis='both', pad=20)

# Add points with text to plot
f = 2.22 # Cardiac frequency [Hz]
T = 1 / f # Cardiac period [s]
dt = T / 20 # Timestep size
x1 = 5/20
x2 = 10/20
x3 = 15/20
y1 = sine_curve(x1)
y2 = sine_curve(x2)
y3 = sine_curve(x3)

xs = [x1, x2, x3]
ys = [y1, y2, y3]

[ax.plot(x, y, color='k', marker='o', markersize=15) for x, y in zip(xs, ys)]
bfont = {'family': 'Liberation Serif',
         'size'  : 50,
         'fontweight' : 'bold'}
ax.text(xs[0], ys[0]*1.125, 'f, h', fontdict=bfont)
ax.text(xs[1]*1.025, ys[1], 'i', fontdict=bfont)
ax.text(xs[2], ys[2]*0.85, 'j', fontdict=bfont)

fig.tight_layout()
fig.savefig('../output/illustrations/misc-figure-material/sine_curve.png')
plt.show()