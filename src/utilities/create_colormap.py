def register_colormaps():
    import matplotlib
    import matplotlib.colors
    import numpy as np
    import scipy.interpolate

    colors_wgk = [[1, 1, 1],
              [0.901961, 0.941176, 0.796078],
              [0.8, 0.878431, 0.580392],
              [0.701961, 0.815686, 0.364706],
              [0.6, 0.752941, 0.145098],
              [0.498039, 0.690196, 0],
              [0.396078, 0.627451, 0],
              [0.294118, 0.564706, 0],
              [0.196078, 0.501961, 0],
              [0.094118, 0.439216, 0],
              [0, 0.376471, 0],
              [0, 0.313725, 0],
              [0, 0.25098, 0],
              [0, 0.188235, 0],
              [0, 0.12549, 0],
              [0, 0.062745, 0],
              [0, 0, 0]
    ]
    # Refine color data
    arr = np.array(colors_wgk)
    x = np.linspace(0, 1, len(arr[:, 0]))
    interp_R = scipy.interpolate.interp1d(x, arr[:, 0])
    interp_G = scipy.interpolate.interp1d(x, arr[:, 1])
    interp_B = scipy.interpolate.interp1d(x, arr[:, 2])
    x_fine = np.linspace(0, 1, 101)

    refined_colors = np.zeros(shape=(len(x_fine), 3))
    refined_colors[:, 0] = interp_R(x_fine)
    refined_colors[:, 1] = interp_G(x_fine)
    refined_colors[:, 2] = interp_B(x_fine)

    # cmap_greens = matplotlib.colors.ListedColormap(colors=colors_greens, name='greens', N=len(colors_greens))
    cmap_wgk = matplotlib.colors.ListedColormap(colors=refined_colors, name='wgk', N=len(refined_colors))

    # matplotlib.colormaps.register(cmap_greens)
    matplotlib.colormaps.register(cmap_wgk)

if __name__=='__main__':
    # Create and plot a colorbar with the selected colormap
    import matplotlib
    import matplotlib.pyplot as plt
    import colormaps as cm
    # Set matplotlib properties
    plt.rcParams.update({
        "font.family" : "Liberation Serif",
        "mathtext.fontset" : "dejavuserif",
        "axes.spines.top" : False,
        "axes.spines.right" : False
    })

    bar_type = 2

    if bar_type==1:
        fig = plt.figure(figsize=(5.5, 1.5))
        ax = fig.add_axes([0.2, 0.4, 0.6, 0.12])
        register_colormaps()
        cmap = 'wgk'
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, orientation='horizontal')
        cb.set_ticks([0, 1])
        cb.set_ticklabels([0, 1.0])
        ax.tick_params(axis='x', labelsize=36)
        ax.set_title(r"Concentration $c$ [-]", fontsize=36)
        fig_name = "../../output/illustrations/colorbars/colorbar_D3.png"
    elif bar_type==2:
        fig = plt.figure(figsize=(15, 4))
        ax = fig.add_axes([0.2, 0.4, 0.6, 0.12])
        cmap = cm.dense_r
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, orientation='horizontal')
        cb.set_ticks([0, 200/800, 400/800, 600/800,  1])
        cb.set_ticklabels([0, 200, 400, 600, 800])
        ax.tick_params(axis='x', labelsize=46)
        ax.set_title(r"Time to threshold [s]", fontsize=46, pad=20.0)
        fig_name = "../../output/illustrations/colorbars/colorbar_ttt.png"

    fig.tight_layout(pad=0)
    fig.savefig(fig_name, bbox_inches=[])
    plt.show()