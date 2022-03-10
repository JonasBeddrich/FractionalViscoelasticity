

from config import *

tikz_folder = config['outputfolder']


"""
==================================================================================================================
Load data
==================================================================================================================
"""

(displacement_norm, velocity_norm, acceleration_norm, modes_norm) = load_data(config['inputfolder']+"modes")

time_steps = np.linspace(0, config['FinalTime'], config['nTimeSteps']+1)[1:]


"""
==================================================================================================================
FIGURES
==================================================================================================================
"""

import tikzplotlib
import matplotlib
# plt.style.use("ggplot")
plt.style.use("bmh")
font = {
    # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)


figure_settings = {
    'figsize'   :   (10,6),
}

plot_settings = {
    'markersize'   :   2,
}

legend_settings = {
    'loc'             :   'center left',
    'bbox_to_anchor'  :   (1.1, 0.5),
}

tikz_settings = {
    'axis_width'  :   '\\textwidth',
    'axis_height'      :   '.5\\textwidth'
}

with torch.no_grad():

    """
    ==================================================================================================================
    Figure 1: Modes
    ==================================================================================================================
    """
    nmodes = modes_norm.shape[1]
    labels = [f"Mode {i+1}" for i in range(nmodes)]
    labels.append("Displacement (scaled)")
    labels.append("Velocity (scaled)")
    labels.append("Acceleration (scaled)")

    maxnorm = np.max(modes_norm)
    displacement_norm *= maxnorm*1.1/np.max(displacement_norm)
    velocity_norm     *= maxnorm*1.1/np.max(velocity_norm)
    acceleration_norm *= maxnorm*1.1/np.max(acceleration_norm)

    plt.figure('Norm of modes and solution', **figure_settings)
    plt.plot(time_steps, modes_norm, "-", **plot_settings)
    plt.plot(time_steps, displacement_norm, ":",  color="k", **plot_settings)
    plt.plot(time_steps, velocity_norm, "--",  color="k", **plot_settings)
    plt.plot(time_steps, acceleration_norm, "-.",  color="k", **plot_settings)

    plt.legend(labels)
    plt.ylabel(r"Norm")
    plt.xlabel(r"$t$")
    plt.xlim([0, 4])
    tikzplotlib.save(tikz_folder+"plt_modes_norm.tex", **tikz_settings)
    plt.show()

    