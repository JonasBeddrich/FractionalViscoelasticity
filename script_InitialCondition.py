from matplotlib.pyplot import figure
from config import *
import copy
import os
    
tikz_folder = config['outputfolder']+'IC/'
if not os.path.exists(tikz_folder):
    os.makedirs(tikz_folder)

style_colors = matplotlib.rcParams['axes.prop_cycle']
style_colors = [color for color in style_colors]
color1 = style_colors[0]['color']
color2 = style_colors[1]['color']

# Simulation settings
fg_export = True  ### write results on the disk (True) or only solve (False)
config['export_vtk'] = False

time_steps = 100 # time steps per second

# Loading (continuous)
magnitude = 1.
tmax = 4/5
tzero = 1.
load_Bending = Expression(("0", "t <= tm ? p0*t/tm : (t <= tz ? p0*(1 - (t-tm)/(tz-tm)) : 0)", "0"), t=0, tm=tmax, tz=tzero, p0=magnitude, degree=0)

"""
==================================================================================================================
Kernel and its rational approximation
==================================================================================================================
"""

infmode = config.get('infmode', False)
config['two_kernels'] = False
zener_kernel = config.get('zener_kernel', False)
zener_kernel = False

for alpha in [0.05, 0.5, 0.75, 0.95]:

    # set loading profile
    config['loading'] = load_Bending

    if zener_kernel:
        print()
        print(f"ZENER KERNEL - ALPHA = {alpha}")
        tau_eps = .2
        tau_sig = .1
        TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha)/(x**-alpha + 1/tau_sig)
        RA = RationalApproximation(alpha=alpha, TargetFunction=TargetFunction, tol=1e-4)
        parameters = list(RA.c) + list(RA.d)
        if infmode==True: parameters.append(RA.c_inf)
    else:
        print()
        print(f"FRACTIONAL KERNEL - ALPHA = {alpha}")
        RA = RationalApproximation(alpha=alpha, tol=1e-4)
        parameters = list(RA.c) + list(RA.d)
        if infmode==True: parameters.append(RA.c_inf)

    """
    ==================================================================================================================
    Generating Initial Condition
    ==================================================================================================================
    """

    print()
    print("================================")
    print("  CALCULATING INITIAL CONDITION")
    print("================================")

    config["FinalTime"] = 1
    config["nTimeSteps"] = time_steps
    kernels_IC = [SumOfExponentialsKernel(parameters=parameters)]
    IC = ViscoelasticityProblem(**config, kernels=kernels_IC)
    IC.forward_solve(loading=config.get("loading"))

    """
    ==================================================================================================================
    Continuing with correct condition
    ==================================================================================================================
    """

    print()
    print("================================")
    print("   CONTINUE CORRECT CONDITION")
    print("================================")

    config["FinalTime"] = 4
    config["nTimeSteps"] = 4*time_steps
    config["loading"] = Expression(("0", "0", "0"), degree=0)

    kernels_correct = [SumOfExponentialsKernel(parameters=parameters)]
    correct = ViscoelasticityProblem(**config, kernels=copy.deepcopy(kernels_correct))
    correct.kernels[0].modes = copy.deepcopy(IC.kernels[0].modes)
    correct.kernels[0].F_old = copy.deepcopy(IC.kernels[0].F_old)

    correct.u = copy.deepcopy(IC.u)
    correct.v = copy.deepcopy(IC.v)
    correct.a = copy.deepcopy(IC.a)
    correct.history = copy.deepcopy(IC.history)

    correct.forward_solve(loading=config.get("loading"))

    """
    ==================================================================================================================
    Continuing with wrong condition
    ==================================================================================================================
    """

    print()
    print("================================")
    print("   CONTINUE WRONG CONDITION")
    print("================================")

    kernels_wrong = [SumOfExponentialsKernel(parameters=parameters)]
    wrong = ViscoelasticityProblem(**config, kernels=copy.deepcopy(kernels_wrong))

    wrong.kernels[0].modes = copy.deepcopy(IC.kernels[0].modes)
    wrong.kernels[0].F_old = copy.deepcopy(IC.kernels[0].F_old)
    wrong.kernels[0].modes = 0
    wrong.kernels[0].F_old = 0

    wrong.u = copy.deepcopy(IC.u)
    wrong.v = copy.deepcopy(IC.v)
    wrong.a = copy.deepcopy(IC.a)

    wrong.forward_solve(loading=config.get("loading"))

    """
    ==================================================================================================================
    Display
    ==================================================================================================================
    """

    with torch.no_grad():
        fig = plt.figure('Tip displacement', **figure_settings)
        plt.plot(IC.time_steps, IC.observations.numpy(), label="initial", **plot_settings, color="k", zorder=10)
        plt.plot(IC.time_steps, IC.velocity_norm, **plot_settings, color="k", linestyle="--", zorder=10)
        plt.plot(IC.time_steps, IC.acceleration_norm, **plot_settings, color="k", linestyle=":", zorder=10)

        plt.plot(IC.time_steps[-1], IC.observations.numpy()[-1], **plot_settings, color="k", marker="o", zorder=10)
        #plt.plot(IC.time_steps[-1], IC.velocity_norm[-1], **plot_settings, color="k", marker="o", zorder=10)
        #plt.plot(IC.time_steps[-1], IC.acceleration_norm[-1], **plot_settings, color="k", marker="o", zorder=10)

        plt.plot([IC.time_steps[-1], *(correct.time_steps + 1.)], [IC.observations.numpy()[-1], *correct.observations.numpy()], color=color1, label="correct", **plot_settings)
        plt.plot([IC.time_steps[-1], *(correct.time_steps + 1.)], [IC.velocity_norm[-1], *correct.velocity_norm], color=color1, **plot_settings, linestyle="--")
        plt.plot([IC.time_steps[-1], *(correct.time_steps + 1.)], [IC.acceleration_norm[-1], *correct.acceleration_norm], color=color1, linestyle=":", **plot_settings)

        plt.plot([IC.time_steps[-1], *(wrong.time_steps + 1.)], [IC.observations.numpy()[-1], *wrong.observations.numpy()], color=color2, label="zeroed", **plot_settings)
        plt.plot([IC.time_steps[-1], *(wrong.time_steps + 1.)], [IC.velocity_norm[-1], *wrong.velocity_norm], color=color2, linestyle="--", **plot_settings)
        plt.plot([IC.time_steps[-1], *(wrong.time_steps + 1.)], [IC.acceleration_norm[-1], *wrong.acceleration_norm], color=color2, linestyle=":", **plot_settings)

        plt.axvline([1], ymin=-0.05, ymax=1.5, color="grey", linestyle="--", **plot_settings, zorder=-10)
        plt.xlim([0, 5])
        plt.legend()
        plt.ylabel(r"Tip displacement")
        plt.xlabel(r"$t$")

        #tikzplotlib.clean_figure(fig)
        tikzplotlib.save(tikz_folder+f"plt_ic_displacement_alpha{alpha}.tex", **tikz_settings)
        plt.close()

        fig = plt.figure('Tip displacement', **figure_settings)
        plt.plot(IC.time_steps, IC.modes_norm, **plot_settings, color="k", zorder=10)

        tmp = IC.modes_norm[-1, :]
        correct_modes = np.vstack((tmp, correct.modes_norm))
        wrong_modes   = np.vstack((np.zeros_like(tmp), wrong.modes_norm))
        plt.plot([IC.time_steps[-1], *(correct.time_steps + 1.)], correct_modes, **plot_settings, zorder=10, color=color1)
        plt.plot([IC.time_steps[-1], *(correct.time_steps + 1.)], wrong_modes, **plot_settings, zorder=10, color=color2)
        plt.ylabel(r"Norm of modes")
        plt.xlabel(r"$t$")
        plt.axvline([1], ymin=-0.05, ymax=1.5, color="grey", linestyle="--", **plot_settings, zorder=-10)
        plt.xlim([0, 5])

        plt.plot([1], [0], color=color1, label="correct", zorder=-10)
        plt.plot([1], [0], color=color2, label="zeroed", zorder=-10)
        plt.plot([1], [0], color="k", label="initial", zorder=-10)
        plt.legend()

        tikz_settings['axis_width'] = '0.45*160mm'
        tikzplotlib.save(tikz_folder+f"plt_ic_modes_alpha{alpha}.tex", **tikz_settings)
        plt.close()