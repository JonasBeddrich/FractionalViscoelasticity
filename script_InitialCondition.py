from config.imports import *
from config.plot_defaults import *
from matplotlib.pyplot import figure
import copy
    
tikz_folder = config['outputfolder']
timestamp = time.strftime("%Y%m%d-%H%M")

style_colors = matplotlib.rcParams['axes.prop_cycle']
style_colors = [color for color in style_colors]
color1 = style_colors[0]['color']
color2 = style_colors[1]['color']

final_time = config["FinalTime"]
time_steps = config["nTimeSteps"]
loading = config["loading"]

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

    # reset loading
    config['loading'] = loading

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
    Continuing with correct modes
    ==================================================================================================================
    """

    print()
    print("================================")
    print("   CONTINUE CORRECT MODES")
    print("================================")

    config["FinalTime"] = final_time-1
    config["nTimeSteps"] = (final_time-1)*time_steps
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
    Continuing with zeroed modes
    ==================================================================================================================
    """

    print()
    print("================================")
    print("   CONTINUE ZEROED MODES")
    print("================================")

    kernels_zeroed = [SumOfExponentialsKernel(parameters=parameters)]
    zeroed = ViscoelasticityProblem(**config, kernels=copy.deepcopy(kernels_zeroed))

    zeroed.kernels[0].modes = copy.deepcopy(IC.kernels[0].modes)
    zeroed.kernels[0].F_old = copy.deepcopy(IC.kernels[0].F_old)
    zeroed.kernels[0].modes = 0
    zeroed.kernels[0].F_old = 0

    zeroed.u = copy.deepcopy(IC.u)
    zeroed.v = copy.deepcopy(IC.v)
    zeroed.a = copy.deepcopy(IC.a)

    zeroed.forward_solve(loading=config.get("loading"))

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

        plt.plot([IC.time_steps[-1], *(zeroed.time_steps + 1.)], [IC.observations.numpy()[-1], *zeroed.observations.numpy()], color=color2, label="zeroed", **plot_settings)
        plt.plot([IC.time_steps[-1], *(zeroed.time_steps + 1.)], [IC.velocity_norm[-1], *zeroed.velocity_norm], color=color2, linestyle="--", **plot_settings)
        plt.plot([IC.time_steps[-1], *(zeroed.time_steps + 1.)], [IC.acceleration_norm[-1], *zeroed.acceleration_norm], color=color2, linestyle=":", **plot_settings)

        plt.axvline([1], ymin=-0.05, ymax=1.5, color="grey", linestyle="--", **plot_settings, zorder=-10)
        plt.xlim([0, final_time])
        plt.legend()
        plt.ylabel(r"Tip displacement")
        plt.xlabel(r"$t$")

        #tikzplotlib.clean_figure(fig)
        tikzplotlib.save(tikz_folder+f"plt_ic_displacement_alpha{alpha}_"+timestamp+".tex", **tikz_settings)
        plt.savefig(tikz_folder+f"plt_ic_displacement_alpha{alpha}_"+timestamp+".pdf")
        plt.close()

        fig = plt.figure('Tip displacement', **figure_settings)
        plt.plot(IC.time_steps, IC.modes_norm, **plot_settings, color="k", zorder=10)

        tmp = IC.modes_norm[-1, :]
        correct_modes = np.vstack((tmp, correct.modes_norm))
        zeroed_modes   = np.vstack((np.zeros_like(tmp), zeroed.modes_norm))
        plt.plot([IC.time_steps[-1], *(correct.time_steps + 1.)], correct_modes, **plot_settings, zorder=10, color=color1)
        plt.plot([IC.time_steps[-1], *(correct.time_steps + 1.)], zeroed_modes, **plot_settings, zorder=10, color=color2)
        plt.ylabel(r"Norm of modes")
        plt.xlabel(r"$t$")
        plt.axvline([1], ymin=-0.05, ymax=1.5, color="grey", linestyle="--", **plot_settings, zorder=-10)
        plt.xlim([0, final_time])

        plt.plot([1], [0], color=color1, label="correct", zorder=-10)
        plt.plot([1], [0], color=color2, label="zeroed", zorder=-10)
        plt.plot([1], [0], color="k", label="initial", zorder=-10)
        plt.legend()

        tikz_settings['axis_width'] = '0.45*160mm'
        tikzplotlib.save(tikz_folder+f"plt_ic_modes_alpha{alpha}_"+timestamp+".tex", **tikz_settings)
        plt.savefig(tikz_folder+f"plt_ic_modes_alpha{alpha}_"+timestamp+".pdf")
        plt.close()