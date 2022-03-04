


from matplotlib.pyplot import figure
from config import *
import copy

fg_export = True  ### write results on the disk (True) or only solve (False)
config['export_vtk'] = False

####### LOADING #############
magnitude      = 1.
tmax = 4/5
tzero = 1.
load_Bending = Expression(("0", "t <= tm ? p0*t/tm : (t <= tz ? p0*(1 - (t-tm)/(tz-tm)) : 0)", "0"), t=0, tm=tmax, tz=tzero, p0=magnitude, degree=0)
config['loading'] = load_Bending

zener_kernel = False

"""
==================================================================================================================
Kernel and its rational approximation
==================================================================================================================
"""

infmode = config.get('infmode', False)

assert config['two_kernels']==False
   
if zener_kernel:
    alpha = 0.5
    tau_eps = .2
    tau_sig = .1
    TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha)/(x**-alpha + 1/tau_sig)
    RA = RationalApproximation(alpha=alpha, TargetFunction=TargetFunction)
    parameters = list(RA.c) + list(RA.d)
    if infmode==True: parameters.append(RA.c_inf)
else:
    alpha2 = 0.7
    RA = RationalApproximation(alpha=alpha2)
    parameters = list(RA.c) + list(RA.d)
    if infmode==True: parameters.append(RA.c_inf)
    # parameters2 = np.array([ 0.32337598,  0.41615834,  0.47182692,  1.03023015,  0.24184555,
    #     0.85714041, -0.11852263,  3.39125769,  0.10000886,  0.26773023,
    #     0.12313697,  0.70114342,  1.4451129 ,  3.89811345, 10.0704782 ,
    #    29.40964287])**2

"""
==================================================================================================================
Generating Initial Condition
==================================================================================================================
"""

print()
print()
print("================================")
print("  CALCULATING INITIAL CONDITION")
print("================================")

config["FinalTime"] = 1
config["nTimeSteps"] = 20
kernels_IC = [SumOfExponentialsKernel(parameters=parameters)]
IC = ViscoelasticityProblem(**config, kernels=kernels_IC)

#def Forward():
#    Model.forward_solve()
#    obs = Model.observations
#    return obs.numpy()
IC.forward_solve(loading=config.get("loading"))

"""
==================================================================================================================
Continuing with correct condition
==================================================================================================================
"""

print()
print()
print("================================")
print("   CONTINUE CORRECT CONDITION")
print("================================")

config["FinalTime"] = 4
config["nTimeSteps"] = 80
config["loading"] = Expression(("0", "0", "0"), degree=0)

kernels_correct = [SumOfExponentialsKernel(parameters=parameters)]
correct = ViscoelasticityProblem(**config, kernels=copy.deepcopy(kernels_correct))
correct.kernels[0].modes = copy.deepcopy(IC.kernels[0].modes)
correct.kernels[0].F_old = copy.deepcopy(IC.kernels[0].F_old)

correct.u = copy.deepcopy(IC.u)
correct.v = copy.deepcopy(IC.v)
correct.a = copy.deepcopy(IC.a)

correct.forward_solve(loading=config.get("loading"))

"""
==================================================================================================================
Continuing with wrong condition
==================================================================================================================
"""

print()
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

import tikzplotlib
import matplotlib

tikz_folder = config['outputfolder']

# plt.style.use("ggplot")
plt.style.use("bmh")
font = {
    # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)
figure_settings = {'figsize' : (10,6)}
plot_settings = {'markersize' : 2}
legend_settings = {'loc' : 'center left', 'bbox_to_anchor' : (1.1, 0.5)}
tikz_settings = {'axis_width' : '\\textwidth'}

with torch.no_grad():
    plt.figure('Tip displacement', **figure_settings)
    plt.plot(IC.time_steps, IC.observations.numpy(), label="Initial Condition", **plot_settings, color="k", zorder=10)
    plt.plot(IC.time_steps[-1], IC.observations.numpy()[-1], **plot_settings, color="k", marker="o", zorder=10)
    plt.plot([IC.time_steps[-1], *(correct.time_steps + 1.)], [IC.observations.numpy()[-1], *correct.observations.numpy()], label="Correct modes", **plot_settings)
    plt.plot([IC.time_steps[-1], *(wrong.time_steps + 1.)], [IC.observations.numpy()[-1], *wrong.observations.numpy()], label="Zeroed modes", **plot_settings)
    plt.axvline([1], ymin=-0.05, ymax=1.5, color="grey", linestyle="--", **plot_settings, zorder=-10)
    plt.xlim([0, 5])
    plt.legend()
    plt.ylabel(r"Tip displacement")
    plt.xlabel(r"$t$")

    tikzplotlib.save(tikz_folder+"plt_initialcondition.tex", **tikz_settings)
    plt.show()

    # model.kernel.plot()