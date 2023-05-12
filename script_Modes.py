from config.imports import *
from config.plot_defaults import *

# run simulation if data does not exist in given path
run = []

if not os.path.exists(config['outputfolder'] + "modes_continuous_loading" + ".pkl"):
    run.append(True)
if not os.path.exists(config['outputfolder'] + "modes_discontinuous_loading" + ".pkl"):
    run.append(False)

for continuos_loading in run:
        
    infmode = config.get('infmode', False)

    cutoff_time = 1.
    magnitude   = 1.
    tmax        = 4/5
    tzero       = 1.
    if continuos_loading:
        load_Bending = Expression(("0", "t <= tm ? p0*t/tm : (t <= tz ? p0*(1 - (t-tm)/(tz-tm)) : 0)", "0"), t=0, tm=tmax, tz=tzero, p0=magnitude, degree=0)
        print("Running simulation with continuous loading")
    else:
        load_Bending = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_time, p0=magnitude, degree=0)
        print("Running simulation with discontinuous loading")

    alpha = 0.7
    RA = RationalApproximation(alpha=alpha, tol=1e-4)
    parameters = list(RA.c) + list(RA.d)
    if infmode==True: parameters.append(RA.c_inf)
    kernel = SumOfExponentialsKernel(parameters=parameters)
    kernels = [kernel]

    Model = ViscoelasticityProblem(**config, kernels=kernels)
    Model.forward_solve(loading=load_Bending)

    if continuos_loading == True:
        file = config['outputfolder'] + "modes_continuous_loading"
    else:
        file = config['outputfolder'] + "modes_discontinuous_loading"

    save_data_modes(file, Model)

for continuos_loading in [True, False]:

    if continuos_loading == True:
        file = config['outputfolder'] + "modes_continuous_loading"
        plt_file = config['outputfolder'] + "plt_modes_continuous_loading.tex"
    else:
        file = config['outputfolder'] + "modes_discontinuous_loading"
        plt_file = config['outputfolder'] + "plt_modes_discontinuous_loading.tex"
    displacement, velocity, acceleration, modes = load_data(file)

    time_steps = np.linspace(0, config['FinalTime'], config['nTimeSteps']+1)[1:]

    nmodes = modes.shape[1]
    labels = [f"mode {i+1}" for i in range(nmodes)]
    labels.append("displacement (scaled)")
    labels.append("velocity (scaled)")
    labels.append("acceleration (scaled)")

    plt.figure('Norm of modes and solution', **figure_settings)
    plt.plot(time_steps, modes, "-", **plot_settings, zorder=10)
    plt.plot(time_steps, displacement/np.max(displacement)*np.max(modes), "-",  color="k", **plot_settings, zorder=-10)
    plt.plot(time_steps, velocity/np.max(velocity)*np.max(modes), "--",  color="k", **plot_settings, zorder=-10)
    plt.plot(time_steps, acceleration/np.max(acceleration)*np.max(modes), ":",  color="k", **plot_settings, zorder=-10)

    plt.legend(labels)
    plt.ylabel(r"Norm")
    plt.xlabel(r"$t$")
    plt.xlim([0, 5])

    tikz_settings['axis_height'] = "0.5*160mm"
    tikzplotlib.save(plt_file, **tikz_settings)
    plt.close()