from config.imports import *
from config.plot_defaults import *
from scipy.optimize import curve_fit

# exclude t < 1 from error
exclude_loading = True

# create figure to display convergence for different alpha in one summary plot
fig_all = plt.figure(1, figsize=(6,6))
ax_all = fig_all.add_subplot()

# get correct color cycle for current theme
style_colors = matplotlib.rcParams['axes.prop_cycle']
style_colors = [color for color in style_colors]

# loop over all alpha, corresponding folders and files have to exist
for idx, alpha in enumerate([0., 0.25, 0.5, 0.75, 1.]):

    print("#"*80)
    print("")
    print(f"Alpha = {alpha}")

    dir = config['outputfolder']
    tikz_folder = dir + "plots/"
    if not os.path.exists(tikz_folder):
        os.makedirs(tikz_folder)
    dir += "alpha" + str(alpha) + "/"

    # store full data and data with excluded loading in dictionaries using number of timesteps per second as keys
    data = {}
    data_full = {}
    numsteps = []
    numsteps_full = []

    if len(sys.argv) > 2:
        timestamp = sys.argv[2]
        filenames = glob.iglob(dir+f"tipdisplacement_{timestamp}*.txt")
    else:
        print("Please pass timestamp of convergence runs to use for analysis. Aborting!")
        sys.exit()

    for filename in filenames:
        try:
            tmp_num = int(float((filename.split("_")[-1]).rstrip(".txt")))//5*4
            tmp_num_full = int(float((filename.split("_")[-1]).rstrip(".txt")))
        except:
            continue

        numsteps.append(tmp_num)
        numsteps_full.append(tmp_num_full)
        tmp_data_full = np.insert(np.loadtxt(filename), 0, 0.)
        tmp_data = np.copy(tmp_data_full[len(tmp_data_full)//5:])
        data.update({tmp_num : tmp_data})
        data_full.update({tmp_num_full : tmp_data_full})

    print(numsteps)
    if len(numsteps) == 0:
        print(f"No data found for alpha={alpha}. Aborting!")
        sys.exit()

    if not exclude_loading:
        numsteps = numsteps_full
        data = data_full

    print("All solutions loaded.")

    numsteps = sorted(numsteps)
    numsteps_full = sorted(numsteps_full)

    reference_steps = numsteps.pop()
    reference_steps_full = numsteps_full.pop()
    reference = data[reference_steps]
    reference_full = data_full[reference_steps_full]   

    dt = 1/np.array(numsteps_full)
    print("Timesteps: ", dt)

    # plot solutions
    t = np.linspace(0, 5, len(reference_full))
    plotskip = len(reference_full)//500
    fig = plt.figure('Solutions', **figure_settings)
    plt.plot(t[::plotskip], reference_full[::plotskip], **plot_settings)
    plt.xlabel("$t$")
    plt.ylabel("$u_{tip}$")

    tikzplotlib.clean_figure(fig)
    tikz_settings['axis_width'] = "0.45*160mm"
    tikzplotlib.save(tikz_folder+f"plt_convergence_solution_{alpha}_{timestamp}.tex", **tikz_settings)
    plt.close(fig)

    error = []
    for i, numstep in enumerate(numsteps):

        # relative error in discrete L2-Norm
        error.append(np.linalg.norm(data[numstep]-reference[::reference_steps//numstep])/np.linalg.norm(reference[::reference_steps//numstep]))

        # relative error in Inf-Norm
        #error.append(np.linalg.norm(data[numstep] - reference[::skip], ord=np.Inf)/np.linalg.norm(reference[::reference_steps//numstep], ord=np.Inf))

        # uncomment following lines to plot difference of solutions
        #plt.plot(np.abs(data[numstep]-reference[::reference_steps//numstep]))
        #plt.show()

    # fit convergence order
    def f(dt, coeff, order):
        return np.log(dt)*order + coeff

    param, param_cov = curve_fit(f, dt[-4:], np.log(np.array(error)[-4:]))
    fit_error = np.exp(f(dt, param[0], param[1]))
    print("Fit: ", param)

    # convergence plot for single alpha
    fig = plt.figure('Solutions', figsize=(6,6))
    plt.plot(dt, error, "o--", label=f"Data", zorder=10, **plot_settings)
    plt.plot(dt, dt**2/dt[0]**2*error[0], label=f"Analytical order", c="tab:blue", linestyle="-", zorder=9, **plot_settings)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("$h$")
    plt.ylabel("$\mathcal{E}_r$")
    plt.legend()

    tikzplotlib.clean_figure(fig)
    tikz_settings['axis_width'] = "0.45*160mm"
    tikzplotlib.save(tikz_folder+f"plt_convergence_{alpha}_{timestamp}.tex", **tikz_settings)
    plt.close(fig)
    
    print("Plots for specific alpha created.")
    print("")

    # add to overview plot
    ax_all.plot(dt, error, "o--", color=style_colors[idx]['color'], label=f"$\\alpha = {alpha}$", zorder=10, **plot_settings)
    ax_all.plot(dt, dt**2/dt[0]**2*error[0], color=style_colors[idx]['color'], label=f"_Analytical order", linestyle="-", zorder=9, **plot_settings)

print("#"*80)
print("")

plt.figure(1)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("$h$")
plt.ylabel("$\mathcal{E}_r$")
#plt.title(f"Alpha= {alpha}")
plt.legend()

tikzplotlib.clean_figure(fig_all)
tikz_settings['axis_width'] = "0.7*160mm"
tikzplotlib.save(tikz_folder+f"plt_convergence_{timestamp}.tex", **tikz_settings)

print("Overview plot generated.")
plt.show()