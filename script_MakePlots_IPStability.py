from config.imports import * 
from config.plot_defaults import *
import matplotlib
from scipy.optimize import minimize, Bounds

tikz_folder = config['outputfolder']


"""
==================================================================================================================
Load data
==================================================================================================================
"""

if len(sys.argv) >= 3:
    timestamp = sys.argv[2]
    filename = config['inputfolder']+"tip_displacement_noisy_"+timestamp+".csv"
else:
    filename = max(glob.iglob(config['inputfolder']+"tip_displacement_noisy_*.csv"), key=os.path.getctime)
    timestamp = filename[-17:-4]
tip_meas = np.loadtxt(filename)

tip_true, EnergyElastic_true, EnergyKinetic_true, EnergyViscous_true, theta_true = load_data(config['inputfolder']+"model_target_"+timestamp)

pred = {}
alphas = []
names = ["tip", "EnergyElastic", "EnergyKinetic", "EnergyViscous", "theta", "convergence"]

tmp = {}
data = load_data(config['inputfolder']+"model_target_"+timestamp)
for i, el in enumerate(data):
    tmp[names[i]] = el
kernel = SumOfExponentialsKernel(parameters = tmp["theta"])
tmp["kernel"] = kernel
pred["true"] = tmp

for file in glob.iglob(config['inputfolder']+f"model_predict_{timestamp}_*"):
    tmp = {}
    data = load_data(file)
    for i, el in enumerate(data):
        tmp[names[i]] = el
    # get fractional parameter
    alpha = float(file.split("_")[-1][:-4])
    alphas.append(alpha)
    # construct kernel object
    with torch.no_grad():
        theta_init = np.array(tmp["convergence"]["parameters"])[0].flatten()
    kernel_init = SumOfExponentialsKernel(parameters = theta_init)
    kernel = SumOfExponentialsKernel(parameters = np.array([i.detach().numpy() for i in tmp["theta"][0]]))
    tmp["kernel_init"] = kernel_init
    tmp["kernel"] = kernel
    # store in global dict
    pred[alpha] = tmp

for i, file in enumerate(glob.iglob(config['inputfolder']+f"tip_displacement_initial_{timestamp}_*")):
    pred[alphas[i]]["tip_init"] = np.loadtxt(file)

time_steps = np.linspace(0, config['FinalTime'], config['nTimeSteps']+1)[1:]
time_steps_meas = time_steps[:tip_meas.size]

alphas.sort()

# Find corresponding alpha for predicted kernel
"""
t = np.geomspace(0.04, 4, 100)

infmode = config.get('infmode', False)
nModes = config.get("nModes", 10)
tau_eps = config.get("tau_eps", 0.2)
tau_sig = config.get("tau_sig", 0.1)
TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha)/(x**-alpha + 1/tau_sig)
tol = 1e-6 # determines number of modes

alpha = 0.8
data = pred[alpha]["kernel"].eval_func(t)

def objective(alpha, data, tol):
    TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha)/(x**-alpha + 1/tau_sig)
    RA = RationalApproximation(alpha=alpha, TargetFunction=TargetFunction, tol=tol)
    parameters = list(RA.c) + list(RA.d)
    print(RA.nModes)
    if infmode==True: parameters.append(RA.c_inf)
    kernel_test = SumOfExponentialsKernel(parameters=parameters)
    test = kernel_test.eval_func(t)
    return np.linalg.norm(data - test, ord=2)

bound = Bounds(0, 1)
opt = minimize(objective, 0.5, args=(data, tol), bounds=bound)
print(opt)

alpha = opt.x
TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha)/(x**-alpha + 1/tau_sig)
RA = RationalApproximation(alpha=alpha, TargetFunction=TargetFunction, tol=tol)
parameters = list(RA.c) + list(RA.d)
if infmode==True: parameters.append(RA.c_inf)
kernel_test = SumOfExponentialsKernel(parameters=parameters)
test = kernel_test.eval_func(t)
true = pred["true"]["kernel"].eval_func(t)
plt.plot(t, true, label="True")
plt.plot(t, data, label="Prediction")
plt.plot(t, test, label="Alpha fit")
plt.legend()
plt.xscale("log")
plt.show()
sys.exit()
"""



t = np.geomspace(0.05, 2, 100)
true_kernel = pred["true"]["kernel"]

for alpha in alphas:
    p = pred[alpha]["convergence"]["parameters"]
    num_steps = len(p)
    cmap = matplotlib.colormaps["viridis_r"]
    colors = [cmap(i/num_steps) for i in range(num_steps)]
    for i in range(num_steps):
        with torch.no_grad():
            theta = np.array(pred[alpha]["convergence"]["parameters"])[i].flatten()
        kernel = SumOfExponentialsKernel(parameters = theta)
        #plt.plot(t, kernel.eval_func(t), color=colors[i])
        plt.plot(t, (kernel.eval_func(t) - true_kernel.eval_func(t))/true_kernel.eval_func(t), color=colors[i]) # relative error

    #plt.plot(t, true_kernel.eval_func(t), "--", color="red", label=f"True", **plot_settings)
    plt.plot(t, t*0, "--", color="red", label=f"True", **plot_settings) # relative error

    plt.xscale('log')
    plt.title(fr"Kernel evolution $\alpha_0={alpha}$")
    plt.ylabel(r"$k_{pred}(t)$")
    cmap = "viridis_r"
    norm = matplotlib.colors.Normalize(vmin=0,vmax=num_steps-1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="Iteration", orientation="horizontal")
    plt.ylabel(r"$\frac{k_{pred}(t) - k(t)}{k(t)}$") # relative error
    plt.xlabel(r"$t$")
    #plt.savefig(config['outputfolder']+f"plt_kernel_evolution_alpha{alpha}.pdf", bbox_inches="tight")
    plt.show()

sys.exit()
"""
cmap = matplotlib.colormaps["tab10"]
colors = [cmap(i) for i in range(10)]


plt.plot(time_steps, pred["true"]["tip"], "-", color="grey", label=f"True")
for i, alpha in enumerate(alphas):
    plt.plot(time_steps, pred[alpha]["tip_init"], "--", color=colors[i], label=f"init {alpha}")
    plt.plot(time_steps, pred[alpha]["tip"], "-", color=colors[i], label=f"pred {alpha}")

plt.xlabel(r"$t$")
plt.title("Tip displacement")
plt.legend()
plt.savefig(config['outputfolder']+f"plt_tipdisplacement_{timestamp}.pdf", bbox_inches="tight")
plt.show()

t = np.geomspace(0.04, 4, 100)
plt.plot(t, pred["true"]["kernel"].eval_func(t), "-", color="grey", label=f"True", **plot_settings)
for i, alpha in enumerate(alphas):
    plt.plot(t, pred[alpha]["kernel"].eval_func(t), "-", color=colors[i], label=f"pred {alpha}", **plot_settings)
    plt.plot(t, pred[alpha]["kernel_init"].eval_func(t), "--", color=colors[i], label=f"init {alpha}", **plot_settings)

plt.xscale('log')
plt.ylabel(r"$k(t)$")
plt.xlabel(r"$t$")
plt.legend()
plt.savefig(config['outputfolder']+f"plt_kernels_{timestamp}.pdf", bbox_inches="tight")
plt.show()

maxiter = 0
for i, alpha in enumerate(alphas):
    loss = pred[alpha]["convergence"]["loss"]
    if len(loss) > maxiter:
        maxiter = len(loss)
    plt.plot(loss, "-", color=colors[i], label=f"Loss {alpha}")
    plt.plot(pred[alpha]["convergence"]["grad"], "--", color=colors[i], label=f"Gradient {alpha}")
plt.legend()
plt.xlim([0, maxiter-1])
plt.yscale("log")
plt.xlabel("Iteration")
plt.savefig(config['outputfolder']+f"plt_convergence_loss_{timestamp}.pdf", bbox_inches="tight")
plt.show()

figweight, axweight = plt.subplots(1, 1)
figexponent, axexponent = plt.subplots(1, 1)

p_true = np.array(pred["true"]["theta"])
nmodes = p_true.shape[-1]//2
for j in range(nmodes):
    axweight.hlines(p_true[j]/(1+p_true[j+nmodes]), ls="--", color="grey", xmin=0, xmax=maxiter, zorder=-10)
    axexponent.hlines(p_true[j+nmodes]/(1+p_true[j+nmodes]), ls="--", color="grey", xmin=0, xmax=maxiter, zorder=-10)
axweight.hlines([], ls="--", color="grey", xmin=0, xmax=maxiter, label="True")
axexponent.hlines([], ls="--", color="grey", xmin=0, xmax=maxiter, label="True")

for i, alpha in enumerate(alphas):
    parameters = pred[alpha]["convergence"]["parameters"]
    with torch.no_grad():
        p = np.array(parameters)
    nmodes = p.shape[-1]//2
    axweight.plot([], [], color=colors[i], label=f"{alpha}")
    axexponent.plot([], [], color=colors[i], label=f"{alpha}")
    for j in range(nmodes):
        axweight.plot(p[:,0,j]/(1+p[:,0,j+nmodes]), color=colors[i], **plot_settings)
        axexponent.plot(p[:,0,j+nmodes]/(1+p[:,0,j+nmodes]), color=colors[i], **plot_settings)

axweight.set_xlim([0, maxiter-1])
axweight.set_xlabel("Iteration")
axweight.set_ylabel(r"$\frac{w}{1+\lambda}$")

axexponent.set_xlim([0, maxiter-1])
axexponent.set_xlabel("Iteration")
axexponent.set_ylabel(r"$\frac{\lambda}{1+\lambda}$")

figweight.legend()
figweight.suptitle("Weight Convergence")
figexponent.legend()
figexponent.suptitle("Exponent Convergence")

figweight.savefig(config['outputfolder']+f"plt_convergence_weights_{timestamp}.pdf", bbox_inches="tight")
figexponent.savefig(config['outputfolder']+f"plt_convergence_exponents_{timestamp}.pdf", bbox_inches="tight")

plt.show()
"""

sys.exit()



"""
==================================================================================================================
FIGURES
==================================================================================================================
"""

tikz_settings['axis_width'] = "0.45*160mm"

with torch.no_grad():

    """
    ==================================================================================================================
    Figure 1: Observations
    ==================================================================================================================
    """
    fig = plt.figure('Tip displacement', **figure_settings)
    # plt.title('Tip displacement')
    plt.plot(time_steps, tip_init, "-",  color="gray", label="initial", **plot_settings)
    plt.plot(time_steps, tip_pred, "r-",  label="predict", **plot_settings)
    plt.plot(time_steps, tip_true, "b--", label="truth", **plot_settings)
    plt.plot(time_steps_meas, tip_meas, "ko:", label="data", **plot_settings)
    plt.legend()
    plt.ylabel(r"Tip displacement")
    plt.xlabel(r"$t$")

    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(tikz_folder+"plt_tip_displacement_"+timestamp+".tex", **tikz_settings)


    """
    ==================================================================================================================
    Figure 2: Energies
    ==================================================================================================================
    """
    fig = plt.figure('Energies', **figure_settings)
    # plt.title('Energies')

    plt.plot(time_steps, EnergyElastic_pred, "-", color='red', label="Elastic energy (predict)", **plot_settings)
    plt.plot(time_steps, EnergyKinetic_pred, "-", color='orange', label="Kinetic energy (predict)", **plot_settings)
    # plt.plot(time_steps, EnergyTotal_pred, "-", color='brown', label="Total energy (predict)")

    plt.plot(time_steps, EnergyElastic_true, "--", color='blue', label="Elastic energy (truth)", **plot_settings)
    plt.plot(time_steps, EnergyKinetic_true, "--", color='cyan', label="Kinetic energy (truth)", **plot_settings)
    # plt.plot(time_steps, EnergyTotal_true, "--", color='magenta', label="Total energy (truth)", **plot_settings)

    plt.grid(True, which='both')
    plt.ylabel(r"Energy")
    plt.xlabel(r"$t$")
    plt.legend()

    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(tikz_folder+"plt_energies_"+timestamp+".tex", **tikz_settings)



    """
    ==================================================================================================================
    Figure 3: Kernels
    ==================================================================================================================
    """
    fig = plt.figure('Kernels', **figure_settings)
    # plt.title('Kernels')
    t = np.geomspace(0.04, 4, 100)
    plt.plot(t, kernel_init.eval_func(t), "-", color="gray", label="sum-of-exponentials (initial guess)", **plot_settings)
    plt.plot(t, kernel_pred.eval_func(t), "r-", label="sum-of-exponentials (predict)", **plot_settings)
    plt.plot(t, kernel_true.eval_func(t), "b-", label="sum-of-exponentials (truth)", **plot_settings)
    #plt.plot(t, kernel_frac_init(t), "o", color="gray", label=r"fractional $\alpha=0.5$", **plot_settings)
    #plt.plot(t, kernel_frac(t), "bo", label=r"fractional $\alpha=0.7$", **plot_settings)
    plt.xscale('log')
    plt.ylabel(r"$k(t)$")
    plt.xlabel(r"$t$")
    plt.legend()

    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(tikz_folder+"plt_kernels_"+timestamp+".tex", **tikz_settings)


    """
    ==================================================================================================================
    Figure 4: Parameters convergence
    ==================================================================================================================
    """
    
    parameters = convergence_history["parameters"]
    p = np.array(parameters)
    nmodes = p.shape[-1]//2
    #nsteps = len(parameters)
    #p = torch.stack(parameters).reshape([nsteps,2,-1]).detach().numpy()

    fig = plt.figure('Parameters convergence: Weights', **figure_settings)
    # plt.title('Parameters convergence: Weights')
    for i in range(nmodes):
        plt.plot(p[:,0,i]/(1+p[:,0,i+nmodes]), label=r'$w_{{%(i)d}}$' % {'i' : i+1}, **plot_settings)
    plt.ylabel(r"$\frac{w}{1+\lambda}$")
    plt.xlabel("Iteration")
    plt.legend()

    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(tikz_folder+"plt_weights_convergence_"+timestamp+".tex", **tikz_settings)
    # plt.yscale('log')


    fig = plt.figure('Parameters convergence: Exponents', **figure_settings)
    # plt.title('Parameters convergence: Exponents')
    for i in range(nmodes):
        plt.plot(p[:,0,i+nmodes]/(1+p[:,0,i+nmodes]), label=r'$\lambda_{{%(i)d}}$' % {'i' : i+1}, **plot_settings)
    # plt.yscale('log')
    plt.ylabel(r"$\frac{\lambda}{1+\lambda}$")
    plt.xlabel("Iteration")
    plt.legend()

    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(tikz_folder+"plt_exponents_convergence_"+timestamp+".tex", **tikz_settings)
    
    if len(p)%2!=0:
        fig = plt.figure('Parameters convergence: Infmode', **figure_settings)
        # plt.title('Parameters convergence: Exponents')
        plt.plot(p[:,0,-1]/(1+p[:,0,-1]), **plot_settings)
        # plt.yscale('log')
        plt.ylabel(r"$w_\infty$")
        plt.xlabel("Iteration")
        plt.legend()

        tikzplotlib.clean_figure(fig)
        tikzplotlib.save(tikz_folder+"plt_infmode_convergence_"+timestamp+".tex", **tikz_settings)


    """
    ==================================================================================================================
    Figure 5: Convergence
    ==================================================================================================================
    """

    loss = convergence_history["loss"]
    grad = convergence_history["grad"]

    plt.figure("Convergence", **figure_settings)
    plt.plot(loss, label="Loss", **plot_settings)
    plt.plot(grad, label="Gradient", **plot_settings)
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Iteration")

    """
    ==================================================================================================================
    """

    plt.show()




