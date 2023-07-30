from config.imports import * 
from config.plot_defaults import *
import matplotlib
from scipy.optimize import minimize, Bounds

if len(sys.argv) >= 3:
    folder = sys.argv[2] + "/"
    config['inputfolder'] = os.path.join(config['inputfolder'], folder)
    config['outputfolder'] = os.path.join(config['outputfolder'], folder)

tikz_folder = config['outputfolder']

"""
==================================================================================================================
Load data
==================================================================================================================
"""

pred = {}
runs = []
names = ["tip", "EnergyElastic", "EnergyKinetic", "EnergyViscous", "theta", "convergence"]

for file in glob.iglob(config['inputfolder']+f"model_predict_*"):
    tmp = {}
    data = load_data(file)
    for i, el in enumerate(data):
        tmp[names[i]] = el
    run = int(file.split("_")[-2])
    runs.append(run)
    # construct kernel object
    kernel = SumOfExponentialsKernel(parameters = np.array([i.detach().numpy() for i in tmp["theta"][0]]))
    tmp["kernel"] = kernel
    # store in global dict
    pred[run] = tmp

runs.sort()

# Find corresponding alpha for predicted kernel
t = np.geomspace(0.05, 2, 100)

infmode = config.get('infmode', False)
nModes = config.get("nModes", 10)
tau_eps = config.get("tau_eps", 0.2)
tau_sig = config.get("tau_sig", 0.1)
alpha = 0.5
TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha)/(x**-alpha + 1/tau_sig)
tol = 1e-6 # determines number of modes
RA = RationalApproximation(alpha=alpha, TargetFunction=TargetFunction, tol=tol)
parameters = list(RA.c) + list(RA.d)
if infmode==True: parameters.append(RA.c_inf)
kernel_test = SumOfExponentialsKernel(parameters=parameters)
true = kernel_test.eval_func(t)

alphas = []

res = np.zeros((len(t), len(runs)))
for i, run in enumerate(runs):
    res[:, i] = pred[run]["kernel"].eval_func(t)

mean = np.mean(res, axis=1)
std = np.std(res, axis=1)

#plt.fill_between(t, mean - std, mean + std, label="Std")
#plt.plot(t, res[:, 0], lw=1, color="k", ls=":", label="Sample")
for i in range(len(runs)):
    plt.plot(t, res[:, i], color="tab:blue", alpha=0.3)
#plt.plot(t, mean, color="red", label="Mean")
plt.plot(t, true, color="tab:red", ls="--", label="True")
plt.legend()
plt.xscale("log")
plt.ylabel(r"$k(t)$")
plt.xlabel(r"$t$")
plt.savefig(config['outputfolder']+"Kernels.pdf", bbox_inches="tight")
plt.show()


"""
for run in runs:

    print()
    print("#"*80)
    print(f"Run number {run}")
    print()

    data = pred[run]["kernel"].eval_func(t)

    def objective(alpha, data, tol):
        TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha)/(x**-alpha + 1/tau_sig)
        RA = RationalApproximation(alpha=alpha, TargetFunction=TargetFunction, tol=tol)
        parameters = list(RA.c) + list(RA.d)
        if infmode==True: parameters.append(RA.c_inf)
        kernel_test = SumOfExponentialsKernel(parameters=parameters)
        test = kernel_test.eval_func(t)
        return np.linalg.norm(data - test, ord=np.inf)

    bound = Bounds(0, 1)
    opt = minimize(objective, 0.5, args=(data, tol), bounds=bound)
    #if opt.success:
    alphas.append(opt.x)
    print(opt)

alphas = np.array(alphas)
plt.hist(alphas)
plt.axvline(0.5, color="k", ls="-", label="True value")
plt.axvline(alphas.mean(), color="r", ls="--", label="Sample mean")
plt.legend()
plt.title(r"Predicted $\alpha$ with 5% noise")
plt.savefig(config['outputfolder']+"AlphaPredict.pdf", bbox_inches="tight")
plt.show()
"""
