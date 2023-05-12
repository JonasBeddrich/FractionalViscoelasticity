from config.imports import * 
from config.plot_defaults import *

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
    filename = min(glob.iglob(config['inputfolder']+"tip_displacement_noisy_*.csv"), key=os.path.getctime)
    timestamp = filename[-17:-4]
tip_meas = np.loadtxt(filename)

tip_init, EnergyElastic_init, EnergyKinetic_init, EnergyViscous_init, theta_init                      = load_data(config['inputfolder']+"model_initial_"+timestamp)
tip_pred, EnergyElastic_pred, EnergyKinetic_pred, EnergyViscous_pred, theta_pred, convergence_history = load_data(config['inputfolder']+"model_predict_"+timestamp)
tip_true, EnergyElastic_true, EnergyKinetic_true, EnergyViscous_true, theta_true                      = load_data(config['inputfolder']+"model_target_"+timestamp)
EnergyTotal_pred = EnergyElastic_pred + EnergyKinetic_pred
EnergyTotal_true = EnergyElastic_true + EnergyKinetic_true

tip_true = np.loadtxt(config['inputfolder']+"tip_displacement_target_"+timestamp+".csv")
tip_init = np.loadtxt(config['inputfolder']+"tip_displacement_initial_"+timestamp+".csv")
tip_pred = np.loadtxt(config['inputfolder']+"tip_displacement_predict_"+timestamp+".csv")

time_steps = np.linspace(0, config['FinalTime'], config['nTimeSteps']+1)[1:]
time_steps_meas = time_steps[:tip_meas.shape[0]]
if config['exclude_loading']:
    time_steps_meas = time_steps_meas + 1

print(tip_true)

if tip_true.ndim == 2:
    tip_true_tr, tip_true_dev = tip_true[...,0], tip_true[...,1]
    tip_meas_tr, tip_meas_dev = tip_meas[...,0], tip_meas[...,1]
    tip_init_tr, tip_init_dev = tip_init[...,0], tip_init[...,1]
    tip_pred_tr, tip_pred_dev = tip_pred[...,0], tip_pred[...,1]
else:
    tip_true_tr, tip_true_dev = tip_true, tip_true
    tip_meas_tr, tip_meas_dev = tip_meas, tip_meas
    tip_init_tr, tip_init_dev = tip_init, tip_init
    tip_pred_tr, tip_pred_dev = tip_pred, tip_pred



"""
==================================================================================================================
Construct kernels
==================================================================================================================
"""

alpha1, alpha2 = 0.9, 0.7
alpha_init = 0.5

RA = RationalApproximation(alpha=alpha_init, tol=1.e-4)
parameters0 = list(RA.c) + list(RA.d)
kernel_init = SumOfExponentialsKernel(parameters=parameters0)

#tmp = np.array([i.detach().numpy() for i in theta_pred[0]])
#n = len(tmp)
#parameters1 = tmp[:n].tolist()
#parameters2 = tmp[n:].tolist()
#parameters1, parameters2 = np.array(theta_pred).tolist()
parameters1, parameters2 = theta_pred
kernel_pred_tr  = SumOfExponentialsKernel(parameters=parameters1)
kernel_pred_dev = SumOfExponentialsKernel(parameters=parameters2)

parameters1, parameters2 = theta_true
kernel_true_tr  = SumOfExponentialsKernel(parameters=parameters1)
kernel_true_dev = SumOfExponentialsKernel(parameters=parameters2)





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



    fig = plt.figure('Tip displacement (tr)', **figure_settings)
    # plt.title('Tip displacement')
    plt.plot(time_steps, tip_init_tr, "-",  color="gray", label="initial", **plot_settings)
    plt.plot(time_steps, tip_pred_tr, "r-",  label="predict", **plot_settings)
    plt.plot(time_steps, tip_true_tr, "b--", label="truth", **plot_settings)
    plt.plot(time_steps_meas, tip_meas_tr, "ko:", label="data", **plot_settings)
    plt.legend()
    plt.ylabel(r"Tip displacement")
    plt.xlabel(r"$t$")
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    
    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(tikz_folder+"plt_two_kernels_tip_displacement_bending_"+timestamp+".tex", **tikz_settings)


    fig = plt.figure('Tip displacement (dev)', **figure_settings)
    # plt.title('Tip displacement')
    plt.plot(time_steps, tip_init_dev, "-",  color="gray", label="initial", **plot_settings)
    plt.plot(time_steps, tip_pred_dev, "r-",  label="predict", **plot_settings)
    plt.plot(time_steps, tip_true_dev, "b--", label="truth", **plot_settings)
    plt.plot(time_steps_meas, tip_meas_dev, "ko:", label="data", **plot_settings)
    plt.legend()
    plt.ylabel(r"Tip displacement")
    plt.xlabel(r"$t$")

    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(tikz_folder+"plt_two_kernels_tip_displacement_extension_"+timestamp+".tex", **tikz_settings)


    """
    ==================================================================================================================
    Figure 2: Energies
    ==================================================================================================================
    """
    # plt.figure('Energies', **figure_settings)
    # # plt.title('Energies')

    # plt.plot(time_steps, EnergyElastic_pred, "-", color='red', label="Elastic energy (predict)", **plot_settings)
    # plt.plot(time_steps, EnergyKinetic_pred, "-", color='orange', label="Kinetic energy (predict)", **plot_settings)
    # # plt.plot(time_steps, EnergyTotal_pred, "-", color='brown', label="Total energy (predict)")

    # plt.plot(time_steps, EnergyElastic_true, "--", color='blue', label="Elastic energy (truth)", **plot_settings)
    # plt.plot(time_steps, EnergyKinetic_true, "--", color='cyan', label="Kinetic energy (truth)", **plot_settings)
    # # plt.plot(time_steps, EnergyTotal_true, "--", color='magenta', label="Total energy (truth)", **plot_settings)

    # plt.grid(True, which='both')
    # plt.ylabel(r"Energy")
    # plt.xlabel(r"$t$")
    # plt.legend(**legend_settings)

    # tikzplotlib.save(tikz_folder+"plt_energies.tex")



    """
    ==================================================================================================================
    Figure 3: Kernels
    ==================================================================================================================
    """
    fig = plt.figure('Kernels (tr)', **figure_settings)
    # plt.title('Kernels')
    t = np.geomspace(0.04, 2, 100)
    plt.plot(t, kernel_init.eval_func(t), "-", color="gray", label="initial guess", **plot_settings)
    plt.plot(t, kernel_pred_tr.eval_func(t), "r-", label="predict", **plot_settings)
    plt.plot(t, kernel_true_tr.eval_func(t), "b-", label="truth", **plot_settings)
    # plt.plot(t, kernel_frac_init(t), "o", color="gray", label=r"fractional $\alpha=0.5$", **plot_settings)
    # plt.plot(t, t**(alpha1-1)/gamma(alpha1), "bo", label=r"fractional $\alpha=0.9$", **plot_settings)
    plt.xscale('log')
    plt.ylabel(r"$k(t)$")
    plt.xlabel(r"$t$")
    plt.legend(**legend_settings)

    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(tikz_folder+"plt_two_kernels_compare_kernels_tr_"+timestamp+".tex", **tikz_settings)



    fig = plt.figure('Kernels (dev)', **figure_settings)
    # plt.title('Kernels')
    t = np.geomspace(0.04, 2, 100)
    plt.plot(t, kernel_init.eval_func(t), "-", color="gray", label="initial guess", **plot_settings)
    plt.plot(t, kernel_pred_dev.eval_func(t), "r-", label="predict", **plot_settings)
    plt.plot(t, kernel_true_dev.eval_func(t), "b-", label="truth", **plot_settings)
    # plt.plot(t, kernel_frac_init(t), "o", color="gray", label=r"fractional $\alpha=0.5$", **plot_settings)
    # plt.plot(t, kernel_frac(t), "bo", label=r"fractional $\alpha=0.7$", **plot_settings)
    # plt.plot(t, t**(alpha2-1)/gamma(alpha2), "bo", label=r"fractional $\alpha=0.7$", **plot_settings)
    plt.xscale('log')
    plt.ylabel(r"$k(t)$")
    plt.xlabel(r"$t$")
    plt.legend(**legend_settings)

    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(tikz_folder+"plt_two_kernels_compare_kernels_dev_"+timestamp+".tex", **tikz_settings)

    """
    ==================================================================================================================
    Figure 4: Convergence
    ==================================================================================================================
    """

    fig = plt.figure('Convergence', **figure_settings)

    plt.plot(convergence_history['loss'], label="loss")
    plt.plot(convergence_history['grad'], label="gradient")
    # plt.plot(t, kernel_frac_init(t), "o", color="gray", label=r"fractional $\alpha=0.5$", **plot_settings)
    # plt.plot(t, kernel_frac(t), "bo", label=r"fractional $\alpha=0.7$", **plot_settings)
    # plt.plot(t, t**(alpha2-1)/gamma(alpha2), "bo", label=r"fractional $\alpha=0.7$", **plot_settings)

    #plt.ylabel("Loss")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.legend(**legend_settings)

    tikzplotlib.clean_figure(fig)
    tikzplotlib.save(tikz_folder+"plt_two_kernels_convergence_"+timestamp+".tex", **tikz_settings)


    """
    ==================================================================================================================
    """

    plt.show()




