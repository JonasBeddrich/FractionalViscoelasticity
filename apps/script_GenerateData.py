


from matplotlib.pyplot import figure
from config import *

fg_export = True  ### write results on the disk (True) or only solve (False)
config['export_vtk'] = False

"""
==================================================================================================================
Kernel and its rational approximation
==================================================================================================================
"""

infmode = config.get('infmode', False)
zener_kernel = config.get('zener_kernel', False)

if config['two_kernels']:
    if zener_kernel:
        tau_eps = .2
        tau_sig = .1
        
        alpha1 = 0.2
        TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha1)/(x**-alpha1 + 1/tau_sig)
        RA = RationalApproximation(alpha=alpha1, TargetFunction=TargetFunction)
        parameters1 = list(RA.c) + list(RA.d)
        if infmode==True: parameters1.append(RA.c_inf)
        kernel1  = SumOfExponentialsKernel(parameters=parameters1)

        alpha2 = 0.5
        TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha2)/(x**-alpha2 + 1/tau_sig)
        RA = RationalApproximation(alpha=alpha2, TargetFunction=TargetFunction)
        parameters2 = list(RA.c) + list(RA.d)
        if infmode==True: parameters2.append(RA.c_inf)
        kernel2  = SumOfExponentialsKernel(parameters=parameters2)

    else:
        alpha1 = 0.9
        RA = RationalApproximation(alpha=alpha1)
        parameters1 = list(RA.c) + list(RA.d)
        if infmode==True: parameters1.append(RA.c_inf)
        kernel1 = SumOfExponentialsKernel(parameters=parameters1)
    
        alpha2 = 0.7
        RA = RationalApproximation(alpha=alpha2)
        parameters2 = list(RA.c) + list(RA.d)
        if infmode==True: parameters2.append(RA.c_inf)
        kernel2 = SumOfExponentialsKernel(parameters=parameters2)

    kernels    = [kernel1, kernel2]
    parameters = [parameters1, parameters2]

else:
    if zener_kernel:
        alpha = 0.5
        tau_eps = .2
        tau_sig = .1
        TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha)/(x**-alpha + 1/tau_sig)
        RA = RationalApproximation(alpha=alpha, TargetFunction=TargetFunction)
        parameters = list(RA.c) + list(RA.d)
        if infmode==True: parameters.append(RA.c_inf)
        kernel  = SumOfExponentialsKernel(parameters=parameters)
        kernels = [kernel]
    else:
        alpha = 0.7
        RA = RationalApproximation(alpha=alpha, tol=1.e-4)
        parameters = list(RA.c) + list(RA.d)
        if infmode==True: parameters.append(RA.c_inf)
        kernel = SumOfExponentialsKernel(parameters=parameters)
        kernels = [kernel]





"""
==================================================================================================================
Forward problem for generating data
==================================================================================================================
"""

print()
print()
print("================================")
print("       FORWARD RUN")
print("================================")

Model = ViscoelasticityProblem(**config, kernels=kernels)

loading = config.get("loading", None)
if isinstance(loading, list): ### multiple loadings case
    def Forward():
        obs = torch.tensor([])
        for loading_instance in loading:
            Model.forward_solve(loading=loading_instance)
            obs = torch.cat([obs, Model.observations], dim=-1)
        return obs.numpy()
else:
    def Forward():
        Model.forward_solve()
        obs = Model.observations
        return obs.numpy()

data = Forward()

if fg_export: ### write data to file
    # data = model.observations.numpy()
    np.savetxt(config['outputfolder']+"data_tip_displacement.csv", data)
    save_data(config['outputfolder']+"target_model", Model, other=[parameters])
    save_data_modes(config['outputfolder']+"modes", Model)

    # np.savetxt(config['outputfolder']+"tip_displacement_init.csv", data)
    # save_data(config['outputfolder']+"initial_model", Model, other=[parameters])


    # np.savetxt(config['outputfolder']+"tip_displacement_pred.csv", data)
    # save_data(config['outputfolder']+"inferred_model", Model, other=[parameters])


"""
==================================================================================================================
Display
==================================================================================================================
"""

with torch.no_grad():
    plt.subplot(1,2,1)
    plt.title('Tip displacement')
    plt.plot(Model.time_steps, data)

    if not Model.fg_inverse:
        plt.subplot(1,2,2)
        plt.title('Energies')
        plt.plot(Model.time_steps, Model.Energy_elastic, "o-", color='blue', label="Elastic energy")
        plt.plot(Model.time_steps, Model.Energy_kinetic, "o-", color='orange', label="Kinetic energy")
        # plt.plot(Model.time_steps, Model.Energy_elastic+Model.Energy_kinetic, "o-", color='red', label="Total energy")
        plt.grid(True, which='both')
        plt.legend()

    plt.show()

    # model.kernel.plot()


