from config.imports import * 

fg_export = True  ### write results on the disk (True) or only solve (False)

"""
==================================================================================================================
Kernel and its rational approximation
==================================================================================================================
"""

print()
print()
print("================================")
print("    RATIONAL APPX OF KERNEL")
print("================================")

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
        print("nModes kernel 1: ", RA.nModes)

        alpha2 = 0.5
        TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha2)/(x**-alpha2 + 1/tau_sig)
        RA = RationalApproximation(alpha=alpha2, TargetFunction=TargetFunction)
        parameters2 = list(RA.c) + list(RA.d)
        if infmode==True: parameters2.append(RA.c_inf)
        kernel2  = SumOfExponentialsKernel(parameters=parameters2)
        print("nModes kernel 2: ", RA.nModes)

    else:
        alpha1 = 0.9
        RA = RationalApproximation(alpha=alpha1)
        parameters1 = list(RA.c) + list(RA.d)
        if infmode==True: parameters1.append(RA.c_inf)
        kernel1 = SumOfExponentialsKernel(parameters=parameters1)
        print("nModes kernel 1: ", RA.nModes)
    
        alpha2 = 0.7
        RA = RationalApproximation(alpha=alpha2)
        parameters2 = list(RA.c) + list(RA.d)
        if infmode==True: parameters2.append(RA.c_inf)
        kernel2 = SumOfExponentialsKernel(parameters=parameters2)
        print("nModes kernel 2: ", RA.nModes)

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
        RA = RationalApproximation(alpha=alpha)
        parameters = list(RA.c) + list(RA.d)
        if infmode==True: parameters.append(RA.c_inf)
        kernel = SumOfExponentialsKernel(parameters=parameters)
        kernels = [kernel]
    print("nModes kernel: ", RA.nModes)





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
observer = config.get("observer", None)
if isinstance(loading, list): ### multiple loadings case
    def Forward():
        obs = torch.tensor([])
        for i, loading_instance in enumerate(loading):
            if isinstance(observer, list): ### multiple observers
                Model.set_observer(observer[i])
            Model.forward_solve(loading=loading_instance)
            obs = torch.cat([obs, Model.observations], dim=-1)
        return obs.numpy()
else:
    def Forward():
        Model.forward_solve(loading=loading)
        obs = Model.observations
        return obs.numpy()

data = Forward()

if fg_export: ### write data to file
    
    if len(sys.argv) >= 3:
        timestamp = sys.argv[2]
    else:
        timestamp = time.strftime("%Y%m%d-%H%M")

    np.savetxt(config['outputfolder']+"tip_displacement_target_"+timestamp+".csv", data)
    save_data(config['outputfolder']+"model_target_"+timestamp+"", Model, other=[parameters])
    save_data_modes(config['outputfolder']+"modes_target_"+timestamp+"", Model)

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
        plt.plot(Model.time_steps, Model.Energy_elastic, 
                 "o-", color='blue', label="Elastic energy")
        plt.plot(Model.time_steps, Model.Energy_kinetic, 
                 "o-", color='orange', label="Kinetic energy")
        plt.plot(Model.time_steps, Model.Energy_viscous, 
                 "o-", color='green', label="Viscous energy")
        plt.plot(Model.time_steps, Model.Energy_elastic + Model.Energy_kinetic + Model.Energy_viscous, 
                 "o-", color='red', label="Total energy")

        # plt.plot(Model.time_steps, Model.Energy_elastic+Model.Energy_kinetic, "o-", color='red', label="Total energy")
        plt.grid(True, which='both')
        plt.legend()

    plt.show()

    # model.kernel.plot()


