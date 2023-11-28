from config.imports import * 

fg_export = True    ### write results on the disk (True) or only solve (False)
config_initial = config.copy()

noise_level = config['noise_level'] ### [%]
exclude_loading = config['exclude_loading']


"""
==================================================================================================================
Initial guess
==================================================================================================================
"""

# alpha = 0.5
# RA = RationalApproximation(alpha=alpha, tol=1.e-4)
# config['nModes'] = RA.nModes
# parameters = list(RA.c) + list(RA.d)
# if config['infmode']:
#   parameters.append(RA.c_inf)
# config['initial_guess'] = [RA.c, RA.d]
config['initial_guess'] = None

"""
==================================================================================================================
Data to fit
==================================================================================================================
"""

# Load file with given timestamp or newest file with matching name
if len(sys.argv) >= 3:
    timestamp = sys.argv[2]
    filename = config['inputfolder']+"tip_displacement_target_"+timestamp+".csv"
else:
    filename = max(glob.iglob(config['inputfolder']+"tip_displacement_target_*.csv"), key=os.path.getctime)
    print(filename)
    timestamp = filename[-17:-4]

data_true = np.loadtxt(filename)
data = data_true.copy()

if exclude_loading:
    
    T, nsteps = config['FinalTime'], config['nTimeSteps']
    steps_per_unit = nsteps // T

    if data.ndim == 2:
        data = data[steps_per_unit:3*steps_per_unit, :]
    else:
        data = data[steps_per_unit:3*steps_per_unit]

    config['nTimeSteps'] = 3*steps_per_unit
    config['FinalTime']  = 3

else:
    ### Optimize on a shorter interval
    T, nsteps = config['FinalTime'], config['nTimeSteps']
    steps_per_unit = nsteps // T
    if data.ndim == 2:
        data = data[:3*steps_per_unit, :]
    else:
        data = data[:3*steps_per_unit]
    config['nTimeSteps'] = data.shape[0]
    config['FinalTime']  = data.shape[0] * (T / nsteps)

### Noisy data
scale = (noise_level/100) * np.abs(data) #.max(axis=0, keepdims=True)
noise = np.random.normal(loc=0, scale=scale, size=data.shape) ### additive noise
data  = data + noise
np.savetxt(config['outputfolder']+"tip_displacement_noisy_"+timestamp+".csv", data)


### Compare data
plt.figure()
plt.plot(data_true, "r-", label="true data")
if exclude_loading:
    plt.plot(range(steps_per_unit, 3*steps_per_unit), data, "bo--", label="measurements")
else:
    plt.plot(data, "bo--", label="measurements")
plt.legend()
plt.show()


"""
==================================================================================================================
Forward run of initial guess
==================================================================================================================
"""

if fg_export:
    print()
    print()
    print("================================")
    print("       RUN INITIAL GUESS")
    print("================================")

    if config_initial["two_kernels"]:
        kernels = [SumOfExponentialsKernel(**config), SumOfExponentialsKernel(**config)] 
        parameters = [kernels[0].default_parameters(), kernels[1].default_parameters()]
    else:
        kernels = [SumOfExponentialsKernel(**config)]
        parameters = kernels[0].default_parameters()

    model = ViscoelasticityProblem(**config_initial, kernels=kernels)
    model.set_time_stepper(nTimeSteps=nsteps, FinalTime=T)
    model.flags["inverse"] = False

    loading = config.get("loading", None)
    observer = config.get("observer", None)
    if isinstance(loading, list): ### multiple loadings case
        obs = torch.tensor([])
        for i, loading_instance in enumerate(loading):
            if isinstance(observer, list): ### multiple observers
                model.set_observer(observer[i])
            model.forward_solve(loading=loading_instance)
            obs = torch.cat([obs, model.observations], dim=-1)
        pred = obs.numpy()
    else:
        model.forward_solve(loading=loading)
        obs = model.observations
        pred =  obs.numpy()
    
    ### write data to file
    save_data(config['outputfolder']+"model_initial_"+timestamp, model, other=[parameters])
    np.savetxt(config['outputfolder']+"tip_displacement_initial_"+timestamp+".csv", pred)



"""
==================================================================================================================
Inverse problem
==================================================================================================================
"""

print()
print()
print("================================")
print("       INVERSE PROBLEM")
print("================================")

if config["two_kernels"]:
    kernels = [SumOfExponentialsKernel(**config), SumOfExponentialsKernel(**config)]
else:
    kernels = [SumOfExponentialsKernel(**config)]

model = ViscoelasticityProblem(**config, kernels=kernels)

if exclude_loading:
    objective = MSE(data=data, start=steps_per_unit)
else:
    objective = MSE(data=data)

IP = InverseProblem(**config, plots=True)

theta_opt = IP.calibrate(model, objective, **config)

print("Optimal parameters :", theta_opt)
print("Final loss         :", IP.loss)

"""
==================================================================================================================
Forward run of the inferred model
==================================================================================================================
"""

print()
print()
print("================================")
print("       RUN RESULTING MODEL")
print("================================")

### Recover the original time settings
model.set_time_stepper(nTimeSteps=nsteps, FinalTime=T)

model.flags["inverse"] = False

loading = config.get("loading", None)
observer = config.get("observer", None)
if isinstance(loading, list): ### multiple loadings case
    obs = torch.tensor([])
    for i, loading_instance in enumerate(loading):
        if isinstance(observer, list): ### multiple observers
            model.set_observer(observer[i])
        model.forward_solve(loading=loading_instance)
        obs = torch.cat([obs, model.observations], dim=-1)
    pred = obs.numpy()
else:
    model.initialize_state()
    model.forward_solve(loading=loading)
    obs = model.observations
    pred =  obs.numpy()

if fg_export: ### write data to file
    save_data(config['outputfolder']+"model_predict_"+timestamp, model, other=[theta_opt, IP.convergence_history])
    np.savetxt(config['outputfolder']+"tip_displacement_predict_"+timestamp+".csv", pred)

"""
==================================================================================================================
Display
==================================================================================================================
"""

with torch.no_grad():
    plt.subplot(1,2,1)
    plt.title('Tip displacement')
    plt.plot(model.time_steps, pred, "r-",  label="prediction")
    plt.plot(model.time_steps, data_true, "b--", label="truth")
    if exclude_loading:
        plt.plot(model.time_steps[steps_per_unit:3*steps_per_unit], data, "bo", label="data")
    else:
        plt.plot(model.time_steps[:data.shape[0]], data, "bo", label="data")
    plt.legend()

    if not model.fg_inverse:
        plt.subplot(1,2,2)
        plt.title('Energies')
        plt.plot(model.time_steps, model.Energy_elastic, "o-", color='blue', label="Elastic energy")
        plt.plot(model.time_steps, model.Energy_kinetic, "o-", color='orange', label="Kinetic energy")
        plt.plot(model.time_steps, model.Energy_viscous, "o-", color='green', label="Viscous energy")
        plt.plot(model.time_steps, model.Energy_elastic+model.Energy_kinetic+model.Energy_viscous, "o-", color='red', label="Total energy")
        plt.grid(True, which='both')
        plt.legend()

    plt.show()

model.kernel.plot()

for kernel in model.kernels: 
    print(kernel.weights)
    print(kernel.exponents)