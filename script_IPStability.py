from config.imports import * 

fg_export = True    ### write results on the disk (True) or only solve (False)
config_initial = config.copy()

noise_level = config['noise_level'] ### [%]
exclude_loading = config['exclude_loading']
infmode = config.get('infmode', False)

config['initial_guess'] = None
config['two_kernels'] = False

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
Configure kernel
==================================================================================================================
"""

if len(sys.argv) >= 4:
    alpha = float(sys.argv[3])
else:
    print("Pass alpha for initial guess as parameter. Aborting!")
    sys.exit()

# generate kernel for given alpha with correct number of modes
nModes = config.get("nModes", 10)
tau_eps = config.get("tau_eps", 0.2)
tau_sig = config.get("tau_sig", 0.1)

TargetFunction = lambda x: (tau_eps/tau_sig - 1) * x**(1-alpha)/(x**-alpha + 1/tau_sig)

tol = 1e-3
while True:
    RA = RationalApproximation(alpha=alpha, TargetFunction=TargetFunction, tol=tol)
    if RA.nModes >= nModes:
        break
    tol *= 0.9

if RA.nModes != nModes:
    print("Could not get RA with correct number of modes. Aborting!")
    sys.exit()

parameters = list(RA.c) + list(RA.d)
if infmode==True: parameters.append(RA.c_inf)
config["kernels"] = [SumOfExponentialsKernel(parameters=parameters)]
config_initial["kernels"] = [SumOfExponentialsKernel(parameters=parameters)]
print("nModes kernel: ", RA.nModes)

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

    model = ViscoelasticityProblem(**config_initial)
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
    #save_data(config['outputfolder']+f"model_initial_{timestamp}_{alpha}", model, other=[parameters])
    np.savetxt(config['outputfolder']+f"tip_displacement_initial_{timestamp}_{alpha}.csv", pred)

print()
print()
print("================================")
print("       INVERSE PROBLEM")
print("================================")

model = ViscoelasticityProblem(**config)

if exclude_loading:
    objective = MSE(data=data, start=steps_per_unit)
else:
    objective = MSE(data=data)
IP        = InverseProblem(**config, plots=True)

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
    save_data(config['outputfolder']+f"model_predict_{timestamp}_{alpha}", model, other=[theta_opt, IP.convergence_history])
    np.savetxt(config['outputfolder']+f"tip_displacement_predict_{timestamp}_{alpha}.csv", pred)
