
from config import *

fg_export = True    ### write results on the disk (True) or only solve (False)

noise_level = config['noise_level'] ### [%]

exclude_loading = True


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

data_true = np.loadtxt(config['inputfolder']+"data_tip_displacement.csv")
data = data_true.copy()

if exclude_loading:
    # currently only supported for one kernel
    assert config['two_kernels'] == False

    T, nsteps = config['FinalTime'], config['nTimeSteps']
    
    steps_per_unit = nsteps // T
    data = data[steps_per_unit:3*steps_per_unit]

    config['nTimeSteps'] = 3*steps_per_unit
    config['FinalTime']  = 3

else:
    ### Optimize on a shorter interval
    if config['two_kernels']:
        data = data[:int(data.shape[0]//2), :]
    else:
        data = data[:int(data.shape[0]//2)]
    T, nsteps = config['FinalTime'], config['nTimeSteps']
    config['nTimeSteps'] = data.shape[0]
    config['FinalTime']  = data.shape[0] * (T / nsteps)

### Noisy data
scale = (noise_level/100) * np.abs(data) #.max(axis=0, keepdims=True)
noise = np.random.normal(loc=0, scale=scale, size=data.shape) ### additive noise
data  = data + noise
np.savetxt(config['outputfolder']+"data_tip_displacement_noisy.csv", data)


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
Inverse problem
==================================================================================================================
"""

print()
print()
print("================================")
print("       INVERSE PROBLEM")
print("================================")

if config["two_kernels"]:
    kernels = [SumOfExponentialsKernel(**config), SumOfExponentialsKernel(**config)] ### default kernels: alpha=0.5, 8 modes
else:
    kernels = [SumOfExponentialsKernel(**config)]

model = ViscoelasticityProblem(**config, kernels=kernels)

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
if isinstance(loading, list): ### multiple loadings case
    obs = torch.tensor([])
    for loading_instance in loading:
        model.forward_solve(loading=loading_instance)
        obs = torch.cat([obs, model.observations], dim=-1)
    pred = obs.numpy()
else:
    model.forward_solve()
    obs = model.observations
    pred =  obs.numpy()

if fg_export: ### write data to file
    save_data(config['outputfolder']+"inferred_model", model, other=[theta_opt, IP.convergence_history])
    np.savetxt(config['outputfolder']+"tip_displacement_pred.csv", pred)


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
        plt.plot(model.time_steps, model.Energy_elastic+model.Energy_kinetic, "o-", color='red', label="Total energy")
        plt.grid(True, which='both')
        plt.legend()

    plt.show()

    # model.kernel.plot()


