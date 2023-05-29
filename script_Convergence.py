from config.imports import *
from datetime import datetime

# get input values
timestring = sys.argv[2]
alpha = float(sys.argv[3])
index = int(sys.argv[4])
maxindex = int(sys.argv[5])

# infmode boolean from config
infmode = config.get('infmode', False)

# compute sum of exponentials approximation for fixed alpha
RA = RationalApproximation(alpha=alpha)
parameters = list(RA.c) + list(RA.d)
if infmode == True:
    parameters.append(RA.c_inf)
kernel = SumOfExponentialsKernel(parameters=parameters)
kernels = [kernel]

path = config['outputfolder']+f"/alpha{alpha}/"
if not os.path.exists(path):
    os.makedirs(path)


n_steps_list = 2**np.arange(0, maxindex)*1e2
n_steps_list = np.append(n_steps_list, n_steps_list[-1]*10)
n_steps = n_steps_list[index]
config['nTimeSteps'] = int(n_steps*5)

print(f"START: dt={1/n_steps} started")

Model = ViscoelasticityProblem(**config, kernels=kernels)

# do not compute energy and norm of modes for performance reasons
Model.flags['inverse'] = True

Model.forward_solve(loading=config['loading'])
obs = Model.observations
data = obs.numpy()

np.savetxt(path+f"tipdisplacement_{timestring}_{alpha}_{n_steps}.txt", data)

print(f"END: dt={1/n_steps} finished")
