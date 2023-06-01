from config.imports import * 

"""
==================================================================================================================
Problem Configuration
==================================================================================================================
"""

inputfolder  = "./workfolder/IPStability/"
outputfolder = "./workfolder/IPStability/"

os.makedirs(inputfolder, exist_ok=True)
os.makedirs(outputfolder, exist_ok=True)

### Beam
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 20, 4, 2)

### Sub domain for clamp at left end
def DirichletBoundary(x, on_boundary):
    return near(x[0], 0.) and on_boundary

### Sub domain for excitation at right end
def NeumannBoundary(x, on_boundary):
    return near(x[0], 1.) and on_boundary

### loading (depending on t)
continuous_loading = True

cutoff_time = 1.
magnitude   = 1.
tmax        = 4/5
tzero       = 1.
if continuous_loading:
    load_Bending = Expression(("0", "t <= tm ? p0*t/tm : (t <= tz ? p0*(1 - (t-tm)/(tz-tm)) : 0)", "0"), t=0, tm=tmax, tz=tzero, p0=magnitude, degree=0) ### Bending
else:
    load_Bending   = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_time, p0=magnitude, degree=0) ### Bending

magnitude = 1.e2
if continuous_loading:
    load_Extension = Expression(("t <= tm ? p0*t/tm : (t <= tz ? p0*(1 - (t-tm)/(tz-tm)) : 0)", "0", "0"), t=0, tm=tmax, tz=tzero, p0=magnitude, degree=0) ### Extension
else:
    load_Extension = Expression(("t <= tc ? p0*t/tc : 0", "0", "0"), t=0, tc=cutoff_time, p0=magnitude, degree=0) ### Extension

nModes = 10
alpha = 0.5
tau_eps = .2
tau_sig = .1
infmode = True

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
kernels = [SumOfExponentialsKernel(parameters=parameters)]

config = {
    'verbose'           :   True,
    'inputfolder'       :   inputfolder,
    'outputfolder'      :   outputfolder,
    'export_vtk'        :   False,

    'FinalTime'         :   5,
    'nTimeSteps'        :   100,

    'mesh'              :   mesh,
    'DirichletBoundary' :   DirichletBoundary,
    'NeumannBoundary'   :   NeumannBoundary,
    'loading'           :   load_Bending,

    'infmode'           :   True,
    'kernels'           :   kernels,
    'parameters'        :   parameters,
    'tau_eps'           :   tau_eps,
    'tau_sig'           :   tau_sig,
    'nModes'            :   nModes,

    ### Material parameters
    'Young'             :   1.e3,
    'Poisson'           :   0.3,
    'density'           :   1.,

    ### Viscous term
    'viscosity'         :   True,
    'two_kernels'       :   False,

    ### Measurements
    'observer'          :   TipYDisplacementObserver,
    'noise_level'       :   2, ### [%]

    ### Optimization
    "init_fractional"   :   {"alpha" : 0.7, "tol" : 1.e-4 },
    'optimizer'         :   torch.optim.LBFGS, ### E.g., torch.optim.SGD, torch.optim.LBFGS (recommended), ...
    'max_iter'          :   100,
    'tol'               :   1.e-4,
    'regularization'    :   None,  ### your regularization function, e.g., "reg", or None/False for no regularization
    'initial_guess'     :   None,  ### initial guess for parameters calibration: (weights, exponents)
    'line_search_fn'    :   'strong_wolfe', ### None, 'strong_wolfe',
    'exclude_loading'   :   True,
}