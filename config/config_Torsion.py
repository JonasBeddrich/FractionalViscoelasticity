from config.imports import * 

"""
==================================================================================================================
Problem Configuration
==================================================================================================================
"""

outputfolder = "./workfolder/Torsion/"
inputfolder  = "./workfolder/Torsion/"

os.makedirs(inputfolder, exist_ok=True)
os.makedirs(outputfolder, exist_ok=True)

### Beam
mesh = BoxMesh(Point(0., -0.1, -0.1), Point(1.0, 0.1, 0.1), 25, 10, 10)

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
    load_Torsion = Expression(("0", "t <= tm ? p0*-1*x[2]*t/tm : (t <= tz ? p0*-1*x[2]*(1 - (t-tm)/(tz-tm)) : 0)", "t <= tm ? p0*x[1]*t/tm : (t <= tz ? p0*x[1]*(1 - (t-tm)/(tz-tm)) : 0)"), t=0, tm=tmax, tz=tzero, p0=magnitude, degree=0) ### Bending
else:
    load_Torsion = Expression(("0", "t <= tc ? p0*-1*x[2]*t/tc : 0", "t <= tc ? p0*x[1]*t/tc : 0"), t=0, tc=cutoff_time, p0=magnitude, degree=0)

config = {
    'verbose'           :   True,
    'inputfolder'       :   inputfolder,
    'outputfolder'      :   outputfolder,
    'export_vtk'        :   True,

    'FinalTime'         :   5,
    'nTimeSteps'        :   100,

    'mesh'              :   mesh,
    'DirichletBoundary' :   DirichletBoundary,
    'NeumannBoundary'   :   NeumannBoundary,
    'loading'           :   load_Torsion,

    'infmode'           :   True,
    'zener_kernel'      :   True,

    ### Material parameters
    'Young'             :   1.e3,
    'Poisson'           :   0.3,
    'density'           :   1.,

    ### Viscous term
    'viscosity'         :   True,
    'two_kernels'       :   False,

    ### Measurements
    'observer'          :   TipNormDisplacementObserver,
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