
from fenics import *
from fenics_adjoint import *

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.Viscoelasticity import ViscoelasticityProblem
from src.Kernels import SumOfExponentialsKernel
from src.InverseProblem import InverseProblem
from src.Observers import TipDisplacementObserver
from src.Objectives import MSE
from src.Regularization import myRegularizationTerm as reg
from src.RationalApproximation import RationalApproximation_AAA as RationalApproximation
from src.data_manager import save_data, save_data_modes, load_data


"""
==================================================================================================================
Problem Configuration
==================================================================================================================
"""

inputfolder  = "./workfolder/"
outputfolder = "./workfolder/"

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

magnitude      = 1.e2
if continuous_loading:
    load_Bending = Expression(("0", "t <= tm ? p0*t/tm : (t <= tz ? p0*(1 - (t-tm)/(tz-tm)) : 0)", "0"), t=0, tm=tmax, tz=tzero, p0=magnitude, degree=0) ### Extension
else:
    load_Extension = Expression(("t <= tc ? p0*t/tc : 0", "0", "0"), t=0, tc=cutoff_time, p0=magnitude, degree=0) ### Extension


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
    'loading'           :   [load_Bending],#, load_Extension], ###  load_Bending, [load_Bending, load_Extension]

    'infmode'           :   True,

    ### Material parameters
    'Young'             :   1.e3,
    'Poisson'           :   0.3,
    'density'           :   1.,

    ### Viscous term
    'viscosity'         :   True,
    'two_kernels'       :   False,

    ### Measurements
    'observer'          :   TipDisplacementObserver,
    'noise_level'       :   2, ### [%]

    ### Optimization
    "init_fractional"   :   {"alpha" : 0.7, "tol" : 1.e-4 },
    'optimizer'         :   torch.optim.LBFGS, ### E.g., torch.optim.SGD, torch.optim.LBFGS (recommended), ...
    'max_iter'          :   100,
    'tol'               :   1.e-4,
    'regularization'    :   None,  ### your regularization function, e.g., "reg", or None/False for no regularization
    'initial_guess'     :   None,  ### initial guess for parameters calibration: (weights, exponents)
    'line_search_fn'    :   'strong_wolfe', ### None, 'strong_wolfe',
}