import os
import glob
import time

import torch
import numpy as np

import tikzplotlib
import matplotlib
import matplotlib.pyplot as plt

from src.Viscoelasticity import ViscoelasticityProblem
from src.Kernels import SumOfExponentialsKernel
from src.InverseProblem import InverseProblem
from src.Observers import TipDisplacementObserver
from src.Objectives import MSE
from src.Regularization import myRegularizationTerm as reg
from src.RationalApproximation import RationalApproximation_AAA as RationalApproximation
from src.data_manager import save_data, save_data_modes, load_data

from fenics import *
from fenics_adjoint import *


"""
==================================================================================================================
Plotting Defaults
==================================================================================================================
"""

# select plot stylesheet
plt.style.use("bmh")

font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size' : 12
    }
matplotlib.rc('font', **font)

figure_settings = {'figsize' : (10,6)}

plot_settings = {
    'markersize' : 2
    }

legend_settings = {}

# full width image in paper
tikz_settings = {
    'axis_width' : '160mm',
    'standalone' : True
    }


"""
==================================================================================================================
Problem Configuration
==================================================================================================================
"""

inputfolder  = "./workfolder/IC/"
outputfolder = "./workfolder/IC/"

os.makedirs(inputfolder, exist_ok=True)
os.makedirs(outputfolder, exist_ok=True)

### Beam
mesh = BoxMesh(Point(0., 0., 0.), Point(1., 0.1, 0.04), 60, 10, 5)

### Sub domain for clamp at left end
def DirichletBoundary(x, on_boundary):
    return near(x[0], 0.) and on_boundary

### Sub domain for excitation at right end
def NeumannBoundary(x, on_boundary):
    return near(x[0], 1.) and on_boundary

cutoff_time = 1.
magnitude   = 1.
tmax        = 4/5
tzero       = 1.
load_Bending = Expression(("0", "t <= tm ? p0*t/tm : (t <= tz ? p0*(1 - (t-tm)/(tz-tm)) : 0)", "0"), t=0, tm=tmax, tz=tzero, p0=magnitude, degree=0) ### Bending

config = {
    'verbose'           :   True,
    'inputfolder'       :   inputfolder,
    'outputfolder'      :   outputfolder,
    'export_vtk'        :   False,

    'FinalTime'         :   50,
    'nTimeSteps'        :   10, ### Steps per time unit!!!

    'mesh'              :   mesh,
    'DirichletBoundary' :   DirichletBoundary,
    'NeumannBoundary'   :   NeumannBoundary,
    'loading'           :   load_Bending, ###  load_Bending, [load_Bending, load_Extension]

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
    'observer'          :   TipDisplacementObserver,
}