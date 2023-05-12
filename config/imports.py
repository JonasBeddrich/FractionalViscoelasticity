import os
import sys
import importlib
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

### import correct config file
if len(sys.argv) == 1:
    sys.argv.append("config")
config_name = "config." + sys.argv[1]

config = importlib.import_module(config_name)
globals().update({k: getattr(config, k)
                  for k in [x for x in config.__dict__ if not x.startswith("_")]})