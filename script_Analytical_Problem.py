import os
import sys
import importlib
import glob
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.RationalApproximation import RationalApproximation_AAA
from scipy.special import gamma 
import pandas as pd 

######################################################################
#                           TESTING
######################################################################

@np.vectorize
def DistributedKernel(alpha): 
    return gamma(6-alpha) / 120

@np.vectorize
def MultiTermKernel(x, alphas, weights):
    tmp = 0
    for (alpha, weight) in zip(alphas,weights): 
        tmp += weight * x ** (1-alpha)
    return tmp 

if __name__ == "__main__":

    n = 10 




    # T  = 1
    # dt = 1.e-6
    # Zmin, Zmax = 1/T,1/dt
    # tol = 1.e-12
    # MaxDegree = 30
    # nSupportPoints=100

    # factor = 2

    # alpha = 0.6
    # beta  = 0.3 

    # TargetFunction = lambda x: (x**(1-alpha) + x**(1-beta)) 
    # TargetFunction_scaled = lambda x: factor * (x**(1-alpha) + x**(1-beta)) 
    
    # Function = lambda x: 1/gamma(1-alpha) * x**-alpha + 1/gamma(1-beta) * x**-beta 

    # # TargetFunction = lambda x: x**(alpha)
    # # Function = lambda x: x**(alpha-1)/gamma(alpha)

    # RA = RationalApproximation_AAA(alpha=alpha,
    #                                 tol=tol, 
    #                                 MaxDegree=MaxDegree, 
    #                                 nSupportPoints=nSupportPoints,
    #                                 Zmin= Zmin, 
    #                                 Zmax= Zmax,
    #                                 verbose=False, 
    #                                 TargetFunction=TargetFunction)
    

    # RA_scaled = RationalApproximation_AAA(alpha=alpha,
    #                                 tol=tol, 
    #                                 MaxDegree=MaxDegree, 
    #                                 nSupportPoints=nSupportPoints,
    #                                 Zmin= Zmin, 
    #                                 Zmax= Zmax,
    #                                 verbose=False, 
    #                                 TargetFunction=TargetFunction_scaled)

    # x = np.geomspace(0.0001,1,1000)
    # # plt.plot(x, RA_alpha(x))
    # plt.plot(x, RA.appx_ker(x))
    # plt.plot(x, 1/factor * RA_scaled.appx_ker(x), "r:", linewidth=6)
    
    # # plt.plot(x, Function(x))
    # plt.xscale("log")
    # plt.show()


    # for alpha in np.linspace(0,1,6):
    #     for beta in np.linspace(0,1,6):  
    #         success = False 
    #         alpha = np.round(alpha,5)
    #         beta = np.round(beta,5)
    #         for prefactor in [1,0.5,0.25,0.125]: 
    #             TargetFunction = lambda x: prefactor * (x ** alpha + x ** beta)
    #             try: 
    #                 RA_alpha = RationalApproximation_AAA(alpha=alpha,
    #                                             tol=tol, 
    #                                             MaxDegree=MaxDegree, 
    #                                             nSupportPoints=nSupportPoints,
    #                                             Zmin= Zmin, 
    #                                             Zmax= Zmax,
    #                                             verbose=False, 
    #                                             TargetFunction=TargetFunction)
    #                 success = True
    #                 break; 
    #             except: 
    #                 i = 1 # print()
    #         # if not success: 
    #         print(alpha, beta, success, prefactor)