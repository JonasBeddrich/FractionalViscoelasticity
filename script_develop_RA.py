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

def TargetFunction(x, alphas, weights):
    tmp = 0
    for (alpha, weight) in zip(alphas,weights): 
        tmp += weight * x ** (1-alpha)
    return tmp 

if __name__ == "__main__":

    import sys
    # sys.path.append("/home/khristen/Projects/FDE/code/source/")
    # from MittagLeffler import ml

    T  = 1
    dt = 1.e-6
    Zmin, Zmax = 1/T,1/dt
    tol = 1.e-12
    nNodes = 500
    verbose = False
    alpha=0.1

    x = np.geomspace(dt, T, 10000)

    n = 1001
    alphas = np.linspace(0.975,1,n)
    success = np.zeros(n)

    start = time.time()

    for idx, alpha in enumerate(alphas): 
        print(alpha)
        try: 
            RA = RationalApproximation_AAA( alpha=alpha,
                                tol=tol, nSupportPoints=nNodes,
                                Zmin= Zmin, Zmax= Zmax,
                                verbose=verbose)
            success[idx]=1
        except: 
            print()

    end = time.time()
    print("Time elapsed", end - start, "seconds")

    df = pd.DataFrame(success)
    df.to_csv("success.csv")
    df = pd.DataFrame(alphas)
    df.to_csv("alphas.csv")


    plt.figure('Compare functions')    
    plt.plot(alphas, success,"g")
    plt.show()

    # y_RA = np.array([RA.appx_ker(z) for z in x])
    # plt.figure('Compare functions')    
    # plt.plot(x,y_RA,'r--', label="RA")
    # plt.plot(x,KernelFunction,'b-', label="KF")
    # plt.xscale('log')
    # plt.legend()
    # # plt.show()