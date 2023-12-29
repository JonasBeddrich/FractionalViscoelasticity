import os
import sys
import importlib
import glob
import time
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.RationalApproximation import RationalApproximation_AAA
from scipy.special import gamma 
import pandas as pd 


class MultiTermKernel: 

    def __init__(self, phis, alphas, **kwargs):
        self.phis = phis
        self.alphas = alphas

        self.RATarget = np.vectorize(self.eval_RATarget) 
        self.kernel = np.vectorize(self.eval_kernel)

    def eval_kernel(self, t):
        return np.sum(self.phis / gamma(1 - self.alphas) * t **(-self.alphas))
    
    def eval_RATarget(self, t):
        return np.sum(self.phis * t ** (1 - self.alphas))
    
# Distribution phi 
@np.vectorize
def Distribution(alpha): 
    return gamma(6 - alpha) / 120

# Right side of the equation 
@np.vectorize
def f(t): 
    return (t ** 5 - t**3) / np.log(t)


#########################################
### Main  
#########################################

if __name__ == "__main__":

    T  = 1
    dt = 1.e-6
    n_timesteps = int(T/dt) + 1
    t = dt * np.asarray(range(n_timesteps))

    #########################################
    ### Quadrature of the integral kernel 
    #########################################

    n_QF = 1000
    # interval length / (n for 0 to 1 + n for 1 to 2 + 3 for 0,1,2)
    QFweight = 2 / (2*n_QF+3)
    quadraturePoints01 = np.arange(0 + 0.5/n_QF,1 + 0.5/n_QF, 1/n_QF)
    quadraturePoints12 = np.arange(1 + 0.5/n_QF,2 + 0.5/n_QF, 1/n_QF)
    
    phis01 = Distribution(quadraturePoints01) * QFweight
    phis12 = Distribution(quadraturePoints12) * QFweight
    phi0 = Distribution(0) * QFweight
    phi1 = Distribution(1) * QFweight
    phi2 = Distribution(2) * QFweight

    MTK01 = MultiTermKernel(phis01,quadraturePoints01)
    MTK12 = MultiTermKernel(phis12,quadraturePoints12-1)

    #########################################
    ### Rational Approximation of the Integral Kernel
    #########################################

    Zmin, Zmax = 1/T,1/dt
    tol = 1.e-10
    MaxDegree = 30
    nSupportPoints = 100

    RA01 = RationalApproximation_AAA(alpha=0.5, tol=tol, MaxDegree=MaxDegree, nSupportPoints=nSupportPoints, Zmin= Zmin, Zmax= Zmax, verbose=False, TargetFunction=MTK01.RATarget)
    RA12 = RationalApproximation_AAA(alpha=0.5, tol=tol, MaxDegree=MaxDegree, nSupportPoints=nSupportPoints, Zmin= Zmin, Zmax= Zmax, verbose=False, TargetFunction=MTK12.RATarget)

    weights01 = RA01.c
    poles01 = RA01.d
    winf01 = RA01.c_inf
    m01 = weights01.shape

    weights12 = RA12.c
    poles12 = RA12.d
    winf12 = RA12.c_inf
    m12 = weights12.shape

    #########################################
    ### Numerical Scheme 
    #########################################

    beta0 = phi0 + np.sum(weights01) - np.sum(weights12 * poles12)
    beta1 = phi1 + np.sum(weights12) + winf01
    beta2 = phi2 + winf12

    modes01 = np.zeros((n_timesteps, m01[0])) 
    modes12 = np.zeros((n_timesteps, m12[0])) 

    u = np.zeros(n_timesteps) 
    v = np.zeros(n_timesteps) 

    # Initial conditions if necessary 
    u[0] = 0
    v[0] = 0 

    # Time loop 
    for n in tqdm(range(n_timesteps-1), ncols=100): 

        Q = np.sum(dt * weights01 * poles01 / (1 + dt * poles01)) - np.sum(dt * weights12 * poles12**2 / (1 + dt * poles12)) - beta0
        H = np.sum(poles01 / (1 + dt * poles01) * modes01[n]) - np.sum(poles12**2 / (1 + dt * poles12) * modes12[n])
        P = 1 - dt ** 2 / beta2 * Q + dt / beta2 * beta1
        
        v[n+1] = (v[n] + dt / beta2 * (f((n+1)*dt) + Q * u[n] + H)) / P 
        u[n+1] = u[n] + dt * v[n+1]

        modes01[n+1] = 1 / (1 + dt*poles01) * (modes01[n] + dt * weights01 * u[n+1])
        modes12[n+1] = 1 / (1 + dt*poles12) * (modes12[n] + dt * weights12 * u[n+1])

    #########################################
    ### Visualization  
    # #########################################

    # Rational Approximation 

    fig, ax = plt.subplots(2,3, figsize=(12,8))
    x = np.linspace(dt,1,1000)

    ax[0][0].semilogy(x,MTK01.kernel(x), color = "blue", linewidth=3)
    ax[0][0].semilogy(x,RA01.appx_ker(x), linestyle = ":", color = "red", linewidth=3)
    ax[0][0].set_title("Kernel for 0 to 1")

    ax[0][1].plot(x,MTK01.RATarget(x), color = "blue", linewidth=3)
    ax[0][1].plot(x,RA01.target_func(x), linestyle = ":", color = "red", linewidth=3)
    ax[0][1].set_title("Target function for 0 to 1")

    ax[0][2].semilogy(x,np.abs(MTK01.kernel(x)-RA01.appx_ker(x)), color = "black", linewidth=3)
    ax[0][2].set_title("Error for 0 to 1")
    
    ax[1][0].semilogy(x,MTK12.kernel(x), color = "darkgreen", linewidth=3)
    ax[1][0].semilogy(x,RA12.appx_ker(x), linestyle = ":", color = "orange", linewidth=3)
    ax[1][0].set_title("Kernel for 1 to 2")

    ax[1][1].plot(x,MTK12.RATarget(x), color = "darkgreen", linewidth=3)
    ax[1][1].plot(x,RA12.target_func(x), linestyle = ":", color = "orange", linewidth=3)
    ax[1][1].set_title("Target Function for 1 to 2")

    ax[1][2].semilogy(x,np.abs(MTK12.kernel(x)-RA12.appx_ker(x)), color = "black", linewidth=3)
    ax[1][2].set_title("Error for 1 to 2")

    # Numerical results 

    fig, ax = plt.subplots(1,3,figsize=(15,5))

    ax[0].plot(t,u, color = "blue", label = "Numerical Result", linewidth=3)
    ax[0].plot(t,t**5, linestyle=":", color = "red", label = "Analytical Solution", linewidth=3)
    ax[0].set_title("Solution")
    ax[0].legend()

    ax[1].plot(t,np.abs(u-t**5), color = "black", linewidth=3)
    ax[1].set_title("Absolute Error")

    ax[2].loglog(t,np.abs((u-t**5)/t**5), color = "navy", linewidth=3)
    ax[2].set_title("Relative Error")

    plt.show()    